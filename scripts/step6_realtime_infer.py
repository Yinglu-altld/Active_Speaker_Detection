import argparse
import asyncio
import base64
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from step3_extract_landmarks import (
    compute_oval_stats,
    indices_from_connections,
    normalize_points,
    passes_quality_filter,
)
from step5_train_cnn import TemporalCNN


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "cnn_vvad"
DEFAULT_WINDOWS_INFO = PROJECT_ROOT / "data" / "windows" / "windows_info.json"
DEFAULT_FURHAT_IP = "192.168.1.109"
DEFAULT_FURHAT_AUTH = None

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from audio_doa.fusion import FusionConfig, UserEvidence, score_users_for_frame
except Exception:
    FusionConfig = None
    UserEvidence = None
    score_users_for_frame = None


def _ensure_no_proxy(host: str | None) -> None:
    if not host:
        return
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    entries = [item.strip() for item in existing.split(",") if item.strip()]
    if host not in entries:
        entries.append(str(host))
    value = ",".join(entries) if entries else str(host)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


@dataclass
class ModelSpec:
    model_path: Path
    threshold: float
    window_frames: int
    target_fps: float | None
    use_delta: bool
    indices: list[int]
    feature_type: str


@dataclass
class TrackState:
    buffer: deque
    ema_prob: float | None = None
    seen: int = 0
    is_speaking: bool | None = None


@dataclass
class UserBox:
    user_id: str
    x: int
    y: int
    w: int
    h: int
    location_x: float | None = None
    location_z: float | None = None


class FrameRateLimiter:
    def __init__(self, target_fps: float | None):
        self.target_fps = target_fps
        self.last_time = 0.0

    def should_process(self, now: float) -> bool:
        if not self.target_fps or self.target_fps <= 0:
            return True
        interval = 1.0 / self.target_fps
        if now - self.last_time < interval:
            return False
        self.last_time = now
        return True


class StreamSource:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        self.cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 6: Real-time inference from a Furhat camera stream."
    )
    parser.add_argument(
        "--source",
        choices=["furhat", "opencv", "file", "stream"],
        default="furhat",
    )
    parser.add_argument(
        "--furhat-ip",
        default=DEFAULT_FURHAT_IP,
        help="Furhat realtime API host/IP (e.g. 127.0.0.1).",
    )
    parser.add_argument(
        "--furhat-auth",
        default=DEFAULT_FURHAT_AUTH,
        help="Authentication key for Furhat realtime API (if required).",
    )
    parser.add_argument(
        "--furhat-url",
        default=None,
        help="Deprecated alias for --stream-url (OpenCV-compatible stream URL).",
    )
    parser.add_argument(
        "--stream-url",
        default=None,
        help="OpenCV-compatible stream URL (e.g. MJPEG/RTSP).",
    )
    parser.add_argument("--video-device", type=int, default=0)
    parser.add_argument("--video-file", default=None)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--face-model-path",
        default=str(PROJECT_ROOT / "data" / "models" / "face_landmarker_v2.task"),
    )
    parser.add_argument("--windows-info", default=str(DEFAULT_WINDOWS_INFO))
    parser.add_argument("--window-frames", type=int, default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--delegate",
        choices=["cpu", "gpu"],
        default="cpu",
        help="MediaPipe delegate selection. Use cpu for headless environments.",
    )
    parser.add_argument("--max-faces", type=int, default=1)
    parser.add_argument("--min-oval-size", type=float, default=0.2)
    parser.add_argument("--max-oval-size", type=float, default=0.98)
    parser.add_argument("--edge-margin", type=float, default=0.02)
    parser.add_argument("--min-det-conf", type=float, default=0.5)
    parser.add_argument("--min-presence-conf", type=float, default=0.5)
    parser.add_argument("--min-track-conf", type=float, default=0.5)
    parser.add_argument("--infer-stride", type=int, default=1)
    parser.add_argument("--ema", type=float, default=0.0)
    parser.add_argument(
        "--bbox-padding",
        type=float,
        default=0.2,
        help="Padding ratio to expand Furhat user camera bbox before cropping.",
    )
    parser.add_argument(
        "--min-bbox-area-ratio",
        type=float,
        default=0.02,
        help="If bbox area / frame area is below this, upsample the crop.",
    )
    parser.add_argument(
        "--min-bbox-side",
        type=int,
        default=160,
        help="Upsample small bboxes so their min side reaches this size (pixels).",
    )
    parser.add_argument(
        "--upsample-small-faces",
        action="store_true",
        help="Enable conditional upsampling for small face crops.",
    )
    parser.add_argument(
        "--speak-on-th",
        type=float,
        default=None,
        help="Optional threshold to turn speaking ON (defaults to model threshold).",
    )
    parser.add_argument(
        "--speak-off-th",
        type=float,
        default=None,
        help="Optional threshold to turn speaking OFF (defaults to model threshold).",
    )
    parser.add_argument(
        "--emit-cnn-jsonl",
        default=None,
        help="Optional output path for fusion snapshots (JSONL).",
    )
    parser.add_argument(
        "--camera-hfov-deg",
        type=float,
        default=60.0,
        help="Approximate camera horizontal FOV for bbox-to-bearing mapping.",
    )
    parser.add_argument(
        "--doa-jsonl-live",
        default=None,
        help="Optional live DOA JSONL path for realtime fusion overlay in --show mode.",
    )
    parser.add_argument(
        "--max-doa-staleness-sec",
        type=float,
        default=0.8,
        help="Max |t_frame - t_doa| to accept DOA for fusion overlay.",
    )
    parser.add_argument("--fusion-default-sigma-deg", type=float, default=25.0)
    parser.add_argument("--fusion-min-sigma-deg", type=float, default=8.0)
    parser.add_argument("--fusion-min-doa-reliability", type=float, default=0.05)
    parser.add_argument("--fusion-doa-reliability-gamma", type=float, default=0.65)
    parser.add_argument(
        "--overlay-gate-speak-by-audio",
        action="store_true",
        default=True,
        help="In fusion overlay mode, force displayed speak=0 when DOA is missing or audio says no speech.",
    )
    parser.add_argument(
        "--no-overlay-gate-speak-by-audio",
        action="store_false",
        dest="overlay_gate_speak_by_audio",
        help="Disable audio gating of displayed speak flag in fusion overlay mode.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-runtime-sec",
        type=float,
        default=None,
        help="Optional wall-clock limit for smoke tests (seconds).",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
    parser.add_argument(
        "--score-plot",
        action="store_true",
        default=True,
        help="Show raw CNN/DOA score plot in the realtime UI.",
    )
    parser.add_argument(
        "--no-score-plot",
        action="store_false",
        dest="score_plot",
        help="Disable raw CNN/DOA score plot in the realtime UI.",
    )
    parser.add_argument(
        "--score-plot-window-sec",
        type=float,
        default=12.0,
        help="History span (seconds) for the raw CNN/DOA score plot.",
    )
    parser.add_argument(
        "--no-draw-landmarks",
        dest="draw_landmarks",
        action="store_false",
        help="Disable landmark drawing (enabled by default).",
    )
    parser.set_defaults(draw_landmarks=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_model_spec(args: argparse.Namespace) -> ModelSpec:
    model_dir = Path(args.model_dir)
    model_path = Path(args.model_path) if args.model_path else (model_dir / "best.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    config = load_json(model_dir / "config.json")
    windows_info = load_json(Path(args.windows_info))

    indices = windows_info.get("indices")
    feature_type = str(windows_info.get("feature_type", "nx"))
    window_frames = args.window_frames or windows_info.get("window_frames")
    if window_frames is None:
        raise ValueError("window_frames not found. Provide --window-frames.")

    target_fps = args.target_fps
    if target_fps is None:
        target_fps = windows_info.get("target_fps")

    threshold = args.threshold
    if threshold is None:
        threshold_data = load_json(model_dir / "threshold.json")
        threshold = float(threshold_data.get("threshold", 0.5))

    use_delta = bool(config.get("use_delta", True))

    if not indices:
        from mediapipe.tasks.python.vision import face_landmarker

        lips = face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_LIPS
        oval = face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
        indices = sorted(
            set(indices_from_connections(lips)) | set(indices_from_connections(oval))
        )

    return ModelSpec(
        model_path=model_path,
        threshold=float(threshold),
        window_frames=int(window_frames),
        target_fps=float(target_fps) if target_fps is not None else None,
        use_delta=use_delta,
        indices=[int(i) for i in indices],
        feature_type=feature_type,
    )


def build_model(
    num_points: int, use_delta: bool, device: torch.device, model_path: Path
) -> torch.nn.Module:
    num_channels = 4 if use_delta else 2
    model = TemporalCNN(num_points=num_points, num_channels=num_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def append_delta(window: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(window)
    delta[1:] = window[1:] - window[:-1]
    return np.concatenate([window, delta], axis=2)


def extract_features(
    landmarks,
    indices: list[int],
    oval_indices: list[int],
    feature_type: str,
    min_oval_size: float,
    max_oval_size: float,
    edge_margin: float,
) -> np.ndarray | None:
    if not landmarks:
        return None
    max_idx = max(max(indices, default=0), max(oval_indices, default=0))
    if max_idx >= len(landmarks):
        return None

    points = np.array(
        [(landmarks[i].x, landmarks[i].y) for i in indices], dtype=np.float32
    )
    oval_xy = np.array(
        [(landmarks[i].x, landmarks[i].y) for i in oval_indices], dtype=np.float32
    )

    min_xy, max_xy, width, height, cx, cy, scale = compute_oval_stats(oval_xy)
    if not passes_quality_filter(
        min_xy,
        max_xy,
        width,
        height,
        edge_margin,
        min_oval_size,
        max_oval_size,
    ):
        return None

    if feature_type == "nx":
        normalized = normalize_points(
            points, np.array([cx, cy], dtype=np.float32), scale
        )
        if normalized is None:
            return None
        return normalized

    return points


def infer_window(
    window: deque,
    model: torch.nn.Module,
    device: torch.device,
    use_delta: bool,
) -> float:
    arr = np.stack(list(window), axis=0).astype(np.float32)
    if use_delta:
        arr = append_delta(arr)
    x = torch.from_numpy(arr).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    return float(prob)


def prob_color(prob: float) -> tuple[int, int, int]:
    if prob < 0.4:
        return (0, 0, 255)
    if prob < 0.8:
        return (0, 255, 255)
    return (0, 255, 0)


def _fmt_float(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _wrap_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _render_score_plot_frame(
    plot_payload: dict | None,
    width: int = 420,
    height: int = 220,
) -> np.ndarray:
    canvas = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    canvas[:, :] = (20, 20, 20)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (70, 70, 70), 1)

    points = [] if not plot_payload else list(plot_payload.get("points") or [])
    chart_l = 10
    chart_r = width - 10
    chart_t = 44
    chart_b = height - 16
    chart_w = max(1, chart_r - chart_l)
    chart_h = max(1, chart_b - chart_t)
    cv2.rectangle(canvas, (chart_l, chart_t), (chart_r, chart_b), (45, 45, 45), 1)

    doa_stale = bool(plot_payload and plot_payload.get("doa_stale"))
    doa_color = (95, 95, 95) if doa_stale else (0, 200, 255)
    doa_text_color = (150, 150, 150) if doa_stale else (0, 200, 255)
    if len(points) >= 2:
        def _draw_series(key: str, color: tuple[int, int, int]) -> None:
            segment: list[tuple[int, int]] = []
            n_points = len(points)
            for idx, item in enumerate(points):
                value = item.get(key)
                if value is None:
                    if len(segment) >= 2:
                        cv2.polylines(
                            canvas,
                            [np.array(segment, dtype=np.int32)],
                            isClosed=False,
                            color=color,
                            thickness=2,
                            lineType=cv2.LINE_AA,
                        )
                    segment = []
                    continue
                x = int(chart_l + (idx / max(1, n_points - 1)) * chart_w)
                y = int(chart_b - _clip01(float(value)) * chart_h)
                segment.append((x, y))
            if len(segment) >= 2:
                cv2.polylines(
                    canvas,
                    [np.array(segment, dtype=np.int32)],
                    isClosed=False,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

        _draw_series("cnn", (0, 220, 0))
        _draw_series("doa", doa_color)
    else:
        cv2.putText(
            canvas,
            "waiting for score history...",
            (14, int((chart_t + chart_b) * 0.5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 180, 180),
            1,
            lineType=cv2.LINE_AA,
        )

    target_id = None if not plot_payload else plot_payload.get("track_id")
    latest_cnn = None if not plot_payload else plot_payload.get("cnn")
    latest_doa = None if not plot_payload else plot_payload.get("doa")
    doa_age_sec = None if not plot_payload else plot_payload.get("doa_age_sec")
    legend = (
        f"raw plot id:{target_id if target_id is not None else '-'} "
        f"cnn:{_fmt_float(latest_cnn)} doa:{_fmt_float(latest_doa)}"
    )
    cv2.putText(
        canvas,
        legend,
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (220, 220, 220),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "cnn(raw)",
        (10, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (0, 220, 0),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "doa(raw stale)" if doa_stale else "doa(raw)",
        (88, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        doa_text_color,
        1,
        lineType=cv2.LINE_AA,
    )
    if doa_age_sec is not None:
        age_color = (120, 120, 255) if doa_stale else (190, 190, 190)
        cv2.putText(
            canvas,
            f"doa_age:{_fmt_float(doa_age_sec)}s",
            (width - 140, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            age_color,
            1,
            lineType=cv2.LINE_AA,
        )
    return canvas


def _draw_status_hud(frame, hud: dict | None) -> None:
    if not hud:
        return
    mode = str(hud.get("mode", "UNKNOWN"))
    speech_active = bool(hud.get("speech_active", False))
    face_count = int(hud.get("face_count", 0))
    speaker_id = hud.get("speaker_id")
    speaker_score = hud.get("speaker_score")
    doa = hud.get("doa") or {}
    azimuth = doa.get("azimuth_deg")
    reliability = doa.get("reliability")
    conf_geom = doa.get("conf_doa_srp")
    audio_conf = doa.get("audio_conf")
    sigma_deg = doa.get("sigma_deg")

    lines = [
        (
            f"mode:{mode} speech:{int(speech_active)} faces:{face_count} "
            f"speaker:{speaker_id if speaker_id is not None else '-'} "
            f"score:{_fmt_float(speaker_score)}"
        ),
        (
            f"doa az:{_fmt_float(azimuth)} rel:{_fmt_float(reliability)} "
            f"geom:{_fmt_float(conf_geom)} audio:{_fmt_float(audio_conf)} "
            f"sigma:{_fmt_float(sigma_deg)}"
        ),
    ]
    box_h = 22 * len(lines) + 12
    cv2.rectangle(frame, (8, 8), (frame.shape[1] - 8, 8 + box_h), (20, 20, 20), -1)
    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        y += 22


def draw_overlays(frame, outputs, hud: dict | None = None) -> None:
    _draw_status_hud(frame, hud)
    y = 24
    for out in outputs:
        track_id = out["track_id"]
        prob_out = out["prob"]
        speak = out["speak"]
        bbox = out.get("bbox")
        overall = out.get("fusion_overall")
        score_cnn = out.get("fusion_cnn")
        score_doa = out.get("fusion_doa")
        is_active = bool(out.get("active_speaker"))
        no_speech = bool(out.get("no_speech"))
        bearing = out.get("bearing_deg")
        if overall is None:
            top_lines = [
                f"{track_id} cnn:{prob_out:.2f}",
                f"speak:{int(speak)}",
            ]
            bottom_lines = [
                f"a:{int(is_active)}",
                f"b:{_fmt_float(bearing)}",
            ]
        else:
            top_lines = [
                f"{track_id} c:{float(score_cnn):.2f}",
                f"d:{float(score_doa):.2f} o:{float(overall):.2f}",
            ]
            bottom_lines = [
                f"a:{int(is_active)} speak:{int(speak)}",
                f"b:{_fmt_float(bearing)}",
            ]
        if no_speech:
            color = (0, 0, 255)
        elif is_active:
            color = (0, 255, 0)
        elif overall is not None:
            color = (0, 165, 255)
        else:
            color = prob_color(prob_out)
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text_x = x1
            line_gap = 20
            h = frame.shape[0]

            # Two lines above the bbox.
            top_last_y = max(18, y1 - 6)
            top_first_y = top_last_y - line_gap * (len(top_lines) - 1)
            if top_first_y < 18:
                top_first_y = 18
            for i, line in enumerate(top_lines):
                text_y = top_first_y + i * line_gap
                cv2.putText(
                    frame,
                    line,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )

            # Two lines below the bbox.
            bottom_first_y = min(h - 8 - line_gap * (len(bottom_lines) - 1), y2 + line_gap)
            for i, line in enumerate(bottom_lines):
                text_y = bottom_first_y + i * line_gap
                cv2.putText(
                    frame,
                    line,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )
        else:
            for line in top_lines + bottom_lines:
                cv2.putText(
                    frame,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )
                y += 22


def draw_landmark_points(frame, landmarks, indices, color=(0, 255, 0)) -> None:
    h, w = frame.shape[:2]
    for i in indices:
        if i >= len(landmarks):
            continue
        pt = landmarks[i]
        x = int(pt.x * w)
        y = int(pt.y * h)
        cv2.circle(frame, (x, y), 1, color, -1, lineType=cv2.LINE_AA)


def draw_landmark_points_offset(
    frame, landmarks, indices, x1: int, y1: int, w: int, h: int, color=(0, 255, 0)
) -> None:
    for i in indices:
        if i >= len(landmarks):
            continue
        pt = landmarks[i]
        x = int(x1 + pt.x * w)
        y = int(y1 + pt.y * h)
        cv2.circle(frame, (x, y), 1, color, -1, lineType=cv2.LINE_AA)


def clamp_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int):
    x1 = max(0, int(np.floor(x1)))
    y1 = max(0, int(np.floor(y1)))
    x2 = min(w, int(np.ceil(x2)))
    y2 = min(h, int(np.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expand_bbox(x: int, y: int, w: int, h: int, pad: float):
    pad_x = w * pad
    pad_y = h * pad
    return x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y


def maybe_upsample_crop(
    crop: np.ndarray, bbox_w: int, bbox_h: int, frame_w: int, frame_h: int, args: argparse.Namespace
) -> np.ndarray:
    area_ratio = (bbox_w * bbox_h) / max(1.0, float(frame_w * frame_h))
    if area_ratio >= args.min_bbox_area_ratio:
        return crop
    min_side = max(1, min(bbox_w, bbox_h))
    if min_side >= args.min_bbox_side:
        return crop
    scale = args.min_bbox_side / float(min_side)
    new_w = int(round(bbox_w * scale))
    new_h = int(round(bbox_h * scale))
    if new_w <= 0 or new_h <= 0:
        return crop
    return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def location_to_bearing_deg(location_x: float | None, location_z: float | None) -> float | None:
    if location_x is None or location_z is None:
        return None
    if abs(location_x) < 1e-6 and abs(location_z) < 1e-6:
        return None
    # Canonical bearing convention in this project:
    # front=0, right=+90, left=-90 with +x meaning right.
    return float(np.degrees(np.arctan2(location_x, location_z)))


def bbox_to_bearing_deg(
    bbox: tuple[int, int, int, int] | None,
    frame_width: int,
    hfov_deg: float,
) -> float | None:
    if bbox is None or frame_width <= 1:
        return None
    x1, _, x2, _ = bbox
    cx = 0.5 * (x1 + x2)
    norm = (cx / float(frame_width)) - 0.5
    return float(norm * float(hfov_deg))


def estimate_user_bearing_deg(
    user: UserBox,
    bbox: tuple[int, int, int, int] | None,
    frame_width: int,
    hfov_deg: float,
) -> float | None:
    # Prefer camera-image geometry for fusion bearing; it is usually more
    # left/right symmetric than robot-world user location estimates.
    bearing = bbox_to_bearing_deg(bbox, frame_width, hfov_deg)
    if bearing is None:
        bearing = location_to_bearing_deg(user.location_x, user.location_z)
    if bearing is None:
        return None
    return float(bearing)


def emit_cnn_snapshot(handle, t_value: float, outputs: list[dict]) -> None:
    users = []
    for out in outputs:
        bearing = out.get("bearing_deg")
        if bearing is None:
            continue
        users.append(
            {
                "user_id": str(out["track_id"]),
                "bearing_deg": float(bearing),
                "cnn_prob": float(out["prob"]),
                "speak": bool(out["speak"]),
                "face_conf": 1.0,
                "track_conf": 1.0,
            }
        )
    if not users:
        return
    handle.write(json.dumps({"t": float(t_value), "users": users}) + "\n")
    handle.flush()


class LiveDOASnapshots:
    def __init__(self, path: str):
        self.path = path
        self.handle = None
        self.offset = 0
        self.latest = None

    def _ensure_handle(self) -> None:
        if self.handle is not None:
            return
        if not os.path.exists(self.path):
            return
        self.handle = open(self.path, "r", encoding="utf-8")
        self.handle.seek(self.offset)

    def poll(self) -> None:
        if os.path.exists(self.path) and os.path.getsize(self.path) < self.offset:
            if self.handle is not None:
                self.handle.close()
            self.handle = None
            self.offset = 0
            self.latest = None

        self._ensure_handle()
        if self.handle is None:
            return

        while True:
            line = self.handle.readline()
            if not line:
                break
            self.offset = self.handle.tell()
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            if item.get("t") is None:
                continue
            self.latest = item

    def get_for_time(self, t_value: float, max_staleness_sec: float):
        self.poll()
        if self.latest is None:
            return None
        t_obs = self.latest.get("t")
        if t_obs is None:
            return None
        if abs(float(t_value) - float(t_obs)) > float(max_staleness_sec):
            return None
        return dict(self.latest)

    def latest_snapshot(self):
        self.poll()
        if self.latest is None:
            return None
        return dict(self.latest)


class FusionOverlayRuntime:
    def __init__(self, args: argparse.Namespace):
        if score_users_for_frame is None or FusionConfig is None or UserEvidence is None:
            raise ImportError(
                "audio_doa.fusion is required for DOA fusion overlay. Ensure repo root is importable."
            )
        self.live_doa = LiveDOASnapshots(args.doa_jsonl_live)
        self.max_staleness_sec = float(args.max_doa_staleness_sec)
        self.cfg = FusionConfig(
            min_sigma_deg=float(args.fusion_min_sigma_deg),
            default_sigma_deg=float(args.fusion_default_sigma_deg),
            min_reliability_for_doa=float(args.fusion_min_doa_reliability),
            reliability_gamma=float(args.fusion_doa_reliability_gamma),
            fixed_doa_weight=0.35,
        )
        self.overlay_gate_speak_by_audio = bool(args.overlay_gate_speak_by_audio)
        self.score_plot_enabled = bool(args.score_plot)
        self.score_plot_window_sec = max(2.0, float(args.score_plot_window_sec))
        self.score_plot_history = deque()

    def _select_plot_target(self, outputs: list[dict], preferred_id: str | None):
        if not outputs:
            return None
        if preferred_id is not None:
            for out in outputs:
                if str(out.get("track_id")) == str(preferred_id):
                    return out
        return max(outputs, key=lambda item: float(item.get("prob") or 0.0))

    def _raw_doa_plot_score(self, doa_obs: dict | None, bearing_deg: float | None) -> float | None:
        if doa_obs is None or bearing_deg is None:
            return None
        azimuth_deg = doa_obs.get("azimuth_deg")
        if azimuth_deg is None:
            return None
        sigma_deg = doa_obs.get("sigma_deg")
        if sigma_deg is None:
            sigma = float(self.cfg.default_sigma_deg)
        else:
            sigma = max(float(self.cfg.min_sigma_deg), float(sigma_deg))
        delta = abs(_wrap_angle_deg(float(azimuth_deg) - float(bearing_deg)))
        return float(math.exp(-0.5 * (delta / max(1e-6, sigma)) ** 2))

    def _append_score_plot_point(
        self,
        t_value: float,
        outputs: list[dict],
        preferred_id: str | None,
        doa_plot_obs: dict | None,
    ) -> None:
        if not self.score_plot_enabled:
            return
        target = self._select_plot_target(outputs, preferred_id)
        track_id = None if target is None else str(target.get("track_id"))
        cnn_raw = None if target is None else _clip01(float(target.get("prob") or 0.0))
        bearing = None if target is None else target.get("bearing_deg")
        doa_raw = self._raw_doa_plot_score(doa_plot_obs, bearing)

        doa_t = None
        doa_rel = None
        if doa_plot_obs is not None:
            t_obs = doa_plot_obs.get("t")
            if t_obs is not None:
                try:
                    doa_t = float(t_obs)
                except (TypeError, ValueError):
                    doa_t = None
            doa_rel = 0.6 * float(doa_plot_obs.get("conf_doa_srp") or 0.0) + 0.4 * float(
                doa_plot_obs.get("audio_conf") or 0.0
            )
            doa_rel = _clip01(doa_rel)

        self.score_plot_history.append(
            {
                "t": float(t_value),
                "track_id": track_id,
                "cnn": cnn_raw,
                "doa": doa_raw,
                "doa_t": doa_t,
                "doa_reliability": doa_rel,
            }
        )
        cutoff = float(t_value) - float(self.score_plot_window_sec)
        while self.score_plot_history and float(self.score_plot_history[0]["t"]) < cutoff:
            self.score_plot_history.popleft()

    def _attach_score_plot(self, hud: dict, t_value: float) -> dict:
        if not self.score_plot_enabled or not self.score_plot_history:
            return hud
        latest = self.score_plot_history[-1]
        doa_age_sec = None
        doa_t = latest.get("doa_t")
        if doa_t is not None:
            doa_age_sec = max(0.0, float(t_value) - float(doa_t))
        stale_threshold_sec = float(self.max_staleness_sec)
        doa_stale = True if doa_age_sec is None else doa_age_sec > stale_threshold_sec
        hud["score_plot"] = {
            "track_id": latest.get("track_id"),
            "cnn": latest.get("cnn"),
            "doa": latest.get("doa"),
            "doa_reliability": latest.get("doa_reliability"),
            "doa_age_sec": doa_age_sec,
            "doa_stale": bool(doa_stale),
            "doa_stale_threshold_sec": stale_threshold_sec,
            "points": [
                {
                    "cnn": item.get("cnn"),
                    "doa": item.get("doa"),
                }
                for item in self.score_plot_history
            ],
        }
        return hud

    def annotate_outputs(self, outputs: list[dict], t_value: float) -> dict:
        doa_plot_obs = self.live_doa.latest_snapshot()
        doa_obs = None
        if doa_plot_obs is not None:
            t_obs = doa_plot_obs.get("t")
            if t_obs is not None:
                try:
                    if abs(float(t_value) - float(t_obs)) <= float(self.max_staleness_sec):
                        doa_obs = doa_plot_obs
                except (TypeError, ValueError):
                    doa_obs = None
        for out in outputs:
            out["visual_speak"] = bool(out.get("speak", False))
        hud = {
            "mode": "DOA_MISSING" if doa_obs is None else "SPEECH_AV",
            "speech_active": False if doa_obs is None else bool(doa_obs.get("speech_active") or doa_obs.get("speech_detected")),
            "face_count": len(outputs),
            "speaker_id": None,
            "speaker_score": None,
            "weights": {"cnn": 1.0, "doa": 0.0},
            "doa": {
                "azimuth_deg": None if doa_obs is None else doa_obs.get("azimuth_deg"),
                "conf_doa_srp": None if doa_obs is None else doa_obs.get("conf_doa_srp"),
                "audio_conf": None if doa_obs is None else doa_obs.get("audio_conf"),
                "sigma_deg": None if doa_obs is None else doa_obs.get("sigma_deg"),
                "reliability": None if doa_obs is None else 0.6 * float(doa_obs.get("conf_doa_srp") or 0.0) + 0.4 * float(doa_obs.get("audio_conf") or 0.0),
            },
        }

        if doa_obs is None:
            for out in outputs:
                out["fusion_overall"] = None
                out["fusion_cnn"] = float(out["prob"])
                out["fusion_doa"] = 0.0
                out["active_speaker"] = False
                out["no_speech"] = False
                if self.overlay_gate_speak_by_audio:
                    out["speak"] = False
            self._append_score_plot_point(t_value, outputs, None, doa_plot_obs)
            return self._attach_score_plot(hud, t_value)

        speech_active = bool(doa_obs.get("speech_active") or doa_obs.get("speech_detected"))
        if not speech_active:
            hud["mode"] = "NO_SPEECH"
            for out in outputs:
                out["fusion_overall"] = None
                out["fusion_cnn"] = float(out["prob"])
                out["fusion_doa"] = 0.0
                out["active_speaker"] = False
                out["no_speech"] = True
                if self.overlay_gate_speak_by_audio:
                    out["speak"] = False
            self._append_score_plot_point(t_value, outputs, None, doa_plot_obs)
            return self._attach_score_plot(hud, t_value)

        users = []
        for out in outputs:
            bearing_deg = out.get("bearing_deg")
            if bearing_deg is None:
                continue
            users.append(
                UserEvidence(
                    user_id=str(out["track_id"]),
                    bearing_deg=float(bearing_deg),
                    cnn_prob=float(out["prob"]),
                    face_conf=1.0,
                    track_conf=1.0,
                )
            )

        if not users:
            hud["mode"] = "SPEECH_CNN_ONLY"
            if outputs:
                top = max(outputs, key=lambda item: float(item["prob"]))
                hud["speaker_id"] = str(top["track_id"])
                hud["speaker_score"] = float(top["prob"])
            for out in outputs:
                out["fusion_overall"] = None
                out["fusion_cnn"] = float(out["prob"])
                out["fusion_doa"] = 0.0
                out["active_speaker"] = False
                out["no_speech"] = False
            self._append_score_plot_point(t_value, outputs, hud.get("speaker_id"), doa_plot_obs)
            return self._attach_score_plot(hud, t_value)

        result = score_users_for_frame(
            doa_obs=doa_obs,
            users=users,
            cfg=self.cfg,
        )
        hud["mode"] = "SPEECH_AV"
        active_id = result.get("speaker_id")
        hud["speaker_id"] = active_id
        hud["speaker_score"] = result.get("speaker_score")
        hud["weights"] = result.get("weights", hud["weights"])
        hud["doa"] = result.get("doa", hud["doa"])
        score_map = {str(item["user_id"]): item for item in result.get("per_user", [])}
        for out in outputs:
            row = score_map.get(str(out["track_id"]))
            if row is None:
                out["fusion_overall"] = None
                out["fusion_cnn"] = float(out["prob"])
                out["fusion_doa"] = 0.0
                out["active_speaker"] = False
                out["no_speech"] = False
                continue
            out["fusion_overall"] = float(row["score"])
            out["fusion_cnn"] = float(row["score_cnn"])
            out["fusion_doa"] = float(row["score_doa"])
            out["active_speaker"] = str(out["track_id"]) == str(active_id)
            out["no_speech"] = False
        self._append_score_plot_point(t_value, outputs, str(active_id) if active_id is not None else None, doa_plot_obs)
        return self._attach_score_plot(hud, t_value)


def process_frame(
    frame: np.ndarray,
    landmarker,
    image_module,
    oval_indices: list[int],
    states: dict[str, TrackState],
    spec: ModelSpec,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    limiter: FrameRateLimiter,
    print_state: dict,
    user_boxes: list[UserBox] | None,
    cnn_jsonl_handle=None,
    fusion_runtime: FusionOverlayRuntime | None = None,
) -> bool:
    now = time.time()
    if not limiter.should_process(now):
        return True

    outputs = []
    speak_on = args.speak_on_th if args.speak_on_th is not None else spec.threshold
    speak_off = args.speak_off_th if args.speak_off_th is not None else spec.threshold
    if user_boxes is None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = image_module.Image(
            image_module.ImageFormat.SRGB, np.ascontiguousarray(frame_rgb)
        )
        result = landmarker.detect(mp_image)
        faces = result.face_landmarks or []
        for idx, face_landmarks in enumerate(faces):
            if args.draw_landmarks:
                draw_landmark_points(
                    frame, face_landmarks, spec.indices, color=(0, 255, 0)
                )
            features = extract_features(
                face_landmarks,
                spec.indices,
                oval_indices,
                spec.feature_type,
                args.min_oval_size,
                args.max_oval_size,
                args.edge_margin,
            )
            if features is None:
                continue
            outputs.extend(
                update_track_state(
                    f"face_{idx}",
                    features,
                    states,
                    spec,
                    model,
                    device,
                    args,
                    speak_on,
                    speak_off,
                    None,
                    None,
                )
            )
    else:
        if not user_boxes:
            if args.show and int(now) != print_state["last_print"]:
                print_state["last_print"] = int(now)
                print("no users")
            if args.show:
                if args.score_plot:
                    cv2.imshow("score_plot_raw", _render_score_plot_frame(None))
                cv2.imshow("vvad_realtime", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return False
            return True
        frame_h, frame_w = frame.shape[:2]
        for user in user_boxes:
            x1, y1, x2, y2 = expand_bbox(
                user.x, user.y, user.w, user.h, args.bbox_padding
            )
            bbox = clamp_bbox(x1, y1, x2, y2, frame_w, frame_h)
            if bbox is None:
                continue
            x1i, y1i, x2i, y2i = bbox
            crop = frame[y1i:y2i, x1i:x2i]
            if crop.size == 0:
                continue
            if args.upsample_small_faces:
                crop = maybe_upsample_crop(
                    crop,
                    x2i - x1i,
                    y2i - y1i,
                    frame_w,
                    frame_h,
                    args,
                )
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_image = image_module.Image(
                image_module.ImageFormat.SRGB, np.ascontiguousarray(crop_rgb)
            )
            result = landmarker.detect(crop_image)
            faces = result.face_landmarks or []
            if not faces:
                continue
            face_landmarks = faces[0]
            if args.draw_landmarks:
                draw_landmark_points_offset(
                    frame,
                    face_landmarks,
                    spec.indices,
                    x1i,
                    y1i,
                    x2i - x1i,
                    y2i - y1i,
                    color=(0, 255, 0),
                )
            features = extract_features(
                face_landmarks,
                spec.indices,
                oval_indices,
                spec.feature_type,
                args.min_oval_size,
                args.max_oval_size,
                args.edge_margin,
            )
            if features is None:
                continue
            outputs.extend(
                update_track_state(
                    user.user_id,
                    features,
                    states,
                    spec,
                    model,
                    device,
                    args,
                    speak_on,
                    speak_off,
                    (x1i, y1i, x2i, y2i),
                    estimate_user_bearing_deg(
                        user,
                        (x1i, y1i, x2i, y2i),
                        frame_w,
                        args.camera_hfov_deg,
                    ),
                )
            )

    hud = None
    if fusion_runtime is not None:
        hud = fusion_runtime.annotate_outputs(outputs, now)

    if outputs:
        for out in outputs:
            if out.get("fusion_overall") is None:
                print(f"{out['track_id']} {out['prob']:.3f} {int(out['speak'])}")
            else:
                print(
                    f"{out['track_id']} cnn={float(out['fusion_cnn']):.3f} "
                    f"doa={float(out['fusion_doa']):.3f} "
                    f"overall={float(out['fusion_overall']):.3f} "
                    f"active={int(bool(out.get('active_speaker')))}"
                )
        if cnn_jsonl_handle is not None:
            emit_cnn_snapshot(cnn_jsonl_handle, now, outputs)
    elif args.show and int(now) != print_state["last_print"]:
        print_state["last_print"] = int(now)
        print("no face")

    if args.show:
        draw_overlays(frame, outputs, hud)
        if args.score_plot:
            plot_payload = None if hud is None else hud.get("score_plot")
            cv2.imshow("score_plot_raw", _render_score_plot_frame(plot_payload))
        cv2.imshow("vvad_realtime", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def update_track_state(
    track_id: str,
    features: np.ndarray,
    states: dict[str, TrackState],
    spec: ModelSpec,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    speak_on: float,
    speak_off: float,
    bbox: tuple[int, int, int, int] | None = None,
    bearing_deg: float | None = None,
) -> list[dict]:
    state = states.get(track_id)
    if state is None:
        state = TrackState(buffer=deque(maxlen=spec.window_frames))
        states[track_id] = state

    state.buffer.append(features)
    state.seen += 1
    if len(state.buffer) < spec.window_frames:
        return []

    if args.infer_stride > 1 and (state.seen % args.infer_stride) != 0:
        return []

    prob = infer_window(state.buffer, model, device, spec.use_delta)
    if args.ema > 0:
        if state.ema_prob is None:
            state.ema_prob = prob
        else:
            state.ema_prob = args.ema * prob + (1.0 - args.ema) * state.ema_prob
        prob_out = state.ema_prob
    else:
        prob_out = prob

    if state.is_speaking is None:
        state.is_speaking = prob_out >= speak_on
    elif state.is_speaking:
        if prob_out < speak_off:
            state.is_speaking = False
    else:
        if prob_out >= speak_on:
            state.is_speaking = True

    return [
        {
            "track_id": track_id,
            "prob": float(prob_out),
            "speak": bool(state.is_speaking),
            "bbox": bbox,
            "bearing_deg": bearing_deg,
        }
    ]


async def run_furhat_stream(
    args: argparse.Namespace,
    spec: ModelSpec,
    model: torch.nn.Module,
    landmarker,
    image_module,
    oval_indices: list[int],
    device: torch.device,
    limiter: FrameRateLimiter,
    cnn_jsonl_handle=None,
    fusion_runtime: FusionOverlayRuntime | None = None,
) -> None:
    try:
        from furhat_realtime_api import AsyncFurhatClient, Events
    except ImportError as exc:
        raise ImportError(
            "furhat-realtime-api is required for --source=furhat. "
            "Install with: pip install furhat-realtime-api"
        ) from exc

    if not args.furhat_ip:
        raise ValueError("--furhat-ip is required when --source=furhat")
    _ensure_no_proxy(args.furhat_ip)

    frame_queue: deque[np.ndarray] = deque(maxlen=1)
    latest_users: list[UserBox] = []

    async def on_camera(event):
        image_b64 = None
        if isinstance(event, dict):
            image_b64 = event.get("image")
        else:
            image_b64 = getattr(event, "image", None)
        if not image_b64:
            return
        try:
            data = base64.b64decode(image_b64)
            arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame_queue.append(frame)
        except Exception:
            return

    async def on_users(event):
        nonlocal latest_users
        users = None
        if isinstance(event, dict):
            users = event.get("users")
        else:
            users = getattr(event, "users", None)
        if users is None:
            return
        if not users:
            latest_users = []
            return
        parsed: list[UserBox] = []
        for user in users:
            camera = user.get("camera") if isinstance(user, dict) else getattr(user, "camera", None)
            user_id = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
            if not camera or not user_id:
                continue
            try:
                location = user.get("location") if isinstance(user, dict) else getattr(user, "location", None)
                location_x = None
                location_z = None
                if isinstance(location, dict):
                    if location.get("x") is not None:
                        location_x = float(location.get("x"))
                    if location.get("z") is not None:
                        location_z = float(location.get("z"))
                parsed.append(
                    UserBox(
                        user_id=str(user_id),
                        x=int(camera.get("x")),
                        y=int(camera.get("y")),
                        w=int(camera.get("w")),
                        h=int(camera.get("h")),
                        location_x=location_x,
                        location_z=location_z,
                    )
                )
            except Exception:
                continue
        latest_users = parsed

    client = AsyncFurhatClient(args.furhat_ip, args.furhat_auth)
    await client.connect()
    client.add_handler(Events.response_camera_data, on_camera)
    client.add_handler(Events.response_users_data, on_users)
    await client.request_camera_start()
    await client.request_users_start()

    states: dict[str, TrackState] = {}
    print_state = {"last_print": 0}
    start_ts = time.time()
    try:
        while True:
            if args.max_runtime_sec is not None and (time.time() - start_ts) >= float(
                args.max_runtime_sec
            ):
                break
            if not frame_queue:
                await asyncio.sleep(0.01)
                continue
            frame = frame_queue.pop()
            keep_running = process_frame(
                frame,
                landmarker,
                image_module,
                oval_indices,
                states,
                spec,
                model,
                device,
                args,
                limiter,
                print_state,
                latest_users,
                cnn_jsonl_handle,
                fusion_runtime,
            )
            if not keep_running:
                break
            await asyncio.sleep(0)
    finally:
        await client.request_camera_stop()
        await client.request_users_stop()
        await client.disconnect()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / ".mplconfig"))
    if args.furhat_url and not args.stream_url:
        args.stream_url = args.furhat_url

    spec = load_model_spec(args)
    device = torch.device(args.device)
    model = build_model(len(spec.indices), spec.use_delta, device, spec.model_path)

    from mediapipe.tasks.python.core import base_options as base_options_module
    from mediapipe.tasks.python.vision import face_landmarker
    from mediapipe.tasks.python.vision.core import image as image_module
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

    oval = face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
    oval_indices = indices_from_connections(oval)

    BaseOptions = base_options_module.BaseOptions
    RunningMode = running_mode_module.VisionTaskRunningMode

    face_model_path = Path(args.face_model_path)
    if not face_model_path.exists():
        raise FileNotFoundError(f"FaceLandmarker model not found: {face_model_path}")

    delegate = (
        BaseOptions.Delegate.GPU if args.delegate == "gpu" else BaseOptions.Delegate.CPU
    )
    base_options = BaseOptions(
        model_asset_path=str(face_model_path),
        delegate=delegate,
    )
    options = face_landmarker.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_faces=args.max_faces,
        min_face_detection_confidence=args.min_det_conf,
        min_face_presence_confidence=args.min_presence_conf,
        min_tracking_confidence=args.min_track_conf,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    limiter = FrameRateLimiter(spec.target_fps)
    fusion_runtime = None
    if args.doa_jsonl_live:
        fusion_runtime = FusionOverlayRuntime(args)
    if args.show:
        cv2.namedWindow("vvad_realtime", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            "vvad_realtime",
            max(320, int(args.window_width)),
            max(240, int(args.window_height)),
        )
        if args.score_plot:
            cv2.namedWindow("score_plot_raw", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("score_plot_raw", 420, 220)
    cnn_jsonl_handle = None
    if args.emit_cnn_jsonl:
        cnn_jsonl_handle = open(args.emit_cnn_jsonl, "a", encoding="utf-8", buffering=1)
    try:
        with face_landmarker.FaceLandmarker.create_from_options(options) as landmarker:
            if args.source == "furhat":
                asyncio.run(
                    run_furhat_stream(
                        args,
                        spec,
                        model,
                        landmarker,
                        image_module,
                        oval_indices,
                        device,
                        limiter,
                        cnn_jsonl_handle,
                        fusion_runtime,
                    )
                )
            else:
                start_ts = time.time()
                if args.source == "file":
                    if not args.video_file:
                        raise ValueError("--video-file is required when --source=file")
                    source = args.video_file
                elif args.source == "stream":
                    if not args.stream_url:
                        raise ValueError("--stream-url is required when --source=stream")
                    source = args.stream_url
                else:
                    source = args.video_device

                cap = StreamSource(source)
                states: dict[str, TrackState] = {}
                print_state = {"last_print": 0}
                while True:
                    if args.max_runtime_sec is not None and (time.time() - start_ts) >= float(
                        args.max_runtime_sec
                    ):
                        break
                    frame = cap.read()
                    if frame is None:
                        break
                    if not process_frame(
                        frame,
                        landmarker,
                        image_module,
                        oval_indices,
                        states,
                        spec,
                        model,
                        device,
                        args,
                        limiter,
                        print_state,
                        None,
                        cnn_jsonl_handle,
                        fusion_runtime,
                    ):
                        break
                cap.release()
    finally:
        if cnn_jsonl_handle is not None:
            cnn_jsonl_handle.close()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
