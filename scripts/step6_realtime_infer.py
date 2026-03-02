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


class ScoreHistory:
    def __init__(self, max_points: int = 140):
        self.max_points = max(20, int(max_points))
        self._series: dict[str, dict[str, deque]] = {}
        self._reality_enabled = False
        self._palette: list[tuple[int, int, int]] = [
            (59, 85, 255),
            (70, 195, 120),
            (255, 150, 60),
            (205, 80, 235),
            (90, 210, 220),
            (230, 220, 70),
            (170, 110, 255),
            (255, 180, 170),
            (110, 210, 120),
            (245, 130, 210),
            (120, 230, 230),
            (170, 170, 170),
        ]
        self._colors: dict[str, tuple[int, int, int]] = {}

    def _ensure(self, track_id: str) -> dict[str, deque]:
        series = self._series.get(track_id)
        if series is None:
            series = {
                "cnn": deque(maxlen=self.max_points),
                "doa": deque(maxlen=self.max_points),
                "overall": deque(maxlen=self.max_points),
                "reality": deque(maxlen=self.max_points),
            }
            self._series[track_id] = series
            if track_id not in self._colors:
                color_idx = len(self._colors) % len(self._palette)
                self._colors[track_id] = self._palette[color_idx]
        return series

    @staticmethod
    def _to_score(value) -> float:
        if value is None:
            return float("nan")
        try:
            score = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return float(np.clip(score, 0.0, 1.0))

    @classmethod
    def _to_binary(cls, value) -> float:
        score = cls._to_score(value)
        if math.isnan(score):
            return 0.0
        return 1.0 if score >= 0.5 else 0.0

    def update(
        self,
        outputs: list[dict],
        hide_values: bool = False,
        gt_values: dict[str, float] | None = None,
    ) -> None:
        frame_gt: dict[str, float] = {}
        if gt_values is not None:
            self._reality_enabled = True
            for track_id, value in gt_values.items():
                tid = str(track_id)
                if not tid:
                    continue
                self._ensure(tid)
                frame_gt[tid] = self._to_binary(value)

        frame_values: dict[str, tuple[float, float, float]] = {}
        for out in outputs:
            track_id = str(out.get("track_id"))
            if not track_id:
                continue
            self._ensure(track_id)
            cnn_score = out.get("plot_cnn")
            if cnn_score is None:
                cnn_score = out.get("fusion_cnn")
            if cnn_score is None:
                cnn_score = out.get("prob")
            doa_score = out.get("plot_doa")
            if doa_score is None:
                doa_score = out.get("fusion_doa")
            overall_score = out.get("fusion_overall")
            frame_values[track_id] = (
                self._to_score(cnn_score),
                self._to_score(doa_score),
                self._to_score(overall_score),
            )

        if hide_values:
            for out in outputs:
                track_id = str(out.get("track_id"))
                if not track_id:
                    continue
                self._ensure(track_id)
            for track_id, series in self._series.items():
                cnn_score, doa_score, _overall_score = frame_values.get(
                    track_id,
                    (float("nan"), float("nan"), float("nan")),
                )
                series["cnn"].append(cnn_score)
                series["doa"].append(doa_score)
                series["overall"].append(0.0)
                if self._reality_enabled:
                    series["reality"].append(frame_gt.get(track_id, 0.0))
            return

        for track_id, series in self._series.items():
            cnn_score, doa_score, overall_score = frame_values.get(
                track_id,
                (float("nan"), float("nan"), float("nan")),
            )
            series["cnn"].append(cnn_score)
            series["doa"].append(doa_score)
            series["overall"].append(overall_score)
            if self._reality_enabled:
                series["reality"].append(frame_gt.get(track_id, 0.0))

    def get(self, track_id: str) -> dict[str, deque] | None:
        return self._series.get(str(track_id))

    def get_color(self, track_id: str) -> tuple[int, int, int]:
        if track_id not in self._colors:
            color_idx = len(self._colors) % len(self._palette)
            self._colors[track_id] = self._palette[color_idx]
        return self._colors[track_id]

    def items(self):
        return self._series.items()

    def reality_enabled(self) -> bool:
        return bool(self._reality_enabled)

    def enable_reality(self) -> None:
        self._reality_enabled = True


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
        "--flip-cnn-bearing",
        action="store_true",
        help="Flip sign of estimated bearing before writing fusion JSONL.",
    )
    parser.add_argument(
        "--doa-jsonl-live",
        default=None,
        help="Optional live DOA JSONL path for realtime fusion overlay in --show mode.",
    )
    parser.add_argument(
        "--gt-jsonl-live",
        default=None,
        help="Optional live button-ground-truth JSONL path for score plots.",
    )
    parser.add_argument(
        "--max-doa-staleness-sec",
        type=float,
        default=0.8,
        help="Max |t_frame - t_doa| to accept DOA for fusion overlay.",
    )
    parser.add_argument(
        "--max-gt-staleness-sec",
        type=float,
        default=0.8,
        help="Max |t_frame - t_gt| to accept button ground-truth values for plots.",
    )
    parser.add_argument("--fusion-default-sigma-deg", type=float, default=25.0)
    parser.add_argument("--fusion-min-sigma-deg", type=float, default=8.0)
    parser.add_argument(
        "--min-speaker-score",
        type=float,
        default=0.30,
        help="Minimum fusion score required to mark a speaker as active in UI overlay.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
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


def _draw_status_hud(frame, hud: dict | None) -> None:
    if not hud:
        return
    mode = str(hud.get("mode", "UNKNOWN"))
    if mode == "NO_SPEECH":
        state_color = (0, 0, 255)
    elif mode in ("SPEECH_AV", "SPEECH_CNN_ONLY", "SPEECH_DOA_ONLY"):
        state_color = (0, 255, 0)
    else:
        state_color = (0, 165, 255)

    state_text = f"STATE: {mode}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 3
    (text_w, text_h), baseline = cv2.getTextSize(state_text, font, scale, thickness)
    pad_x = 18
    pad_y = 12
    x1 = max(8, int((frame.shape[1] - (text_w + 2 * pad_x)) / 2))
    y1 = 8
    x2 = min(frame.shape[1] - 8, x1 + text_w + 2 * pad_x)
    y2 = y1 + text_h + 2 * pad_y + baseline
    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), -1)
    text_x = x1 + pad_x
    text_y = y2 - pad_y - baseline
    cv2.putText(
        frame,
        state_text,
        (text_x, text_y),
        font,
        scale,
        state_color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_score_line(
    frame: np.ndarray,
    values: list[float],
    x0: int,
    y0: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    if len(values) < 2 or width <= 1 or height <= 1:
        return
    segments: list[list[list[int]]] = []
    current: list[list[int]] = []
    denom = max(1, len(values) - 1)
    for i, value in enumerate(values):
        if math.isnan(value):
            if len(current) >= 2:
                segments.append(current)
            current = []
            continue
        px = x0 + int(round((i / denom) * (width - 1)))
        py = y0 + int(round((1.0 - value) * (height - 1)))
        current.append([px, py])
    if len(current) >= 2:
        segments.append(current)
    for segment in segments:
        pts = np.array(segment, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, color, 2, lineType=cv2.LINE_AA)


def _draw_doa_compass(
    board: np.ndarray,
    doa: dict | None,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> None:
    cv2.rectangle(board, (x1, y1), (x2, y2), (38, 38, 38), 1)
    title_h = 24
    footer_h = 24
    cv2.putText(
        board,
        "DOA Azimuth (Ungated+Offset)",
        (x1 + 8, y1 + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        lineType=cv2.LINE_AA,
    )

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    cx = int((x1 + x2) / 2)
    free_h = max(30, box_h - title_h - footer_h - 10)
    radius = max(20, min((box_w - 36) // 2, free_h // 2))
    cy = y1 + title_h + 6 + radius
    cv2.circle(board, (cx, cy), radius, (120, 120, 120), 1, lineType=cv2.LINE_AA)
    cv2.circle(board, (cx, cy), 2, (170, 170, 170), -1, lineType=cv2.LINE_AA)

    # Rotate display coordinates by 90 deg so 0° is at the bottom.
    # Label placement is kept inside the circle to avoid title/value overlap.
    cv2.putText(board, "180", (cx - 16, cy - radius + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (165, 165, 165), 1, lineType=cv2.LINE_AA)
    cv2.putText(board, "0", (cx - 5, cy + radius - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (165, 165, 165), 1, lineType=cv2.LINE_AA)
    cv2.putText(board, "90", (cx + radius - 22, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (165, 165, 165), 1, lineType=cv2.LINE_AA)
    cv2.putText(board, "270", (cx - radius + 4, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (165, 165, 165), 1, lineType=cv2.LINE_AA)

    raw_deg = None
    if isinstance(doa, dict):
        raw_deg = doa.get("azimuth_plot_deg")
        if raw_deg is None:
            raw_deg = doa.get("azimuth_deg")

    try:
        raw_value = None if raw_deg is None else float(raw_deg)
    except (TypeError, ValueError):
        raw_value = None

    if raw_value is None:
        cv2.putText(
            board,
            "az(off): -",
            (x1 + 8, y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (180, 180, 180),
            1,
            lineType=cv2.LINE_AA,
        )
        return

    theta = math.radians(raw_value - 90.0)
    px = int(round(cx + radius * math.cos(theta)))
    py = int(round(cy - radius * math.sin(theta)))
    cv2.line(board, (cx, cy), (px, py), (80, 220, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(board, (px, py), 4, (80, 220, 255), -1, lineType=cv2.LINE_AA)
    cv2.putText(
        board,
        f"az(off): {raw_value:.1f} deg",
        (x1 + 8, y2 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (200, 200, 200),
        1,
        lineType=cv2.LINE_AA,
    )


def render_score_board(
    outputs: list[dict],
    hud: dict | None,
    score_history: ScoreHistory,
    width: int = 980,
    height: int = 700,
) -> np.ndarray:
    board = np.full((height, width, 3), 16, dtype=np.uint8)
    mode = "UNKNOWN" if hud is None else str(hud.get("mode", "UNKNOWN"))
    if mode == "NO_SPEECH":
        mode_color = (0, 0, 255)
    elif mode in ("SPEECH_AV", "SPEECH_CNN_ONLY", "SPEECH_DOA_ONLY"):
        mode_color = (0, 255, 0)
    else:
        mode_color = (0, 165, 255)
    cv2.putText(
        board,
        f"STATE: {mode}",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        mode_color,
        2,
        lineType=cv2.LINE_AA,
    )
    doa_hud = {} if hud is None else dict(hud.get("doa") or {})
    compass_w = 230
    compass_h = 168
    compass_x2 = width - 16
    compass_x1 = max(16, compass_x2 - compass_w)
    compass_y1 = 10
    compass_y2 = compass_y1 + compass_h
    _draw_doa_compass(
        board=board,
        doa=doa_hud,
        x1=compass_x1,
        y1=compass_y1,
        x2=compass_x2,
        y2=compass_y2,
    )

    user_ids: list[str] = []
    for out in outputs:
        user_id = str(out.get("track_id"))
        if user_id not in user_ids:
            user_ids.append(user_id)
    for user_id, _series in score_history.items():
        if user_id not in user_ids:
            user_ids.append(user_id)
    user_ids = user_ids[:8]

    legend_y = 60
    legend_x = 16
    for user_id in user_ids:
        color = score_history.get_color(user_id)
        cv2.rectangle(board, (legend_x, legend_y), (legend_x + 14, legend_y + 14), color, -1)
        cv2.putText(
            board,
            user_id,
            (legend_x + 20, legend_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
            lineType=cv2.LINE_AA,
        )
        legend_x += 125
        if legend_x > (width - 140):
            legend_x = 16
            legend_y += 20

    header_h = max(legend_y + 24, compass_y2 + 10)
    metrics = [("CNN Score", "cnn"), ("DOA Score", "doa"), ("Overall Score", "overall")]
    if score_history.reality_enabled():
        metrics.append(("Reality (Buttons)", "reality"))
    pad = 16
    section_gap = 12
    usable_h = height - header_h - pad
    section_h = max(110, (usable_h - section_gap * (len(metrics) - 1)) // len(metrics))
    section_w = width - 2 * pad

    for idx, (title, key) in enumerate(metrics):
        y1 = header_h + idx * (section_h + section_gap)
        y2 = y1 + section_h
        x1 = pad
        x2 = x1 + section_w
        cv2.rectangle(board, (x1, y1), (x2, y2), (38, 38, 38), 1)
        cv2.putText(
            board,
            title,
            (x1 + 8, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            lineType=cv2.LINE_AA,
        )

        plot_x = x1 + 52
        plot_y = y1 + 26
        plot_w = section_w - 64
        plot_h = section_h - 36
        cv2.rectangle(board, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (70, 70, 70), 1)
        mid_y = plot_y + plot_h // 2
        cv2.line(board, (plot_x, mid_y), (plot_x + plot_w, mid_y), (55, 55, 55), 1)
        cv2.putText(board, "1.0", (x1 + 8, plot_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, lineType=cv2.LINE_AA)
        cv2.putText(board, "0.5", (x1 + 8, mid_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, lineType=cv2.LINE_AA)
        cv2.putText(board, "0.0", (x1 + 8, plot_y + plot_h), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, lineType=cv2.LINE_AA)

        for user_id in user_ids:
            series = score_history.get(user_id)
            if not series:
                continue
            _draw_score_line(
                board,
                list(series[key]),
                plot_x,
                plot_y,
                plot_w,
                plot_h,
                score_history.get_color(user_id),
            )

    return board


def draw_overlays(
    frame,
    outputs,
    hud: dict | None = None,
) -> None:
    _draw_status_hud(frame, hud)
    no_speech_mode = bool(hud and str(hud.get("mode", "")) == "NO_SPEECH")
    for out in outputs:
        bbox = out.get("bbox")
        track_id = str(out.get("track_id") or "?")
        score_cnn = out.get("fusion_cnn")
        score_doa = out.get("fusion_doa")
        overall = out.get("fusion_overall")
        is_active = bool(out.get("active_speaker"))
        no_speech = bool(out.get("no_speech"))
        if no_speech_mode or no_speech:
            color = (0, 0, 255)
        elif is_active:
            color = (0, 255, 0)
        elif out.get("fusion_overall") is not None:
            color = (0, 165, 255)
        else:
            color = prob_color(float(out.get("prob", 0.0)))
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if no_speech_mode:
                label = f"{track_id} cnn:- doa:- o:-"
            else:
                cnn_txt = "-" if score_cnn is None else f"{float(score_cnn):.2f}"
                doa_txt = "-" if score_doa is None else f"{float(score_doa):.2f}"
                over_txt = "-" if overall is None else f"{float(overall):.2f}"
                label = f"{track_id} cnn:{cnn_txt} doa:{doa_txt} o:{over_txt}"
            text_x = x1
            text_y = max(18, y1 - 6)
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                lineType=cv2.LINE_AA,
            )


def _wrap_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _compute_plot_doa_score(
    azimuth_deg: float | None,
    sigma_deg: float | None,
    bearing_deg: float | None,
    min_sigma_deg: float,
    default_sigma_deg: float,
) -> float:
    if azimuth_deg is None or bearing_deg is None:
        return float("nan")
    sigma = float(default_sigma_deg if sigma_deg is None else sigma_deg)
    sigma = max(float(min_sigma_deg), sigma)
    delta = abs(_wrap_deg(float(azimuth_deg) - float(bearing_deg)))
    return float(math.exp(-0.5 * (delta / sigma) ** 2))


def _prepare_plot_values(outputs: list[dict], hud: dict | None, args: argparse.Namespace) -> None:
    doa_payload = {} if hud is None else dict(hud.get("doa") or {})
    azimuth_value = doa_payload.get("azimuth_plot_deg")
    if azimuth_value is None:
        azimuth_value = doa_payload.get("azimuth_deg")
    sigma_value = doa_payload.get("sigma_deg")
    try:
        plot_azimuth = None if azimuth_value is None else float(azimuth_value)
    except (TypeError, ValueError):
        plot_azimuth = None
    try:
        plot_sigma = None if sigma_value is None else float(sigma_value)
    except (TypeError, ValueError):
        plot_sigma = None
    for out in outputs:
        try:
            out["plot_cnn"] = float(np.clip(float(out.get("prob")), 0.0, 1.0))
        except (TypeError, ValueError):
            out["plot_cnn"] = float("nan")
        bearing = out.get("bearing_deg")
        try:
            bearing_value = None if bearing is None else float(bearing)
        except (TypeError, ValueError):
            bearing_value = None
        out["plot_doa"] = _compute_plot_doa_score(
            azimuth_deg=plot_azimuth,
            sigma_deg=plot_sigma,
            bearing_deg=bearing_value,
            min_sigma_deg=float(args.fusion_min_sigma_deg),
            default_sigma_deg=float(args.fusion_default_sigma_deg),
        )


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
    flip: bool,
) -> float | None:
    bearing = location_to_bearing_deg(user.location_x, user.location_z)
    if bearing is None:
        bearing = bbox_to_bearing_deg(bbox, frame_width, hfov_deg)
    if bearing is None:
        return None
    return float(-bearing if flip else bearing)


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
        return self.latest


class LiveGroundTruthSnapshots:
    def __init__(self, path: str):
        self.path = path
        self.handle = None
        self.offset = 0
        self.latest = None
        self._known_user_ids: set[str] = set()

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
            self._known_user_ids.clear()

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
            gt = item.get("gt")
            if not isinstance(gt, dict):
                continue
            if item.get("t") is None:
                continue
            self.latest = item
            self._known_user_ids = {str(k) for k in gt.keys() if str(k)}

    @staticmethod
    def _normalize_gt_map(value: dict) -> dict[str, float]:
        result: dict[str, float] = {}
        for key, raw in value.items():
            user_id = str(key)
            if not user_id:
                continue
            if isinstance(raw, bool):
                result[user_id] = 1.0 if raw else 0.0
                continue
            try:
                result[user_id] = 1.0 if float(raw) >= 0.5 else 0.0
            except (TypeError, ValueError):
                result[user_id] = 0.0
        return result

    def get_for_time(self, t_value: float, max_staleness_sec: float) -> dict[str, float] | None:
        self.poll()
        if self.latest is None:
            return None
        t_obs = self.latest.get("t")
        gt_map = self.latest.get("gt")
        if t_obs is None or not isinstance(gt_map, dict):
            return None
        if abs(float(t_value) - float(t_obs)) > float(max_staleness_sec):
            if not self._known_user_ids:
                return None
            return {user_id: 0.0 for user_id in sorted(self._known_user_ids)}
        return self._normalize_gt_map(gt_map)


class FusionOverlayRuntime:
    def __init__(self, args: argparse.Namespace):
        if score_users_for_frame is None or FusionConfig is None or UserEvidence is None:
            raise ImportError(
                "audio_doa.fusion is required for DOA fusion overlay. Ensure repo root is importable."
            )
        self.live_doa = LiveDOASnapshots(args.doa_jsonl_live)
        self.max_staleness_sec = float(args.max_doa_staleness_sec)
        self.min_speaker_score = float(args.min_speaker_score)
        self.cfg = FusionConfig(
            min_sigma_deg=float(args.fusion_min_sigma_deg),
            default_sigma_deg=float(args.fusion_default_sigma_deg),
            fixed_doa_weight=0.35,
        )

    def annotate_outputs(self, outputs: list[dict], t_value: float) -> dict:
        doa_obs = self.live_doa.get_for_time(t_value=t_value, max_staleness_sec=self.max_staleness_sec)
        hud = {
            "mode": "NO_SPEECH" if doa_obs is None else "SPEECH_AV",
            "speech_active": False if doa_obs is None else bool(doa_obs.get("speech_active") or doa_obs.get("speech_detected")),
            "face_count": len(outputs),
            "speaker_id": None,
            "speaker_score": None,
            "weights": {"cnn": 1.0, "doa": 0.0},
            "doa": {
                "raw_azimuth_deg": None if doa_obs is None else doa_obs.get("raw_azimuth_deg"),
                "raw_azimuth_plot_deg": None if doa_obs is None else doa_obs.get("raw_azimuth_plot_deg"),
                "azimuth_deg": None if doa_obs is None else doa_obs.get("azimuth_deg"),
                "azimuth_plot_deg": None if doa_obs is None else doa_obs.get("azimuth_plot_deg"),
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
                out["no_speech"] = True
            return hud

        speech_active = bool(doa_obs.get("speech_active") or doa_obs.get("speech_detected"))
        if not speech_active:
            hud["mode"] = "NO_SPEECH"
            for out in outputs:
                out["fusion_overall"] = None
                out["fusion_cnn"] = float(out["prob"])
                out["fusion_doa"] = 0.0
                out["active_speaker"] = False
                out["no_speech"] = True
            return hud

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
            if doa_obs.get("azimuth_deg") is not None:
                hud["mode"] = "SPEECH_DOA_ONLY"
                hud["speaker_id"] = None
                hud["speaker_score"] = None
            else:
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
            return hud

        result = score_users_for_frame(doa_obs=doa_obs, users=users, cfg=self.cfg)
        hud["mode"] = "SPEECH_AV"
        raw_speaker_id = result.get("speaker_id")
        raw_speaker_score = result.get("speaker_score")
        active_id = None
        if raw_speaker_score is not None and float(raw_speaker_score) >= self.min_speaker_score:
            active_id = raw_speaker_id
        hud["speaker_id"] = active_id
        hud["speaker_score"] = raw_speaker_score
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
        return hud


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
    gt_runtime: LiveGroundTruthSnapshots | None = None,
    score_history: ScoreHistory | None = None,
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
        else:
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
                            args.flip_cnn_bearing,
                        ),
                    )
                )

    hud = None
    if fusion_runtime is not None:
        hud = fusion_runtime.annotate_outputs(outputs, now)
    _prepare_plot_values(outputs=outputs, hud=hud, args=args)
    if score_history is not None:
        hide_values = bool(hud and str(hud.get("mode", "")) == "NO_SPEECH")
        gt_values = None
        if gt_runtime is not None:
            gt_values = gt_runtime.get_for_time(
                t_value=now,
                max_staleness_sec=float(args.max_gt_staleness_sec),
            )
        score_history.update(outputs, hide_values=hide_values, gt_values=gt_values)

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
        cv2.imshow("vvad_realtime", frame)
        if score_history is not None:
            score_board = render_score_board(outputs, hud, score_history)
            cv2.imshow("vvad_scores", score_board)
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
    gt_runtime: LiveGroundTruthSnapshots | None = None,
    score_history: ScoreHistory | None = None,
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

    frame_queue: deque[np.ndarray] = deque(maxlen=1)
    latest_users: list[UserBox] = []
    furhat_id_to_alias: dict[str, str] = {}

    def _alloc_alias() -> str:
        used = set(furhat_id_to_alias.values())
        idx = 0
        while True:
            alias = f"user-{idx}"
            if alias not in used:
                return alias
            idx += 1

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
            furhat_id_to_alias.clear()
            return
        active_real_ids: set[str] = set()
        for user in users:
            camera = user.get("camera") if isinstance(user, dict) else getattr(user, "camera", None)
            user_id = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
            if not camera or user_id is None:
                continue
            active_real_ids.add(str(user_id))
        stale_real_ids = [real_id for real_id in furhat_id_to_alias.keys() if real_id not in active_real_ids]
        for real_id in stale_real_ids:
            furhat_id_to_alias.pop(real_id, None)

        parsed: list[UserBox] = []
        for user in users:
            camera = user.get("camera") if isinstance(user, dict) else getattr(user, "camera", None)
            user_id = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
            if not camera or user_id is None:
                continue
            real_id = str(user_id)
            alias_id = furhat_id_to_alias.get(real_id)
            if alias_id is None:
                alias_id = _alloc_alias()
                furhat_id_to_alias[real_id] = alias_id
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
                        user_id=alias_id,
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
    try:
        while True:
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
                gt_runtime,
                score_history,
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
    gt_runtime = None
    if args.gt_jsonl_live:
        gt_runtime = LiveGroundTruthSnapshots(args.gt_jsonl_live)
    score_history = ScoreHistory() if args.show else None
    if score_history is not None and gt_runtime is not None:
        score_history.enable_reality()
    if args.show:
        cv2.namedWindow("vvad_realtime", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            "vvad_realtime",
            max(320, int(args.window_width)),
            max(240, int(args.window_height)),
        )
        cv2.namedWindow("vvad_scores", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            "vvad_scores",
            max(700, int(args.window_width)),
            max(520, int(args.window_height)),
        )
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
                        gt_runtime,
                        score_history,
                    )
                )
            else:
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
                        gt_runtime,
                        score_history,
                    ):
                        break
                cap.release()
    finally:
        if cnn_jsonl_handle is not None:
            cnn_jsonl_handle.close()

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
