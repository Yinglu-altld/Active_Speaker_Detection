import argparse
import asyncio
import base64
import json
import os
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--draw-landmarks", action="store_true")
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


def draw_overlays(frame, outputs) -> None:
    y = 24
    for out in outputs:
        track_id = out["track_id"]
        prob_out = out["prob"]
        speak = out["speak"]
        bbox = out.get("bbox")
        label = f"{track_id} {prob_out:.2f} {int(speak)}"
        color = prob_color(prob_out)
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text_x = x1
            text_y = max(18, y1 - 6)
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                label,
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
                )
            )
    else:
        if not user_boxes:
            if args.show and int(now) != print_state["last_print"]:
                print_state["last_print"] = int(now)
                print("no users")
            if args.show:
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
                )
            )

    if outputs:
        for out in outputs:
            print(f"{out['track_id']} {out['prob']:.3f} {int(out['speak'])}")
    elif args.show and int(now) != print_state["last_print"]:
        print_state["last_print"] = int(now)
        print("no face")

    if args.show:
        draw_overlays(frame, outputs)
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
                parsed.append(
                    UserBox(
                        user_id=str(user_id),
                        x=int(camera.get("x")),
                        y=int(camera.get("y")),
                        w=int(camera.get("w")),
                        h=int(camera.get("h")),
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
                ):
                    break
            cap.release()

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
