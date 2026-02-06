import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
LABEL_DIR = PROJECT_ROOT / "data" / "labels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3: Run MediaPipe FaceMesh on face crops and extract "
            "mouth + jaw (face oval) landmarks for a quick stability sanity check."
        )
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "data" / "models" / "face_landmarker_v2.task"),
        help="Path to a MediaPipe FaceLandmarker .task file.",
    )
    parser.add_argument("--video-id", default="WwoTG3_OjUg")
    parser.add_argument("--entity-id", default=None)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--padding", type=float, default=0.2)
    parser.add_argument("--square", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument(
        "--save-failures",
        action="store_true",
        help="Save crops for frames where face detection fails.",
    )
    parser.add_argument(
        "--min-oval-size",
        type=float,
        default=0.2,
        help="Minimum face-oval bbox size in crop-normalized coords.",
    )
    parser.add_argument(
        "--max-oval-size",
        type=float,
        default=0.98,
        help="Maximum face-oval bbox size in crop-normalized coords.",
    )
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=0.02,
        help="Reject if face-oval bbox touches crop edges (normalized).",
    )
    parser.add_argument("--min-det-conf", type=float, default=0.5)
    parser.add_argument("--min-presence-conf", type=float, default=0.5)
    parser.add_argument("--min-track-conf", type=float, default=0.5)
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "data" / "landmarks")
    )
    return parser.parse_args()


def indices_from_connections(connections) -> list[int]:
    indices: set[int] = set()
    for conn in connections:
        if hasattr(conn, "start") and hasattr(conn, "end"):
            a, b = conn.start, conn.end
        else:
            a, b = conn
        indices.add(int(a))
        indices.add(int(b))
    return sorted(indices)


def draw_points(image, points_px, color, radius: int = 1) -> None:
    for x, y in points_px:
        cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


def draw_connections(image, landmarks_px, connections, color, thickness: int = 1) -> None:
    for conn in connections:
        if hasattr(conn, "start") and hasattr(conn, "end"):
            a, b = conn.start, conn.end
        else:
            a, b = conn
        ax, ay = landmarks_px[int(a)]
        bx, by = landmarks_px[int(b)]
        cv2.line(
            image,
            (ax, ay),
            (bx, by),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def mean_motion(prev_xy: np.ndarray, curr_xy: np.ndarray) -> float:
    if prev_xy.size == 0 or curr_xy.size == 0:
        return float("nan")
    diff = curr_xy - prev_xy
    dist = np.sqrt((diff * diff).sum(axis=1))
    return float(dist.mean())


def compute_oval_stats(oval_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    min_xy = np.min(oval_xy, axis=0)
    max_xy = np.max(oval_xy, axis=0)
    width = float(max_xy[0] - min_xy[0])
    height = float(max_xy[1] - min_xy[1])
    cx = float((min_xy[0] + max_xy[0]) / 2.0)
    cy = float((min_xy[1] + max_xy[1]) / 2.0)
    scale = float(max(width, height))
    return min_xy, max_xy, width, height, cx, cy, scale


def passes_quality_filter(
    min_xy: np.ndarray,
    max_xy: np.ndarray,
    width: float,
    height: float,
    edge_margin: float,
    min_size: float,
    max_size: float,
) -> bool:
    if width <= 0 or height <= 0:
        return False
    if width < min_size or height < min_size:
        return False
    if width > max_size or height > max_size:
        return False
    if min_xy[0] < edge_margin or min_xy[1] < edge_margin:
        return False
    if max_xy[0] > 1.0 - edge_margin or max_xy[1] > 1.0 - edge_margin:
        return False
    return True


def normalize_points(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray | None:
    if scale <= 0:
        return None
    return (points - center) / scale


def main() -> None:
    args = parse_args()

    # MediaPipe imports can trigger matplotlib cache creation. Point that cache
    # at a project-writable directory to avoid noisy warnings and slow imports.
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / ".mplconfig"))

    # Reuse the exact same label parsing + bbox-to-crop logic as Step 2,
    # so Step 3 stays consistent with the verified crop behavior.
    from step2_crop_face import bbox_to_pixels, load_labels, pick_entity

    from mediapipe.tasks.python.core import base_options as base_options_module
    from mediapipe.tasks.python.vision import face_landmarker
    from mediapipe.tasks.python.vision.core import image as image_module
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

    video_path = VIDEO_DIR / f"{args.video_id}.mp4"
    label_path = LABEL_DIR / f"{args.video_id}-activespeaker.csv"
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            "FaceLandmarker model not found.\n"
            f"Expected: {model_path}\n"
            "Download a FaceLandmarker .task model (e.g. face_landmarker_v2.task) "
            "and place it at that path, or pass --model-path."
        )

    df = load_labels(label_path)
    entity_id, df_entity = pick_entity(df, args.entity_id)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        raise RuntimeError(f"Could not read FPS from video: {video_path}")

    output_root = Path(args.output_dir) / args.video_id / entity_id
    output_root.mkdir(parents=True, exist_ok=True)
    vis_dir = output_root / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    fail_dir = output_root / "fail"
    if args.save_failures:
        fail_dir.mkdir(parents=True, exist_ok=True)

    lips_connections = face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_LIPS
    oval_connections = (
        face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
    )
    lips_indices = indices_from_connections(lips_connections)
    oval_indices = indices_from_connections(oval_connections)
    selected_indices = sorted(set(lips_indices).union(oval_indices))

    BaseOptions = base_options_module.BaseOptions
    RunningMode = running_mode_module.VisionTaskRunningMode
    Image = image_module.Image
    ImageFormat = image_module.ImageFormat

    base_options = BaseOptions(model_asset_path=str(model_path))
    options = face_landmarker.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=args.min_det_conf,
        min_face_presence_confidence=args.min_presence_conf,
        min_tracking_confidence=args.min_track_conf,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    meta = {
        "video_id": args.video_id,
        "entity_id": entity_id,
        "model_path": str(model_path),
        "fps": float(fps),
        "frame_size": {"width": frame_width, "height": frame_height},
        "stride": args.stride,
        "padding": args.padding,
        "square": bool(args.square),
        "quality_filter": {
            "min_oval_size": args.min_oval_size,
            "max_oval_size": args.max_oval_size,
            "edge_margin": args.edge_margin,
        },
        "normalization": {
            "center": "oval_bbox_center",
            "scale": "max(oval_w, oval_h)",
        },
        "min_detection_confidence": args.min_det_conf,
        "min_presence_confidence": args.min_presence_conf,
        "min_tracking_confidence": args.min_track_conf,
        "lips_indices": lips_indices,
        "oval_indices": oval_indices,
        "selected_indices": selected_indices,
    }
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2))

    # We store per-frame landmarks in crop-normalized coordinates (x/y in [0,1]),
    # plus a centered+scaled variant (nx/ny) based on the face-oval bounding box.
    rows = []
    saved = 0
    attempted = 0
    detected = 0
    valid = 0

    prev_lips_xy = None
    prev_oval_xy = None
    lips_motion = []
    oval_motion = []

    with face_landmarker.FaceLandmarker.create_from_options(options) as landmarker:
        for idx in range(0, len(df_entity), args.stride):
            if saved >= args.max_frames:
                break
            row = df_entity.iloc[idx]
            frame_idx = int(round(row["frame_timestamp"] * fps))
            if frame_idx < 0 or frame_idx >= frame_count:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            bbox = bbox_to_pixels(
                row, frame_width, frame_height, args.padding, args.square
            )
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = Image(ImageFormat.SRGB, np.ascontiguousarray(crop_rgb))
            timestamp_ms = int(round(row["frame_timestamp"] * 1000.0))
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            success = bool(results.face_landmarks)
            attempted += 1

            record = {
                "frame_idx": int(frame_idx),
                "timestamp": float(row["frame_timestamp"]),
                "success": bool(success),
                "valid": False,
            }

            if success:
                detected += 1
                landmarks = results.face_landmarks[0]

                # Build full 468 landmark pixel coords for drawing connections.
                crop_h, crop_w = crop.shape[:2]
                landmarks_px = [
                    (int(lm.x * crop_w), int(lm.y * crop_h)) for lm in landmarks
                ]

                lips_xy = np.array(
                    [(landmarks[i].x, landmarks[i].y) for i in lips_indices],
                    dtype=np.float32,
                )
                oval_xy = np.array(
                    [(landmarks[i].x, landmarks[i].y) for i in oval_indices],
                    dtype=np.float32,
                )
                min_xy, max_xy, oval_w, oval_h, oval_cx, oval_cy, oval_scale = (
                    compute_oval_stats(oval_xy)
                )
                quality_ok = passes_quality_filter(
                    min_xy,
                    max_xy,
                    oval_w,
                    oval_h,
                    args.edge_margin,
                    args.min_oval_size,
                    args.max_oval_size,
                )

                if prev_lips_xy is not None and prev_oval_xy is not None:
                    lips_motion.append(mean_motion(prev_lips_xy, lips_xy))
                    oval_motion.append(mean_motion(prev_oval_xy, oval_xy))
                prev_lips_xy = lips_xy
                prev_oval_xy = oval_xy

                record["valid"] = bool(quality_ok)
                record["oval_w"] = oval_w
                record["oval_h"] = oval_h
                record["oval_cx"] = oval_cx
                record["oval_cy"] = oval_cy
                record["oval_scale"] = oval_scale
                if quality_ok:
                    valid += 1

                selected_xy = []
                for i in selected_indices:
                    lm = landmarks[i]
                    record[f"x_{i}"] = float(lm.x)
                    record[f"y_{i}"] = float(lm.y)
                    record[f"z_{i}"] = float(lm.z)
                    selected_xy.append((lm.x, lm.y))

                selected_xy = np.array(selected_xy, dtype=np.float32)
                center = np.array([oval_cx, oval_cy], dtype=np.float32)
                norm_xy = normalize_points(selected_xy, center, oval_scale)
                if norm_xy is not None:
                    for i, (nx, ny) in zip(selected_indices, norm_xy):
                        record[f"nx_{i}"] = float(nx)
                        record[f"ny_{i}"] = float(ny)

                if args.save_vis:
                    vis = crop.copy()
                    draw_connections(vis, landmarks_px, oval_connections, (255, 0, 0), 1)
                    draw_connections(vis, landmarks_px, lips_connections, (0, 255, 0), 1)
                    draw_points(
                        vis,
                        [landmarks_px[i] for i in oval_indices],
                        (255, 0, 0),
                        1,
                    )
                    draw_points(
                        vis,
                        [landmarks_px[i] for i in lips_indices],
                        (0, 255, 0),
                        1,
                    )
                    cv2.putText(
                        vis,
                        f"frame={frame_idx} t={row['frame_timestamp']:.2f}s",
                        (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )
                    out_path = vis_dir / f"{frame_idx:06d}_{row['frame_timestamp']:.2f}.jpg"
                    cv2.imwrite(str(out_path), vis)
            elif args.save_failures:
                fail_path = (
                    fail_dir / f"{frame_idx:06d}_{row['frame_timestamp']:.2f}.jpg"
                )
                cv2.imwrite(str(fail_path), crop)

            rows.append(record)
            saved += 1

    cap.release()

    # Write a compact CSV for inspection / plotting.
    # Frames where detection fails will have missing landmark columns.
    import pandas as pd

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_root / "landmarks.csv", index=False)

    def safe_mean(values) -> float:
        if not values:
            return float("nan")
        return float(np.nanmean(np.array(values, dtype=np.float32)))

    print(f"video_id={args.video_id}")
    print(f"entity_id={entity_id}")
    print(f"fps={fps}, frames={frame_count}, size={frame_width}x{frame_height}")
    print(f"attempted={attempted}, detected={detected}, valid={valid}, saved={saved}")
    print(f"detection_rate={detected / attempted if attempted else 0.0:.3f}")
    print(f"valid_rate={valid / attempted if attempted else 0.0:.3f}")
    print(f"mean_motion_lips={safe_mean(lips_motion):.6f} (normalized crop coords)")
    print(f"mean_motion_oval={safe_mean(oval_motion):.6f} (normalized crop coords)")
    print(f"output_dir={output_root}")
    if args.save_vis:
        print(f"vis_dir={vis_dir}")


if __name__ == "__main__":
    main()
