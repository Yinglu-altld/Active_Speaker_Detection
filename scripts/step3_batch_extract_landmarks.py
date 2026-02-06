import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
LABEL_DIR = PROJECT_ROOT / "data" / "labels"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "landmarks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch Step 3: Extract MediaPipe face landmarks for all videos/entities "
            "and write per-entity landmarks.csv + meta.json under data/landmarks/."
        )
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "data" / "models" / "face_landmarker_v2.task"),
        help="Path to a MediaPipe FaceLandmarker .task file.",
    )
    parser.add_argument(
        "--delegate",
        choices=["cpu", "gpu"],
        default="cpu",
        help="MediaPipe delegate selection. Use cpu for headless environments.",
    )
    parser.add_argument(
        "--video-ids",
        default=None,
        help="Comma-separated list of video IDs to process. Default: all found in data/labels.",
    )
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-entities", type=int, default=None)
    parser.add_argument(
        "--min-rows",
        type=int,
        default=50,
        help="Skip entity tracks with fewer label rows than this.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--padding", type=float, default=0.2)
    parser.add_argument("--square", action="store_true")
    parser.add_argument("--min-det-conf", type=float, default=0.5)
    parser.add_argument("--min-presence-conf", type=float, default=0.5)
    parser.add_argument("--min-track-conf", type=float, default=0.5)
    parser.add_argument("--min-oval-size", type=float, default=0.2)
    parser.add_argument("--max-oval-size", type=float, default=0.98)
    parser.add_argument("--edge-margin", type=float, default=0.02)
    parser.add_argument(
        "--min-valid-rate",
        type=float,
        default=0.0,
        help="If > 0, only write outputs for entities whose valid_rate >= threshold.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--summary-name",
        default="batch_summary",
        help="Summary files: <output-dir>/<summary-name>.csv and .json",
    )
    return parser.parse_args()


def discover_video_ids_from_labels(label_dir: Path) -> list[str]:
    ids = []
    for p in sorted(label_dir.glob("*-activespeaker.csv")):
        ids.append(p.name.replace("-activespeaker.csv", ""))
    return ids


def list_entity_ids(df, min_rows: int) -> list[str]:
    counts = df["entity_id"].value_counts()
    entity_ids = [entity_id for entity_id, cnt in counts.items() if int(cnt) >= min_rows]
    return entity_ids


def main() -> None:
    args = parse_args()

    # Keep MediaPipe imports quieter/faster on macOS by using a writable cache directory.
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / ".mplconfig"))

    from mediapipe.tasks.python.core import base_options as base_options_module
    from mediapipe.tasks.python.vision import face_landmarker
    from mediapipe.tasks.python.vision.core import image as image_module
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

    from step2_crop_face import bbox_to_pixels, load_labels
    from step3_extract_landmarks import (
        compute_oval_stats,
        indices_from_connections,
        normalize_points,
        passes_quality_filter,
    )

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            "FaceLandmarker model not found.\n"
            f"Expected: {model_path}\n"
            "Download a FaceLandmarker .task model and place it at that path, "
            "or pass --model-path."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_ids = None
    if args.video_ids:
        video_ids = [v.strip() for v in args.video_ids.split(",") if v.strip()]
    else:
        video_ids = discover_video_ids_from_labels(LABEL_DIR)

    if args.max_videos is not None:
        video_ids = video_ids[: args.max_videos]

    BaseOptions = base_options_module.BaseOptions
    RunningMode = running_mode_module.VisionTaskRunningMode
    Image = image_module.Image
    ImageFormat = image_module.ImageFormat

    lips_connections = face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_LIPS
    oval_connections = face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
    lips_indices = indices_from_connections(lips_connections)
    oval_indices = indices_from_connections(oval_connections)
    selected_indices = sorted(set(lips_indices).union(oval_indices))

    delegate = (
        BaseOptions.Delegate.GPU if args.delegate == "gpu" else BaseOptions.Delegate.CPU
    )
    base_options = BaseOptions(model_asset_path=str(model_path), delegate=delegate)
    landmarker_options = face_landmarker.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=args.min_det_conf,
        min_face_presence_confidence=args.min_presence_conf,
        min_tracking_confidence=args.min_track_conf,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    summary_rows = []

    with face_landmarker.FaceLandmarker.create_from_options(landmarker_options) as landmarker:
        for video_id in video_ids:
            video_path = VIDEO_DIR / f"{video_id}.mp4"
            label_path = LABEL_DIR / f"{video_id}-activespeaker.csv"
            if not video_path.exists() or not label_path.exists():
                continue

            df = load_labels(label_path)
            entity_ids = list_entity_ids(df, args.min_rows)
            if args.max_entities is not None:
                entity_ids = entity_ids[: args.max_entities]

            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0:
                cap.release()
                continue

            for entity_id in entity_ids:
                entity_root = output_dir / video_id / entity_id
                landmarks_csv = entity_root / "landmarks.csv"
                meta_json = entity_root / "meta.json"
                if args.skip_existing and landmarks_csv.exists() and meta_json.exists():
                    continue

                df_entity = df[df["entity_id"] == entity_id].sort_values("frame_timestamp")
                if df_entity.empty:
                    continue

                rows = []
                attempted = 0
                detected = 0
                valid = 0

                vis_dir = entity_root / "vis"
                fail_dir = entity_root / "fail"

                sampled = df_entity.iloc[:: args.stride] if args.stride > 1 else df_entity
                if args.max_frames is not None:
                    sampled = sampled.iloc[: args.max_frames]

                for _, label_row in sampled.iterrows():
                    timestamp = float(label_row["frame_timestamp"])
                    frame_idx = int(round(timestamp * fps))
                    if frame_idx < 0 or frame_idx >= frame_count:
                        continue

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    bbox = bbox_to_pixels(
                        label_row, frame_width, frame_height, args.padding, args.square
                    )
                    if bbox is None:
                        continue

                    x1, y1, x2, y2 = bbox
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    mp_image = Image(ImageFormat.SRGB, np.ascontiguousarray(crop_rgb))
                    result = landmarker.detect(mp_image)

                    success = bool(result.face_landmarks)
                    attempted += 1

                    record = {
                        "frame_idx": int(frame_idx),
                        "timestamp": float(timestamp),
                        "success": bool(success),
                        "valid": False,
                    }

                    if success:
                        detected += 1
                        landmarks = result.face_landmarks[0]

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
                            vis_dir.mkdir(parents=True, exist_ok=True)
                            out_path = vis_dir / f"{frame_idx:06d}_{timestamp:.2f}.jpg"
                            cv2.imwrite(str(out_path), crop)
                    else:
                        if args.save_failures:
                            fail_dir.mkdir(parents=True, exist_ok=True)
                            out_path = fail_dir / f"{frame_idx:06d}_{timestamp:.2f}.jpg"
                            cv2.imwrite(str(out_path), crop)

                    rows.append(record)

                detected_rate = (detected / attempted) if attempted else 0.0
                valid_rate = (valid / attempted) if attempted else 0.0

                summary_rows.append(
                    {
                        "video_id": video_id,
                        "entity_id": entity_id,
                        "attempted": attempted,
                        "detected": detected,
                        "valid": valid,
                        "detection_rate": detected_rate,
                        "valid_rate": valid_rate,
                        "written": False,
                    }
                )

                if args.min_valid_rate and valid_rate < args.min_valid_rate:
                    continue

                entity_root.mkdir(parents=True, exist_ok=True)
                import pandas as pd

                pd.DataFrame(rows).to_csv(landmarks_csv, index=False)
                meta = {
                    "video_id": video_id,
                    "entity_id": entity_id,
                    "model_path": str(model_path),
                    "running_mode": "IMAGE",
                    "delegate": args.delegate,
                    "fps": float(fps),
                    "frame_size": {"width": frame_width, "height": frame_height},
                    "stride": int(args.stride),
                    "padding": float(args.padding),
                    "square": bool(args.square),
                    "min_detection_confidence": float(args.min_det_conf),
                    "min_presence_confidence": float(args.min_presence_conf),
                    "min_tracking_confidence": float(args.min_track_conf),
                    "quality_filter": {
                        "min_oval_size": float(args.min_oval_size),
                        "max_oval_size": float(args.max_oval_size),
                        "edge_margin": float(args.edge_margin),
                    },
                    "selected_indices": selected_indices,
                }
                meta_json.write_text(json.dumps(meta, indent=2))
                summary_rows[-1]["written"] = True

            cap.release()

    # Write a batch summary to help diagnose low-quality tracks before Step 4.
    summary_csv = output_dir / f"{args.summary_name}.csv"
    summary_json = output_dir / f"{args.summary_name}.json"
    import pandas as pd

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "num_videos": len(video_ids),
                "num_entities_seen": int(len(summary_rows)),
                "num_entities_written": int(summary_df["written"].sum()) if not summary_df.empty else 0,
                "args": vars(args),
            },
            indent=2,
        )
    )

    written = int(summary_df["written"].sum()) if not summary_df.empty else 0
    print(f"entities_seen={len(summary_rows)}, entities_written={written}")
    print(f"summary_csv={summary_csv}")
    print(f"summary_json={summary_json}")


if __name__ == "__main__":
    main()
