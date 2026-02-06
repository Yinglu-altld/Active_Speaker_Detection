import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
LABEL_DIR = PROJECT_ROOT / "data" / "labels"


def load_labels(label_path: Path) -> pd.DataFrame:
    columns = [
        "video_id",
        "frame_timestamp",
        "x1",
        "y1",
        "x2",
        "y2",
        "label",
        "entity_id",
    ]
    return pd.read_csv(label_path, header=None, names=columns)


def pick_entity(df: pd.DataFrame, entity_id: str | None) -> tuple[str, pd.DataFrame]:
    if entity_id is None:
        entity_id = df["entity_id"].iloc[0]
    df_entity = df[df["entity_id"] == entity_id].sort_values("frame_timestamp")
    return entity_id, df_entity


def is_normalized_bbox(row) -> bool:
    return max(row["x1"], row["y1"], row["x2"], row["y2"]) <= 1.5


def bbox_to_pixels(
    row,
    frame_width: int,
    frame_height: int,
    padding: float,
    square: bool,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
    if is_normalized_bbox(row):
        x1, x2 = x1 * frame_width, x2 * frame_width
        y1, y2 = y1 * frame_height, y2 * frame_height

    if x2 <= x1 or y2 <= y1:
        return None

    width = x2 - x1
    height = y2 - y1
    pad_x = width * padding
    pad_y = height * padding

    x1 -= pad_x
    x2 += pad_x
    y1 -= pad_y
    y2 += pad_y

    if square:
        width = x2 - x1
        height = y2 - y1
        if width > height:
            delta = (width - height) / 2
            y1 -= delta
            y2 += delta
        else:
            delta = (height - width) / 2
            x1 -= delta
            x2 += delta

    x1i = max(0, int(math.floor(x1)))
    y1i = max(0, int(math.floor(y1)))
    x2i = min(frame_width, int(math.ceil(x2)))
    y2i = min(frame_height, int(math.ceil(y2)))

    if x2i <= x1i or y2i <= y1i:
        return None

    return x1i, y1i, x2i, y2i


def save_crop(crop, out_path: Path) -> bool:
    if crop is None or crop.size == 0:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), crop))


def save_overlay(frame, bbox, out_path: Path) -> bool:
    x1, y1, x2, y2 = bbox
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), overlay))


def build_montage(images, cols: int, cell_size: int):
    if not images:
        return None
    cols = max(1, cols)
    rows = math.ceil(len(images) / cols)
    montage = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        y1, y2 = row * cell_size, (row + 1) * cell_size
        x1, x2 = col * cell_size, (col + 1) * cell_size
        montage[y1:y2, x1:x2] = img
    return montage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: Crop faces from AVA bounding boxes"
    )
    parser.add_argument("--video-id", default="WwoTG3_OjUg")
    parser.add_argument("--entity-id", default=None)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--padding", type=float, default=0.2)
    parser.add_argument("--square", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--montage", action="store_true")
    parser.add_argument("--montage-size", type=int, default=112)
    parser.add_argument("--montage-cols", type=int, default=6)
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "data" / "crops")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = VIDEO_DIR / f"{args.video_id}.mp4"
    label_path = LABEL_DIR / f"{args.video_id}-activespeaker.csv"

    df = load_labels(label_path)
    entity_id, df_entity = pick_entity(df, args.entity_id)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = Path(args.output_dir) / args.video_id / entity_id
    overlay_dir = Path(args.output_dir) / "overlays" / args.video_id / entity_id

    saved = 0
    attempted = 0
    montage_images = []

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
        crop_path = output_dir / f"{frame_idx:06d}_{row['frame_timestamp']:.2f}.jpg"
        if save_crop(crop, crop_path):
            saved += 1
            if args.montage:
                resized = cv2.resize(
                    crop, (args.montage_size, args.montage_size)
                )
                montage_images.append(resized)
            if args.save_overlays:
                overlay_path = (
                    overlay_dir / f"{frame_idx:06d}_{row['frame_timestamp']:.2f}.jpg"
                )
                save_overlay(frame, bbox, overlay_path)
        attempted += 1

    cap.release()

    print(f"video_id={args.video_id}")
    print(f"entity_id={entity_id}")
    print(f"fps={fps}, frames={frame_count}, size={frame_width}x{frame_height}")
    print(f"attempted={attempted}, saved={saved}")
    print(f"output_dir={output_dir}")
    if args.save_overlays:
        print(f"overlay_dir={overlay_dir}")
    if args.montage and montage_images:
        montage = build_montage(
            montage_images, args.montage_cols, args.montage_size
        )
        montage_path = output_dir / "montage.jpg"
        cv2.imwrite(str(montage_path), montage)
        print(f"montage={montage_path}")


if __name__ == "__main__":
    main()
