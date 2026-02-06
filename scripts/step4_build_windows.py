import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANDMARKS_DIR = PROJECT_ROOT / "data" / "landmarks"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 4: Build fixed-length time windows from face landmarks "
            "for CNN training. Supports subset or full processing."
        )
    )
    parser.add_argument("--landmarks-dir", default=str(LANDMARKS_DIR))
    parser.add_argument("--labels-dir", default=str(LABELS_DIR))
    parser.add_argument("--video-ids", default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-entities", type=int, default=None)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument("--window-frames", type=int, default=None)
    parser.add_argument("--hop-frames", type=int, default=None)
    parser.add_argument("--label-strategy", choices=["majority", "center"], default="majority")
    parser.add_argument("--max-gap-frames", type=int, default=None)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "windows"))
    parser.add_argument("--output-name", default="windows")
    return parser.parse_args()


def load_labels(labels_dir: Path, video_id: str, entity_id: str) -> pd.DataFrame:
    label_path = labels_dir / f"{video_id}-activespeaker.csv"
    cols = [
        "video_id",
        "frame_timestamp",
        "x1",
        "y1",
        "x2",
        "y2",
        "label",
        "entity_id",
    ]
    df = pd.read_csv(label_path, header=None, names=cols)
    df = df[df["entity_id"] == entity_id].copy()
    df["ts_key"] = df["frame_timestamp"].round(3)
    df = df.drop_duplicates(subset=["ts_key"])
    return df[["ts_key", "label"]]


def discover_videos(landmarks_dir: Path, video_ids: list[str] | None) -> list[Path]:
    if video_ids:
        return [landmarks_dir / vid for vid in video_ids]
    return sorted([p for p in landmarks_dir.iterdir() if p.is_dir()])


def parse_indices(prefix: str, columns: list[str]) -> list[int]:
    indices = []
    for col in columns:
        if not col.startswith(prefix):
            continue
        try:
            indices.append(int(col.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(indices))


def select_feature_columns(df: pd.DataFrame) -> tuple[list[int], str]:
    columns = list(df.columns)
    nx_indices = parse_indices("nx_", columns)
    ny_indices = parse_indices("ny_", columns)
    if nx_indices and ny_indices:
        indices = sorted(set(nx_indices).intersection(ny_indices))
        return indices, "nx"
    x_indices = parse_indices("x_", columns)
    y_indices = parse_indices("y_", columns)
    indices = sorted(set(x_indices).intersection(y_indices))
    return indices, "x"


def build_feature_tensor(df: pd.DataFrame, indices: list[int], prefix: str) -> np.ndarray:
    if prefix == "nx":
        x_cols = [f"nx_{i}" for i in indices]
        y_cols = [f"ny_{i}" for i in indices]
    else:
        x_cols = [f"x_{i}" for i in indices]
        y_cols = [f"y_{i}" for i in indices]
    x_vals = df[x_cols].to_numpy(dtype=np.float32)
    y_vals = df[y_cols].to_numpy(dtype=np.float32)
    return np.stack([x_vals, y_vals], axis=2)


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def resolve_window_params(args: argparse.Namespace, fps: float) -> tuple[int, int]:
    if args.window_frames is not None:
        window_frames = args.window_frames
    else:
        window_frames = max(1, int(round(args.window_sec * fps)))
    if args.hop_frames is not None:
        hop_frames = args.hop_frames
    else:
        hop_frames = max(1, int(round(args.hop_sec * fps)))
    return window_frames, hop_frames


def label_from_window(labels: np.ndarray, strategy: str) -> int:
    if strategy == "center":
        return int(labels[len(labels) // 2])
    return int(labels.mean() >= 0.5)


def main() -> None:
    args = parse_args()
    landmarks_dir = Path(args.landmarks_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_ids = None
    if args.video_ids:
        video_ids = [v.strip() for v in args.video_ids.split(",") if v.strip()]

    video_dirs = discover_videos(landmarks_dir, video_ids)
    if args.max_videos is not None:
        video_dirs = video_dirs[: args.max_videos]

    all_windows = []
    all_labels = []
    all_meta = []
    feature_indices = None
    feature_prefix = None

    for video_dir in video_dirs:
        video_id = video_dir.name
        entity_dirs = sorted([p for p in video_dir.iterdir() if p.is_dir()])
        if args.max_entities is not None:
            entity_dirs = entity_dirs[: args.max_entities]

        for entity_dir in entity_dirs:
            landmarks_path = entity_dir / "landmarks.csv"
            meta_path = entity_dir / "meta.json"
            if not landmarks_path.exists() or not meta_path.exists():
                continue

            meta = json.loads(meta_path.read_text())
            fps = float(meta.get("fps", 0))
            if fps <= 0:
                continue

            df = pd.read_csv(landmarks_path)
            if df.empty:
                continue

            label_df = load_labels(labels_dir, video_id, entity_dir.name)
            df["ts_key"] = df["timestamp"].round(3)
            df = df.merge(label_df, on="ts_key", how="left")
            df = df.dropna(subset=["label"])
            if df.empty:
                continue

            valid_col = "valid" if "valid" in df.columns else "success"
            df[valid_col] = to_bool(df[valid_col])
            df = df[df[valid_col]].copy()
            if df.empty:
                continue

            indices, prefix = select_feature_columns(df)
            if not indices:
                continue

            if feature_indices is None:
                feature_indices = indices
                feature_prefix = prefix
            elif indices != feature_indices or prefix != feature_prefix:
                continue

            feature_tensor = build_feature_tensor(df, indices, prefix)
            frame_idx = df["frame_idx"].to_numpy()
            timestamps = df["timestamp"].to_numpy()
            labels = (df["label"] == "SPEAKING").astype(np.int32).to_numpy()

            window_frames, hop_frames = resolve_window_params(args, fps)
            if window_frames <= 1 or hop_frames <= 0:
                continue

            for start in range(0, len(df) - window_frames + 1, hop_frames):
                end = start + window_frames
                if args.max_gap_frames is not None:
                    gaps = np.diff(frame_idx[start:end])
                    if gaps.size and gaps.max() > args.max_gap_frames:
                        continue

                window_feat = feature_tensor[start:end]
                window_labels = labels[start:end]
                window_label = label_from_window(window_labels, args.label_strategy)
                speech_ratio = float(window_labels.mean())

                all_windows.append(window_feat)
                all_labels.append(window_label)
                all_meta.append(
                    {
                        "video_id": video_id,
                        "entity_id": entity_dir.name,
                        "start_frame": int(frame_idx[start]),
                        "end_frame": int(frame_idx[end - 1]),
                        "start_ts": float(timestamps[start]),
                        "end_ts": float(timestamps[end - 1]),
                        "speech_ratio": speech_ratio,
                    }
                )

    if not all_windows:
        print("No windows created. Check inputs or loosen constraints.")
        return

    X = np.stack(all_windows, axis=0)
    y = np.array(all_labels, dtype=np.int64)
    meta_df = pd.DataFrame(all_meta)

    npz_path = output_dir / f"{args.output_name}.npz"
    meta_path = output_dir / f"{args.output_name}_meta.csv"
    info_path = output_dir / f"{args.output_name}_info.json"

    np.savez_compressed(npz_path, X=X, y=y)
    meta_df.to_csv(meta_path, index=False)
    info_path.write_text(
        json.dumps(
            {
                "num_windows": int(X.shape[0]),
                "window_frames": int(X.shape[1]),
                "num_points": int(X.shape[2]),
                "feature_type": feature_prefix,
                "indices": feature_indices,
                "label_strategy": args.label_strategy,
            },
            indent=2,
        )
    )

    print(f"windows={X.shape}, labels={y.shape}")
    print(f"npz={npz_path}")
    print(f"meta={meta_path}")
    print(f"info={info_path}")


if __name__ == "__main__":
    main()
