import cv2
import pandas as pd
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
LABEL_DIR = PROJECT_ROOT / "data" / "labels"

# Choose one video to test
video_id = "WwoTG3_OjUg"  # change if needed

video_path = VIDEO_DIR / f"{video_id}.mp4"
label_path = LABEL_DIR / f"{video_id}-activespeaker.csv"

# Load CSV labels (AVA Active Speaker files have no header row)
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
df = pd.read_csv(label_path, header=None, names=columns)

# Pick one face track (entity_id)
entity_id = df["entity_id"].iloc[0]
df_entity = df[df["entity_id"] == entity_id].sort_values("frame_timestamp")

print(f"Using video_id: {video_id}")
print(f"Using entity_id: {entity_id}")
print(df_entity.head())

# Open video
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video FPS: {fps}")

# Try reading a few frames based on timestamps
for i in range(min(5, len(df_entity))):
    row = df_entity.iloc[i]
    timestamp = row["frame_timestamp"]

    frame_idx = int(round(timestamp * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    print(
        f"timestamp={timestamp:.2f}s -> frame_idx={frame_idx}, read_success={ret}"
    )

cap.release()
