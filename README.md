# Active Speaker Detection (CNN + MediaPipe)

This repo builds a Visual Voice Activity Detection (VVAD) system using facial motion (mouth + jaw landmarks).

## Pipeline overview

1) Step 1: CSV ↔ video timestamp alignment (`scripts/step1_check_alignment.py`)
2) Step 2: AVA bbox → face crops (`scripts/step2_crop_face.py`)
3) Step 3: face crops → MediaPipe FaceLandmarker landmarks (`scripts/step3_extract_landmarks.py`)
4) Step 3 (batch): all videos/entities → landmarks (`scripts/step3_batch_extract_landmarks.py`)
5) Step 4: landmarks → fixed-length windows for CNN training (`scripts/step4_build_windows.py`)
6) Step 5: train a small CNN (`scripts/step5_train_cnn.py`)
7) Step 6: real-time inference (`scripts/step6_realtime_infer.py`)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## MediaPipe model (required for Step 3)

Step 3 uses MediaPipe Tasks `FaceLandmarker` and requires a `.task` model file.

- Put the model at `data/models/face_landmarker_v2.task`

## Batch processing (data prep)

Extract landmarks for all tracks (or a subset):

```bash
./venv/bin/python scripts/step3_batch_extract_landmarks.py --max-videos 1 --max-entities 5
```

Then build training windows:

```bash
./venv/bin/python scripts/step4_build_windows.py
```

Train a model (split by video_id):

```bash
./venv/bin/python scripts/step5_train_cnn.py --val-video WwoTG3_OjUg
```

Optionally evaluate on a held-out test video (excluded from training):

```bash
./venv/bin/python scripts/step5_train_cnn.py \
  --val-video WwoTG3_OjUg --test-video Ag-pXiLrd48
```

Evaluate a held-out test video without retraining (uses saved config + best.pt):

```bash
./venv/bin/python scripts/step5_train_cnn.py \
  --eval-only --test-video Ag-pXiLrd48
```

For clearer labels, keep only strong negatives/positives (optional):

```bash
./venv/bin/python scripts/step5_train_cnn.py \
  --val-video WwoTG3_OjUg --filter-extremes --neg-max 0.1 --pos-min 0.6
```

Disable delta (motion) features if needed:

```bash
./venv/bin/python scripts/step5_train_cnn.py --val-video WwoTG3_OjUg --no-delta
```

Run real-time inference (Furhat stream):

```bash
./venv/bin/python scripts/step6_realtime_infer.py --furhat-ip <FURHAT_IP> --show
```

For multi-user Furhat mode, the script uses `response.users.data` camera bboxes
to crop each user and runs inference per `user_id`.

Run real-time inference (local webcam):

```bash
./venv/bin/python scripts/step6_realtime_infer.py --source opencv --video-device 0 --show
```

Run real-time inference from a generic stream URL (MJPEG/RTSP):

```bash
./venv/bin/python scripts/step6_realtime_infer.py --source stream --stream-url <STREAM_URL> --show
```

## Final run used (current best)

These are the exact choices used for the current model:

- Step 3 batch: full data (no stride), default quality filters.
- Step 4: 1.5s window, 0.5s hop, 25 fps.
- Step 5: no speech_ratio filtering, delta features ON, val video `WwoTG3_OjUg`.

Commands:

```bash
./venv/bin/python scripts/step3_batch_extract_landmarks.py --min-rows 50
./venv/bin/python scripts/step4_build_windows.py --window-sec 1.5 --hop-sec 0.5 --target-fps 25
./venv/bin/python scripts/step5_train_cnn.py --val-video WwoTG3_OjUg --epochs 40 --batch-size 128
./venv/bin/python scripts/step5_train_cnn.py --eval-only --test-video Ag-pXiLrd48
```

Artifacts to use for inference:

- `data/models/cnn_vvad/best.pt`
- `data/models/cnn_vvad/threshold.json`
- `data/models/cnn_vvad/config.json`
- `data/windows/windows_info.json`

Final test metrics (held-out `Ag-pXiLrd48`):

- `test_f1=0.744`, `test_acc=0.843`, `test_pos_rate=0.251`
- `threshold=0.30`

## Notes

- `data/` contains large inputs (videos/labels) and generated artifacts (crops/landmarks/windows) and is ignored by git.
