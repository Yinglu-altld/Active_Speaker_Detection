# Pipeline Walkthrough

This project is an **active speaker detection** system with three branches:

- Visual branch: per-user speaking probability from face landmark motion.
- Audio branch: ReSpeaker direction-of-arrival (DOA) + voice activity.
- Fusion branch: combines visual + audio evidence and drives Furhat attention.

## 1. Fast Orientation (What runs where)

- Offline data/model pipeline:
  - `scripts/step1_check_alignment.py`
  - `scripts/step2_crop_face.py`
  - `scripts/step3_batch_extract_landmarks.py`
  - `scripts/step4_build_windows.py`
  - `scripts/step5_train_cnn.py`
- Realtime visual inference:
  - `scripts/step6_realtime_infer.py`
- Realtime audio + fusion:
  - `audio_doa/doa_core.py`
  - `audio_doa/fusion.py`
  - `audio_doa/run_live_fusion.py`

## 2. Data Contract Between Stages

Each stage writes artifacts consumed by the next stage:

1. Step 3 writes per-entity landmark tracks:
   - `data/landmarks/<video_id>/<entity_id>/landmarks.csv`
   - `data/landmarks/<video_id>/<entity_id>/meta.json`
2. Step 4 converts tracks into training windows:
   - `data/windows/windows.npz` (`X`, `y`)
   - `data/windows/windows_meta.csv`
   - `data/windows/windows_info.json`
3. Step 5 trains CNN and writes model package:
   - `data/models/cnn_vvad/best.pt`
   - `data/models/cnn_vvad/config.json`
   - `data/models/cnn_vvad/threshold.json`
4. Step 6 loads model + windows metadata for realtime inference.
5. `run_live_fusion.py` starts Step 6 + `doa_core.py` and fuses both streams.

## 3. Offline Training Pipeline (Step-by-step)

1. **Alignment sanity check** (`step1`):
   - Verifies AVA timestamp -> frame indexing.
2. **Face crop verification** (`step2`):
   - Converts AVA bboxes to pixel crops with padding/square options.
3. **Landmark extraction** (`step3`):
   - Uses MediaPipe FaceLandmarker.
   - Keeps lips + face-oval points and quality-filters low-quality detections.
   - Produces raw (`x`,`y`) and normalized (`nx`,`ny`) features.
4. **Window building** (`step4`):
   - Merges landmarks with AVA labels.
   - Builds fixed temporal windows (default from sec -> frames via `target_fps`).
   - Creates binary speaking labels per window (`majority` or `center` strategy).
5. **CNN training** (`step5`):
   - Small temporal 1D CNN over flattened landmark sequences.
   - Splits by `video_id` (important leakage prevention).
   - Tunes decision threshold on validation set and saves it.

## 4. Realtime Pipeline (Runtime)

1. Step 6 receives frames (Furhat, webcam, file, or stream).
2. For each tracked user face:
   - Crop by user bbox (Furhat mode) or full-frame face detection.
   - Extract landmarks.
   - Build feature window buffer.
   - Run CNN -> speaking probability.
3. `doa_core.py` polls ReSpeaker over USB:
   - Emits speech flag + azimuth (`azimuth_deg`) JSON events.
4. `fusion.py` computes per-user fused score from:
   - CNN probability
   - DOA alignment to user bearing
   - DOA reliability/confidence
5. `run_live_fusion.py` selects active speaker and sends Furhat attend commands.

## 5. Current Model Metadata In This Repo

From checked-in artifacts:

- `data/windows/windows_info.json`
  - `window_frames = 38`
  - `target_fps = 25.0`
  - `num_points = 76`
  - `feature_type = "nx"`
- `data/models/cnn_vvad/threshold.json`
  - `threshold = 0.3`

## 6. Minimal Repro Commands

Train path:

```bash
python scripts/step3_batch_extract_landmarks.py --min-rows 50
python scripts/step4_build_windows.py --window-sec 1.5 --hop-sec 0.5 --target-fps 25
python scripts/step5_train_cnn.py --val-video WwoTG3_OjUg --epochs 40 --batch-size 128
```

Live fused path:

```bash
python audio_doa/run_live_fusion.py \
  --furhat-ip <FURHAT_IP> \
  --show-child-logs \
  --step6-extra "--show --window-width 960 --window-height 540" \
  --doa-extra "--emit-idle --poll-hz 20"
```

## 7. How To Explain This To New Team Members

Use this sequence:

1. Start with the three-branch architecture (visual/audio/fusion).
2. Walk through the stage artifacts (`landmarks -> windows -> model -> realtime`).
3. Run one dry pass of each command category (offline then live).
4. Open `windows_info.json`, `config.json`, and `threshold.json` together to show model assumptions.
5. Show how `run_live_fusion.py` orchestrates child processes and sends Furhat attend actions.

This order keeps onboarding concrete and avoids diving into implementation details too early.
