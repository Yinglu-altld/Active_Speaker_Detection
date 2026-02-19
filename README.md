# Active Speaker Detection (CNN + MediaPipe + Audio DOA + Fusion)

This project performs **real-time active speaker detection** (who is speaking now).

The core idea is to fuse two modalities:

- Visual: estimate each user's speaking probability from facial landmarks (CNN)
- Audio: estimate where speech comes from and how reliable that estimate is (DOA)
- Fusion: compute an overall score per `user_id`; the top score is the active speaker

---

## 1. Project Logic (Understand First)

### 1.1 CNN (Visual) Module

Related scripts:

- `scripts/step3_extract_landmarks.py`
- `scripts/step3_batch_extract_landmarks.py`
- `scripts/step4_build_windows.py`
- `scripts/step5_train_cnn.py`
- `scripts/step6_realtime_infer.py`

Pipeline:

1. Extract facial landmarks with MediaPipe
2. Build temporal feature windows
3. Train `TemporalCNN` to output per-user speaking probability
4. During realtime inference, output `cnn_prob` for each `user_id`

---

### 1.2 Audio DOA Module

Related scripts:

- `audio_doa/doa_core.py`
- `audio_doa/srp_phat.py`

Pipeline:

1. Capture multichannel microphone audio (for example ReSpeaker 6-channel)
2. Use VAD + energy gating to detect speech activity
3. Use SRP-PHAT to estimate `azimuth_deg` (source direction)
4. Output JSON fields such as `conf_doa`, `conf_doa_srp`, `audio_conf`, `sigma_deg`

---

### 1.3 Fusion / Integration Module

Related scripts:

- `audio_doa/fusion.py` (fusion scoring logic)
- `audio_doa/run_live_fusion.py` (one-command realtime entrypoint)

Fusion output:

- One `overall score` per `user_id`
- `speaker_id` = highest-scoring user
- Optional Furhat action: `attend.user`

---

## 2. From Clone to First Run (Step by Step)

### Step 0: Clone the repo

```bash
git clone https://github.com/Yinglu-altld/Active_Speaker_Detection.git
cd Active_Speaker_Detection
```

### Step 1: Create virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install -r audio_doa/requirements.txt
```

### Step 2: Check bundled runtime assets

The repository already includes the pretrained runtime files, so teammates can run inference directly after cloning:

- `data/models/cnn_vvad/best.pt`
- `data/models/cnn_vvad/config.json`
- `data/models/cnn_vvad/threshold.json`
- `data/windows/windows_info.json`
- `data/models/face_landmarker_v2.task`

You only need to provide extra data files such as `data/videos/*.mp4` and `data/labels/*-activespeaker.csv` if you want to retrain or run offline dataset processing.

---

## 3. Fastest Full Pipeline Run (Recommended)

Run the complete realtime chain (CNN + DOA + Fusion + Furhat):

```bash
./venv/bin/python audio_doa/run_live_fusion.py \
  --furhat-ip 192.168.1.109 \
  --attend-furhat \
  --audio-device 1 --audio-channels 6 --mic-channels 1,2,3,4 \
  --step6-extra "--show --window-width 960 --window-height 540"
```

This command internally starts:

- `scripts/step6_realtime_infer.py`
- `audio_doa/doa_core.py`
- fusion logic and Furhat control

---

## 4. Realtime Overlay Legend

Each face bounding box shows:

- `c`: CNN score
- `d`: DOA score for that user
- `o`: fused overall score
- `a`: active speaker flag (`1` or `0`)

Color mapping:

- Green: current active speaker
- Orange: non-active speaker (still tracked and scored)

---

## 5. Optional: Retrain the CNN Model

### 5.1 Batch extract landmarks

```bash
./venv/bin/python scripts/step3_batch_extract_landmarks.py --min-rows 50
```

### 5.2 Build training windows

```bash
./venv/bin/python scripts/step4_build_windows.py --window-sec 1.5 --hop-sec 0.5 --target-fps 25
```

### 5.3 Train and evaluate

```bash
./venv/bin/python scripts/step5_train_cnn.py --val-video WwoTG3_OjUg --epochs 40 --batch-size 128
./venv/bin/python scripts/step5_train_cnn.py --eval-only --test-video Ag-pXiLrd48
```

Common optional flags:

- `--no-delta`: disable delta features
- `--filter-extremes --neg-max 0.1 --pos-min 0.6`: use cleaner extreme labels

---

## 6. Visual-Only Realtime Mode (No Fusion)

Furhat camera source:

```bash
./venv/bin/python scripts/step6_realtime_infer.py --source furhat --furhat-ip 192.168.1.109 --show
```

Local webcam source:

```bash
./venv/bin/python scripts/step6_realtime_infer.py --source opencv --video-device 0 --show
```

---

## 7. Reference Data and Current Metrics

Primary video IDs currently used:

- `2bxKkUgcqpk`
- `9bK05eBt1GM`
- `Ag-pXiLrd48`
- `B1MAUxpKaV8`
- `WwoTG3_OjUg`
- `a5mEmM6w_ks`

Reference held-out test metrics (`Ag-pXiLrd48`):

- `test_f1=0.744`
- `test_acc=0.843`
- `test_pos_rate=0.251`
- `threshold=0.30`

---

## 8. Notes

- `data/` stores large/generated artifacts and is not tracked by default.
- For detailed audio-side docs, see `audio_doa/README.md`.
