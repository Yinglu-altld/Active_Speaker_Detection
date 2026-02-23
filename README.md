# Active Speaker Detection (CNN + MediaPipe + ReSpeaker DOA + Fusion)

This project performs realtime active speaker detection for Furhat.

- Visual branch: CNN on temporal face landmarks (`scripts/step6_realtime_infer.py`)
- Audio branch: ReSpeaker built-in DOA + VAD over USB (`audio_doa/doa_core.py`)
- Fusion branch: per-user overall score + Furhat head control (`audio_doa/run_live_fusion.py`)

## 1. Quick Setup

```bash
git clone https://github.com/Yinglu-altld/Active_Speaker_Detection.git
cd Active_Speaker_Detection
python -m venv venv
source venv/bin/activate
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install -r audio_doa/requirements.txt
```

USB dependency for ReSpeaker:

- macOS: `brew install libusb`
- Linux: install `libusb-1.0` package and ensure USB permissions

## 2. Runtime Files

Current realtime inference expects:

- `data/models/cnn_vvad/best.pt`
- `data/models/cnn_vvad/config.json`
- `data/models/cnn_vvad/threshold.json`
- `data/windows/windows_info.json`
- `data/models/face_landmarker_v2.task`

## 3. Run Full Live Fusion (Recommended)

```bash
./venv/bin/python audio_doa/run_live_fusion.py \
  --furhat-ip 192.168.1.109 \
  --show-child-logs \
  --step6-extra "--show --window-width 960 --window-height 540" \
  --doa-extra "--emit-idle --poll-hz 20"
```

What this starts internally:

- `scripts/step6_realtime_infer.py`
- `audio_doa/doa_core.py`
- Fusion + Furhat attend control

UI windows:

- `vvad_realtime`: camera + bbox + state
- `vvad_scores`: separate 3-panel plots (`CNN`, `DOA`, `Overall`) in range `[0, 1]`

## 4. Important Flag Notes (Current Version)

`audio_doa/doa_core.py` now uses ReSpeaker USB control, so old audio-stream flags are not valid anymore.

Invalid old flags (do not use in `--doa-extra`):

- `--audio-device`
- `--audio-channels`
- `--vad-channel`
- `--vad-threshold`
- `--speech-hold-ms`

Valid `doa_core.py` flags:

- `--usb-vendor-id`
- `--usb-product-id`
- `--poll-hz`
- `--emit-idle` / `--no-emit-idle`
- `--max-frames`

## 5. VAD / Angle Tuning (Code-Level)

Tune in `audio_doa/doa_core.py`:

- `VAD_THRESHOLD_DB`: ReSpeaker built-in VAD threshold (dB)
- `DOA_AZ_OFFSET_DEG`: angle offset (`-90` means hardware `90°` maps to front)
- `APPLY_VAD_THRESHOLD_ON_START`: whether to write threshold to device on startup

If your USB stack is unstable when writing VAD threshold, set `APPLY_VAD_THRESHOLD_ON_START = False`.

## 6. Visual-Only Realtime

Furhat camera:

```bash
./venv/bin/python scripts/step6_realtime_infer.py \
  --source furhat \
  --furhat-ip 192.168.1.109 \
  --show \
  --window-width 960 --window-height 540
```

Local webcam:

```bash
./venv/bin/python scripts/step6_realtime_infer.py --source opencv --video-device 0 --show
```

## 7. Optional Training Pipeline

```bash
./venv/bin/python scripts/step3_batch_extract_landmarks.py --min-rows 50
./venv/bin/python scripts/step4_build_windows.py --window-sec 1.5 --hop-sec 0.5 --target-fps 25
./venv/bin/python scripts/step5_train_cnn.py --val-video WwoTG3_OjUg --epochs 40 --batch-size 128
```

## 8. More Audio Details

See `audio_doa/README.md` for DOA-only and audio/fusion debugging commands.
