# Active Speaker Detection for Furhat

Realtime active-speaker system built from:

- **Visual branch**: MediaPipe face landmarks + temporal CNN (`scripts/step6_realtime_infer.py`)
- **Audio branch**: ReSpeaker USB built-in DOA + built-in VAD (`audio_doa/doa_core.py`)
- **Fusion/control branch**: user scoring + Furhat attend control (`audio_doa/run_live_fusion.py`)
- **Optional ground-truth branch**: NodeMCU button stream for reality plot (`audio_doa/button_gt_bridge.py`)

---

## 1) What changed in this version

- ReSpeaker-based pipeline (no SRP-PHAT runtime path in live fusion).
- Added optional UDP button GT integration (`u0`, `u1`) for a per-user **Reality** plot.
- Added separate DOA compass panel showing **ungated + offset azimuth** (`azimuth_plot_deg`).
- UI score behavior:
  - `CNN` plot: ungated raw CNN probability.
  - `DOA` plot: ungated DOA-alignment score.
  - `Overall` plot: unchanged fusion output (gated by runtime state logic).

---

## 2) Setup

```bash
git clone https://github.com/Yinglu-altld/Active_Speaker_Detection.git
cd Active_Speaker_Detection
python -m venv venv
source venv/bin/activate
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install -r audio_doa/requirements.txt
```

ReSpeaker USB dependency:

- macOS: `brew install libusb`
- Linux: install `libusb-1.0` and ensure device permissions

Required model/runtime files:

- `data/models/cnn_vvad/best.pt`
- `data/models/cnn_vvad/config.json`
- `data/models/cnn_vvad/threshold.json`
- `data/windows/windows_info.json`
- `data/models/face_landmarker_v2.task`

---

## 3) Coordinate convention

ReSpeaker hardware gives `raw_azimuth_deg` in `[0, 360)`.

Current project conversion:

```text
azimuth_deg = wrap(raw_azimuth_deg - 90)
```

With this mounting:

- `azimuth_deg = 0` → front (`+Z`)
- `azimuth_deg > 0` → user-right (`+X`)
- `azimuth_deg < 0` → user-left (`-X`)

`azimuth_plot_deg` is the same conversion but published continuously for plotting (ungated).

---

## 4) Run full system (recommended)

### A) With NodeMCU button GT (Reality plot)

```bash
./venv/bin/python audio_doa/run_live_fusion.py \
  --furhat-ip 192.168.1.109 \
  --show-child-logs \
  --use-button-gt \
  --gt-listen-port 5005 \
  --gt-user0-id user-0 --gt-user1-id user-1 \
  --gt-button0-key u0 --gt-button1-key u1 \
  --doa-extra="--emit-idle --poll-hz 20" \
  --step6-extra="--show --window-width 960 --window-height 540"
```

### B) Without button GT

```bash
./venv/bin/python audio_doa/run_live_fusion.py \
  --furhat-ip 192.168.1.109 \
  --show-child-logs \
  --doa-extra="--emit-idle --poll-hz 20" \
  --step6-extra="--show --window-width 960 --window-height 540"
```

---

## 5) NodeMCU button payload (for GT)

Send UDP JSON to host/port configured in `run_live_fusion.py` (default `5005`):

```json
{"u0":1,"u1":0}
```

- `u0` usually maps to D2 button, `u1` to D6 button.
- `1` = pressed, `0` = released.

---

## 6) UI windows

- `vvad_realtime`: camera, bbox, state text, per-bbox user id and scores.
- `vvad_scores`:
  - top state text + DOA compass (`Ungated+Offset` azimuth),
  - `CNN / DOA / Overall / Reality` per-user time series.

Notes:

- User colors are shared across all plot panels.
- Furhat user IDs are remapped to stable aliases (`user-0`, `user-1`, ...), so plots and bbox labels match.

---

## 7) Important flags and migration note

`audio_doa/doa_core.py` is USB-control based. Old audio-stream flags are invalid and will fail:

- `--audio-device`
- `--audio-channels`
- `--vad-channel`
- `--vad-threshold`
- `--speech-hold-ms`

Use `--doa-extra` only with valid DOA-core flags, e.g.:

- `--emit-idle` / `--no-emit-idle`
- `--poll-hz`
- `--usb-vendor-id`
- `--usb-product-id`

---

## 8) Tuning points (code-level)

In `audio_doa/doa_core.py`:

- `DOA_AZ_OFFSET_DEG`
- `VAD_THRESHOLD_DB`
- `APPLY_VAD_THRESHOLD_ON_START`

If USB becomes unstable when writing VAD threshold, set:

```python
APPLY_VAD_THRESHOLD_ON_START = False
```

---

## 9) Visual-only run (no fusion)

```bash
./venv/bin/python scripts/step6_realtime_infer.py \
  --source furhat \
  --furhat-ip 192.168.1.109 \
  --show \
  --window-width 960 --window-height 540
```

---

## 10) Audio docs

For audio-only commands and lower-level details, see:

- `audio_doa/README.md`
