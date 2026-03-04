# Audio + Fusion Module (ReSpeaker USB)

This folder contains realtime DOA ingest, fusion scoring, Furhat attend control, and optional button-based ground truth.

## Files

- `doa_core.py`: reads ReSpeaker `direction` and `is_voice()` over USB and emits JSON observations.
- `respeaker_tuning.py`: low-level USB register wrapper.
- `fusion.py`: per-user `cnn/doa/overall` scoring.
- `run_live_fusion.py`: launcher for step6 + DOA + fusion + Furhat + optional GT bridge.
- `button_gt_bridge.py`: receives UDP button packets (`u0`, `u1`) and emits GT JSONL.
- `fusion_stub.py`: quick scoring debug without full UI.

---

## Install

```bash
./venv/bin/pip install -r audio_doa/requirements.txt
```

USB dependency:

- macOS: `brew install libusb`
- Linux: install `libusb-1.0` and set USB permissions

---

## DOA observation fields

`doa_core.py` now emits both gated and ungated angle fields:

- `raw_azimuth_deg`: last speech-valid raw angle (legacy/gated behavior source).
- `azimuth_deg`: gated + offset angle used by fusion decisions.
- `raw_azimuth_plot_deg`: current raw hardware angle each poll (ungated).
- `azimuth_plot_deg`: current raw angle after offset each poll (ungated, for plotting).

This allows UI DOA plots/compass to stay ungated while keeping fusion logic unchanged.

---

## Angle convention

Hardware DOA is in `[0, 360)`. Project conversion:

```text
azimuth = wrap(raw_azimuth - 90)
```

Interpretation:

- `0°`: front (`+Z`)
- positive: user-right (`+X`)
- negative: user-left (`-X`)

---

## DOA core commands

```bash
./venv/bin/python audio_doa/doa_core.py
```

Continuous output (recommended for ungated plotting):

```bash
./venv/bin/python audio_doa/doa_core.py --emit-idle --poll-hz 20
```

Valid flags:

- `--usb-vendor-id`
- `--usb-product-id`
- `--poll-hz`
- `--emit-idle` / `--no-emit-idle`
- `--max-frames`

---

## VAD threshold tuning

Configured in code (`doa_core.py`):

- `VAD_THRESHOLD_DB`
- `APPLY_VAD_THRESHOLD_ON_START`

If writing threshold causes USB instability on your machine, disable write-on-start:

```python
APPLY_VAD_THRESHOLD_ON_START = False
```

---

## Run full live fusion

### With button ground-truth

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

### Without button ground-truth

```bash
./venv/bin/python audio_doa/run_live_fusion.py \
  --furhat-ip 192.168.1.109 \
  --show-child-logs \
  --doa-extra="--emit-idle --poll-hz 20" \
  --step6-extra="--show --window-width 960 --window-height 540"
```

---

## NodeMCU GT payload format

Send UDP JSON to the configured host/port:

```json
{"u0":1,"u1":0}
```

- `u0`: button 0 pressed state
- `u1`: button 1 pressed state
- pressed=`1`, released=`0`

Arduino IDE sketch:

- `../hardware/nodemcu_buttons/nodemcu_buttons.ino`

Set sketch constants before upload:

- `WIFI_SSID`
- `WIFI_PASSWORD`
- `DEST_IP`
- `DEST_PORT`

---

## Invalid legacy flags (do not use)

These were from the old audio-stream branch and are no longer supported in `doa_core.py`:

- `--audio-device`
- `--audio-channels`
- `--vad-channel`
- `--vad-threshold`
- `--speech-hold-ms`

---

## Fusion-only quick check

```bash
./venv/bin/python audio_doa/doa_core.py --no-emit-idle \
| ./venv/bin/python audio_doa/fusion_stub.py --text \
  --users u1,-26,0.80 \
  --users u2,154,0.30
```
