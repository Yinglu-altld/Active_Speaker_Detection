# Audio DOA (ReSpeaker USB) + Live Fusion

This folder contains the audio/fusion side of the realtime system.

- `doa_core.py`: reads ReSpeaker built-in `direction` + `is_voice()` over USB and emits JSONL
- `respeaker_tuning.py`: low-level USB parameter read/write wrapper
- `run_live_fusion.py`: starts visual realtime + doa_core + fusion + Furhat attend control
- `fusion.py`: scoring logic
- `fusion_stub.py`: isolated fusion debugging

## Install

```bash
./venv/bin/pip install -r audio_doa/requirements.txt
```

USB dependency:

- macOS: `brew install libusb`
- Linux: install `libusb-1.0` and set USB permissions

## DOA Core Commands

Basic:

```bash
./venv/bin/python audio_doa/doa_core.py
```

Speech-only output:

```bash
./venv/bin/python audio_doa/doa_core.py --no-emit-idle
```

Custom USB IDs:

```bash
./venv/bin/python audio_doa/doa_core.py --usb-vendor-id 0x2886 --usb-product-id 0x0018
```

Valid CLI flags in current version:

- `--usb-vendor-id`
- `--usb-product-id`
- `--poll-hz`
- `--emit-idle` / `--no-emit-idle`
- `--max-frames`

## Angle Convention

ReSpeaker hardware reports `raw_azimuth_deg` in `[0, 360)`.

Current conversion in `doa_core.py`:

- `azimuth_deg = wrap(raw_azimuth_deg - 90)`
- `0°` = front (`+Z`)
- `+` = user-right (`+X`)
- `-` = user-left (`-X`)

This assumes board placement where hardware `90°` points to front/users.

## VAD Tuning

Current VAD threshold is configured in code, not CLI:

- `audio_doa/doa_core.py` -> `VAD_THRESHOLD_DB`

Related settings:

- `APPLY_VAD_THRESHOLD_ON_START`: whether to write threshold at startup
- If USB write causes instability on your machine, set it to `False`

## Run Live Fusion

```bash
./venv/bin/python audio_doa/run_live_fusion.py \
  --furhat-ip 192.168.1.109 \
  --show-child-logs \
  --step6-extra "--show --window-width 960 --window-height 540" \
  --doa-extra "--emit-idle --poll-hz 20"
```

This opens:

- `vvad_realtime` (camera + bbox)
- `vvad_scores` (separate score plots)

## Common Mistake After Migration

These old flags are no longer supported by `doa_core.py` and will crash startup:

- `--audio-device`
- `--audio-channels`
- `--vad-channel`
- `--vad-threshold`
- `--speech-hold-ms`

## Fusion Isolation Test

```bash
./venv/bin/python audio_doa/doa_core.py --no-emit-idle \
| ./venv/bin/python audio_doa/fusion_stub.py --text \
  --users u1,-26,0.80 \
  --users u2,154,0.30
```
