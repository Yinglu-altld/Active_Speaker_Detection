# Audio DOA (ReSpeaker Built-in) + Live Fusion

This folder contains the audio-side modules used by the realtime active speaker demo.

- `doa_core.py`: reads **ReSpeaker built-in DOA + VAD** via USB control (pyusb) and prints JSONL frames.
- `run_live_fusion.py`: single-command runner that starts visual inference + `doa_core.py`, fuses, and drives Furhat head.
- `fusion.py`: fusion scoring (CNN + DOA alignment).
- `fusion_stub.py`: quick fusion test harness.

## Install

```bash
./venv/bin/pip install -r audio_doa/requirements.txt
```

Notes:
- `pyusb` typically requires `libusb`.
  - macOS: `brew install libusb`
  - Linux: install your distro's `libusb-1.0` package and ensure permissions to access USB devices.

## DOA Core

Run and print JSON lines:

```bash
./venv/bin/python audio_doa/doa_core.py
```

Only emit speech-active frames:

```bash
./venv/bin/python audio_doa/doa_core.py --no-emit-idle
```

If your device IDs differ, override:

```bash
./venv/bin/python audio_doa/doa_core.py --usb-vendor-id 0x2886 --usb-product-id 0x0018
```

### Angle Conventions

The ReSpeaker firmware reports a hardware angle `raw_azimuth_deg` in `[0, 360)` (see the hardware overview).

This project converts it to the **robot bearing convention** used everywhere else:

- `azimuth_deg = wrap(raw_azimuth_deg - 90)`
- `0°` means **front** (`+Z`)
- `+` means **user-right** (`+X`)
- `-` means **user-left** (`-X`)

This assumes you place the board so that the hardware DOA `90°` points to the users/front.

## True Live Fusion (Recommended)

Single-command runner (starts Step 6 + DOA core internally):

```bash
./venv/bin/python audio_doa/run_live_fusion.py \
  --furhat-ip 192.168.1.108 \
  --step6-extra "--show --window-width 960 --window-height 540"
```

Pause-tolerant profile (recommended for natural pauses):

```bash
./venv/bin/python audio_doa/run_live_fusion.py \
  --furhat-ip 192.168.1.108 \
  --idle-after-no-speech-sec 5 \
  --send-hz 2.5 --switch-hits 4 \
  --step6-extra "--show --window-width 960 --window-height 540" \
  --doa-extra "--speech-hold-ms 900"
```

## Test Fusion Logic in Isolation

Pipe live DOA JSON into the fusion stub with static users:

```bash
./venv/bin/python audio_doa/doa_core.py --no-emit-idle \
| ./venv/bin/python audio_doa/fusion_stub.py --text \
  --users u1,-26,0.80 \
  --users u2,154,0.30
```

