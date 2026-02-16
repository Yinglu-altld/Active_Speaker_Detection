# ReSpeaker DOA + Furhat Tracker

This project tracks speaking direction from a multi-channel microphone array and drives Furhat attention in real time.

Core files:
- `doa_furhat.py`: main runtime loop (audio capture, VAD gating, DOA mapping, Furhat control).
- `srp_phat.py`: SRP-PHAT azimuth estimator with a confidence score.

## What It Does

1. Captures multi-channel audio from a USB microphone array.
2. Uses Silero VAD + energy/SNR gates to decide if speech is active.
3. Estimates DOA azimuth from mic channels using SRP-PHAT.
4. Maps azimuth to Furhat `request.attend.location` (`x,y,z`).
5. Optionally combines DOA with Furhat user locations (`hybrid` / `doa-user-match`).

## Requirements

- Python 3.10+ (recommended)
- A Furhat robot reachable over network
- A multi-channel input device for DOA (minimum 4 input channels)
- Packages in `requirements.txt`

Install:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Start

Default run:

```powershell
.venv\Scripts\python.exe doa_furhat.py
```

Default mode is `doa-user-match`.

## Main Run Modes

- `location`: pure DOA-driven `attend.location`.
- `closest-user`: repeatedly calls Furhat `attend.user` with `user_id=closest` (no DOA loop).
- `hybrid`: DOA intent + camera users blend.
- `doa-user-match` (default): DOA first, then tries matching DOA bearing to Furhat users; falls back to DOA location.

Examples:

```powershell
.venv\Scripts\python.exe doa_furhat.py --attend-mode location
.venv\Scripts\python.exe doa_furhat.py --attend-mode hybrid
.venv\Scripts\python.exe doa_furhat.py --attend-mode doa-user-match
```

## Current Defaults (Important)

Audio and DOA:
- `--fs 16000`
- `--channels 6`
- `--mic-channels 1,2,3,4`
- `--frame-ms 80`
- `--srp-az-step-deg 2.0`
- `--srp-interp 4`
- `--srp-f-low-hz 300`
- `--srp-f-high-hz 3400`

Speech gates:
- `--vad-threshold 0.22`
- `--vad-update-threshold 0.30`
- `--energy-threshold 80`
- `--energy-update-threshold 140`
- `--snr-speech-ratio 1.4`
- `--snr-speech-add 20`
- `--snr-update-ratio 1.7`
- `--snr-update-add 35`
- `--speech-hold-ms 140`

Mapping and motion:
- `--board-zero-offset -45`
- `--front-only` enabled
- `--max-az-deg 60`
- `--az-gain 0.70`
- `--target-distance-m 1.2`
- `--target-y-m 0.0`
- `--flip-x` enabled

Smoothing:
- `--smooth-alpha 0.32`
- `--lock-alpha 0.30`
- `--min-consistent-updates 1`
- `--consistency-deg 18`
- `--speaker-switch-deg 25`
- `--speaker-switch-updates 1`
- `--doa-quality-threshold 0.15`
- `--min-update-frames 1`
- `--speech-hold-ms 140`

## Device Selection

By default, the script uses `--device 0`.

If DOA does not respond, explicitly set device and channels:

```powershell
.venv\Scripts\python.exe doa_furhat.py --device <INDEX> --channels 6 --mic-channels 1,2,3,4
```

To list devices from Python:

```powershell
.venv\Scripts\python.exe -c "import sounddevice as sd; print(sd.query_devices())"
```

Notes:
- DOA requires a real multi-channel endpoint. Mono endpoints cannot estimate direction.
- If you select a 4-channel endpoint, the script auto-adjusts channels and mic indices when possible.

## Log Output

When tracking:

```text
[track] mode=... src=... doa=... conf=... vad=... e=... az=... xyz=(x,y,z)
```

Meaning:
- `doa`: raw SRP azimuth (degrees).
- `conf`: SRP confidence from `srp_phat.py`.
- `vad`: Silero speech probability.
- `e`: mean absolute energy of the selected VAD mic channel.
- `az`: smoothed/locked azimuth used for control.
- `xyz`: target sent to Furhat.

When idle:

```text
[idle] vad=... e=... noise=... gate=...
```

## Furhat Connection

Defaults:
- `--furhat-ip 192.168.1.108`
- `--furhat-port 9000`
- `--furhat-auth-key ""`

If your Furhat requires auth:

```powershell
.venv\Scripts\python.exe doa_furhat.py --furhat-ip <FURHAT_IP> --furhat-auth-key "<KEY>"
```

## Left/Right Calibration Tips

- If left/right is mirrored: toggle `--flip-x` / `--no-flip-x`.
- If board orientation differs: try `--board-cw` and `--board-zero-offset`.
- If front/back appears inverted: try `--add-180`.

## What Is Not In This Version

- No `--audio-json` flag.
- No debug JSON payload output from `doa_furhat.py`.
- Confidence is the current SRP score from `srp_phat.py` (`0.5*sharpness + 0.5*contrast`), not a multi-term fusion metric.
