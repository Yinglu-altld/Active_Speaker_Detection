# Furhat ASD Upgrade (Realtime API + ReSpeaker DOA/VAD)

This repo contains an external **Perception Controller** (runs on your PC) that:

- Connects to Furhat via the **Realtime API** (WebSocket) to receive `users` tracking (and camera frames if enabled).
- Captures audio from an external **ReSpeaker USB mic array**.
- Runs **VAD + DOA**, then picks the active Furhat `user_id` by fusing DOA with Furhat user positions.
- Sends low-latency attention commands back to Furhat (e.g. `request.attend.user`).

## Quick start (Windows)

### 0) PowerShell rule (important)

To run a file from the current folder, prefix it with `.\`:

- OK: `.\run_furhat_asd.cmd`
- OK: `python .\run_2speaker_test.py`
- NO: `run run_furhat_asd.cmd`
- NO: `run_2speaker_test.py`

### 1) Create a venv

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) Install (editable) + Silero VAD

```powershell
pip install -e ".[vad_silero]"
```

### 3) Create `config.json`

```powershell
Copy-Item config.example.json config.json
```

Then edit `config.json`:

- Set `furhat.ip` to your Furhat robot IP.
- If you use **ReSpeaker 4-Mic Array v2.0 (6-channel UAC1.0 mode)**:
  - Keep `audio.channels: 6`
  - Keep `audio.channel_indices: [1,2,3,4]` (raw mic channels)
  - Keep `audio.sample_rate: 16000`

### 4) Run the full controller

```powershell
.\run_furhat_asd.cmd
```

`run_furhat_asd.cmd` is a Windows-only convenience wrapper. The cross-platform “main command” is:

```powershell
python -m furhat_asd --config config.json
```

If the CLI entrypoint is installed, this also works:

```powershell
furhat-asd --config config.json
```

## Quick start (macOS / Linux)

macOS/Linux cannot run `.cmd` files. Use the Python command instead.

### 1) Create a venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 2) Install (editable) + Silero VAD

```bash
pip install -e ".[vad_silero]"
```

### 3) Create `config.json`

```bash
cp config.example.json config.json
```

Edit `config.json` and set `furhat.ip` to your Furhat robot IP.

### 4) Run the full controller (main command)

```bash
python -m furhat_asd --config config.json
```

## VAD

Supported VAD backends:

- `silero` (recommended; best quality; requires PyTorch)
- `webrtc` (good, but building `webrtcvad` on Windows needs MSVC Build Tools)
- `energy` (fallback; works everywhere; not robust in noise)

If `vad.silero_model_path` is empty, we load via the `silero-vad` PyPI package.

### VAD sanity check

```powershell
.\run_vad_test.cmd
```

You should see `speech_prob` go high (e.g. >0.6) when you speak normally.

## DOA (direction of arrival)

### DOA sanity check (audio-only; no Furhat needed)

If ReSpeaker is plugged into your PC:

```powershell
furhat-asd-doa-test --config config.doa_pc_test.json
```

### ReSpeaker 4-Mic Array v2.0 channel note

With the common **6-channel firmware**, channel `0` is a processed stream and channels **1-4** are the raw microphone signals (channel 5 is playback). For DOA, select channels 1-4 via `audio.channel_indices`.

On many systems, the **6-channel UAC1.0 mode** is only available at **16 kHz**. If you see "Invalid sample rate", set `audio.sample_rate` to `16000`.

### Calibrate channel order + offset (recommended)

Guided calibration:

```powershell
furhat-asd-doa-calibrate --config config.doa_pc_test.json --duration-s 2.5
```

This prints:

- the best `audio.channel_indices` order for your device
- an initial `control.doa_sign` and `control.doa_offset_deg` for your mounting

## Multi-user behavior (important)

- DOA gives **one dominant direction at a time**. If two people speak at the same time, this system will usually pick the louder / more dominant speaker. Separating overlapping speakers requires adding the CNN/vision score (late fusion) and/or diarization/beamforming.
- To map DOA -> `user_id`, Furhat must detect both people as `users` at the same time. If you see `users=0`, make sure faces are visible to the robot camera.

## Repo hygiene (remove "junk files")

Do not commit:

- `.venv/` (your local Python environment)
- `out/` (logs / segments)
- `__pycache__/` and `*.pyc`
- `config.json` (your local IPs + calibration)

If these were already committed, remove them from tracking and push a cleanup commit:

```powershell
git rm -r --cached .venv out
git rm --cached config.json
git rm -r --cached **/__pycache__
git rm -r --cached **/*.pyc
git commit -m "Cleanup: ignore local env/logs/caches"
git push
```
