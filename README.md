# Minimal ReSpeaker DOA -> Furhat (Silero VAD)

`doa_furhat.py` does only this:
- gate audio with Silero VAD
- estimate DOA azimuth from ReSpeaker raw 4-mic ring channels (SRP-PHAT in `srp_phat.py`)
- send `request.attend.location` (`x,y,z`) to Furhat realtime WS API

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python doa_furhat.py
```

Soft/medium speech pickup (recommended starting point):

```powershell
python doa_furhat.py --snr-speech-ratio 1.4 --snr-speech-add 20 --snr-update-ratio 1.7 --snr-update-add 35 --energy-threshold 80 --energy-update-threshold 140
```

If auth key is enabled:

```powershell
python doa_furhat.py --furhat-ip 192.168.1.108 --furhat-auth-key "YOUR_KEY"
```

If you want to override defaults quickly:

```powershell
python doa_furhat.py --furhat-ip 192.168.1.108 --board-zero-offset -45 --no-board-cw --flip-x
```

Hybrid mode (VAD trigger + DOA side intent + camera user location):

```powershell
python doa_furhat.py --attend-mode hybrid
```

DOA-first + user-coordinate match mode:

```powershell
python doa_furhat.py --attend-mode doa-user-match
```

SRP-PHAT tuning (optional):

```powershell
python doa_furhat.py --srp-az-step-deg 2.0 --srp-interp 4
```

## Notes

- ReSpeaker raw setup provides azimuth, not elevation. `y` is fixed via `--target-y-m`.
- If left/right is mirrored, toggle `--board-cw` vs `--no-board-cw`.
- If still mirrored in user view, add `--flip-x`.
- If front/back is opposite, add `--add-180`.
- For one-user-in-front setup, keep `--front-only` and set `--max-az-deg` (e.g. 55-65).
- If you want robust one-user following without DOA mapping ambiguity, use `--attend-mode closest-user` (still VAD-gated).
