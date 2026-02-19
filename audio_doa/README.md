# Minimal ReSpeaker DOA Core + Furhat Adapter (Silero VAD)

Primary entry point: `doa_core.py` (prints JSON `DOAObservation` only).

Adapter: `doa_furhat.py` (uses the core and sends `request.attend.location` to Furhat).

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python doa_core.py
```

If you want only speech frames:

```powershell
python doa_core.py --no-emit-idle
```

Tune gates / SRP-PHAT (optional):

```powershell
python doa_core.py --snr-speech-ratio 1.4 --snr-speech-add 20 --snr-update-ratio 1.7 --snr-update-add 35 --energy-threshold 80 --energy-update-threshold 140
```

SRP-PHAT tuning:

```powershell
python doa_core.py --srp-az-step-deg 2.0 --srp-interp 4
```

Furhat adapter:

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

## JSON Output Contract (`doa_core.py`)

Each frame is one JSON line:

```json
{
  "t": 1739701000.12,
  "azimuth_deg": 118.0,
  "conf_doa": 0.61,
  "conf_doa_srp": 0.84,
  "sigma_deg": 11.4,
  "entropy": 0.33,
  "conf_components": {
    "peak_ratio_score": 0.79,
    "contrast_score": 0.67,
    "entropy_score": 0.67,
    "sigma_score": 0.63,
    "sharpness_score": 0.54,
    "peak_ratio_raw": 2.58,
    "audio_conf": 0.73,
    "vad_conf": 0.81,
    "snr_conf": 0.55
  },
  "peaks": [
    {"azimuth_deg": 118.0, "score": 53.0, "score_norm": 1.0, "score_prob": 0.20},
    {"azimuth_deg": 116.0, "score": 49.1, "score_norm": 0.93, "score_prob": 0.18}
  ],
  "vad_prob": 0.78,
  "speech_prob": 0.72,
  "snr_db": 11.6,
  "audio_conf": 0.73,
  "energy": 211.0,
  "noise_energy": 95.0,
  "speech_gate_energy": 206.0,
  "update_gate_energy": 269.0,
  "speech_detected": true,
  "speech_active": true,
  "speech_ended": false,
  "allow_update": true,
  "doa_updated": true
}
```

`conf_doa` uses:

```text
conf_doa = conf_doa_srp
```

Where:
- `conf_doa_srp` comes from SRP-PHAT peak quality (peak ratio, contrast, entropy, sigma).
- `audio_conf` remains a separate reliability signal from VAD + SNR.
- Fusion combines reliability as `0.6 * conf_doa_srp + 0.4 * audio_conf`.

## Fusion Options

You now have both integration styles:

1. Drop-in function (`fusion.py`) for direct embedding.
2. Runnable stub (`fusion_stub.py`) for fast end-to-end score checks.

### 1) Drop-in function (`fusion.py`)

```python
from audio_doa.fusion import UserEvidence, score_users_for_frame

users = [
    UserEvidence(user_id="u1", bearing_deg=-20.0, cnn_prob=0.82),
    UserEvidence(user_id="u2", bearing_deg=35.0, cnn_prob=0.31),
]
result = score_users_for_frame(doa_obs, users)
print(result["speaker_id"], result["speaker_score"])
```

### 2) Runnable fusion stub (`fusion_stub.py`)

Read DOA JSONL from stdin and use static users:

```powershell
python doa_core.py --device 1 --channels 6 --mic-channels 1,2,3,4 --no-emit-idle ^
| python fusion_stub.py --text ^
  --users u1,-26,0.80 ^
  --users u2,154,0.30
```

Use CNN snapshots from file:

```powershell
python fusion_stub.py --doa-jsonl doa.jsonl --cnn-jsonl cnn.jsonl --text
```

`cnn.jsonl` format (one line per snapshot):

```json
{"t":1739701000.10,"users":[{"user_id":"u1","bearing_deg":-26,"cnn_prob":0.84},{"user_id":"u2","bearing_deg":154,"cnn_prob":0.22}]}
```

Export `cnn.jsonl` directly from Step 6:

```powershell
python ../scripts/step6_realtime_infer.py --source furhat --model-dir ../data/models/cnn_vvad --emit-cnn-jsonl cnn.jsonl
```

Then run fusion on recorded logs:

```powershell
python fusion_stub.py --doa-jsonl doa.jsonl --cnn-jsonl cnn.jsonl --text
```

### True Live Fusion (with Furhat attend)

Single command runner (recommended):

```powershell
python run_live_fusion.py --furhat-ip 192.168.1.108 --audio-device 1 --audio-channels 6 --mic-channels 1,2,3,4 --step6-extra "--show --window-width 960 --window-height 540"
```

Pause-tolerant profile (recommended for natural speaking pauses):

```powershell
python run_live_fusion.py --furhat-ip 192.168.1.108 --audio-device 1 --audio-channels 6 --mic-channels 1,2,3,4 --speech-hold-sec 1.8 --max-doa-staleness-sec 1.2 --send-hz 2.5 --switch-hits 4 --step6-extra "--show --window-width 960 --window-height 540" --doa-extra "--vad-threshold 0.18 --vad-update-threshold 0.22 --speech-hold-ms 900"
```

VAD-priority profile (suppress impulse-noise triggers):

```powershell
python run_live_fusion.py --furhat-ip 192.168.1.108 --audio-device 1 --audio-channels 6 --mic-channels 1,2,3,4 --step6-extra "--show --window-width 960 --window-height 540" --doa-extra "--vad-threshold 0.18 --vad-update-threshold 0.22 --speech-hold-ms 300 --energy-threshold 99999 --energy-update-threshold 99999 --snr-speech-ratio 999 --snr-speech-add 99999 --snr-update-ratio 999 --snr-update-add 99999"
```

This command starts `step6_realtime_infer.py` and `doa_core.py` internally, fuses in realtime, and sends Furhat attend commands.
In the Step6 window, each bbox shows:
- `c`: CNN score
- `d`: DOA score for that user
- `o`: overall fused score
- `a`: active speaker flag (`1` active, `0` inactive)

Active speaker bbox is green; other fused users are orange.

You can tune audio device quickly:

```powershell
python run_live_fusion.py --audio-device 1 --audio-channels 6 --mic-channels 1,2,3,4
```

Run these in parallel terminals.

Terminal A (CNN -> live snapshots):

```powershell
python ../scripts/step6_realtime_infer.py --source furhat --model-dir ../data/models/cnn_vvad --emit-cnn-jsonl /tmp/cnn_live.jsonl
```

Terminal B (DOA -> fusion -> Furhat):

```powershell
python doa_core.py --device 1 --channels 6 --mic-channels 1,2,3,4 --no-emit-idle | python fusion_stub.py --doa-jsonl - --cnn-jsonl-live /tmp/cnn_live.jsonl --text --attend-furhat --furhat-ip 192.168.1.108
```

If you require 3 terminals, use a FIFO:

```powershell
mkfifo /tmp/doa_live.pipe
# Terminal B:
python doa_core.py --device 1 --channels 6 --mic-channels 1,2,3,4 --no-emit-idle > /tmp/doa_live.pipe
# Terminal C:
python fusion_stub.py --doa-jsonl /tmp/doa_live.pipe --cnn-jsonl-live /tmp/cnn_live.jsonl --text --attend-furhat --furhat-ip 192.168.1.108
```

## Notes

- ReSpeaker raw setup provides azimuth, not elevation. `y` is fixed via `--target-y-m`.
- If left/right is mirrored, toggle `--board-cw` vs `--no-board-cw`.
- If still mirrored in user view, add `--flip-x`.
- If front/back is opposite, add `--add-180`.
- For one-user-in-front setup, keep `--front-only` and set `--max-az-deg` (e.g. 55-65).
- If you want robust one-user following without DOA mapping ambiguity, use `--attend-mode closest-user` (still VAD-gated).
