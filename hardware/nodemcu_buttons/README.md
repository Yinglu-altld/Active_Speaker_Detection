# NodeMCU Button Ground-Truth Sender

This sketch sends two button states over UDP for the runtime `Reality (Buttons)` plot.

## Files

- `nodemcu_buttons.ino`

## Wiring

- Button 0: `D2` to `G` (ground)
- Button 1: `D6` to `G` (ground)
- Buttons are configured as `INPUT_PULLUP` (pressed = low level).

## Payload

UDP JSON sent at ~20 Hz (and immediately on changes):

```json
{"u0":1,"u1":0}
```

## Required sketch configuration

Set these constants in `nodemcu_buttons.ino`:

- `WIFI_SSID`
- `WIFI_PASSWORD`
- `DEST_IP` (machine running `run_live_fusion.py`)
- `DEST_PORT` (default `5005`)
