from __future__ import annotations

import argparse
import asyncio

from furhat_asd.config import load_config
from furhat_asd.controller import run_controller


def main() -> None:
    parser = argparse.ArgumentParser(prog="furhat_asd")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--list-audio-devices", action="store_true", help="List audio input devices and exit")
    args = parser.parse_args()

    if args.list_audio_devices:
        try:
            import sounddevice as sd
        except Exception as e:  # pragma: no cover
            raise SystemExit("sounddevice is required to list devices") from e
        print(sd.query_devices())
        return

    config = load_config(args.config)
    try:
        asyncio.run(run_controller(config))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
