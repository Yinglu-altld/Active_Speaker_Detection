from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import Any

from furhat_asd.config import load_config
from furhat_asd.model import FurhatUser
from furhat_asd.realtime.client import FurhatRealtimeClient
from furhat_asd.fusion.user_selector import user_azimuth_deg


def _parse_users_payload(users_payload: list[dict[str, Any]]) -> list[FurhatUser]:
    out: list[FurhatUser] = []
    for u in users_payload:
        if not isinstance(u, dict):
            continue
        user_id = u.get("id") or u.get("user_id")
        pos = u.get("location") or u.get("position") or u.get("pos")
        x = y = z = None
        if isinstance(pos, dict):
            x = pos.get("x", None)
            y = pos.get("y", None)
            z = pos.get("z", None)
        elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
            x, y, z = pos[0], pos[1], pos[2]
        try:
            x_m = float(x) if x is not None else None
            y_m = float(y) if y is not None else None
            z_m = float(z) if z is not None else None
        except Exception:
            x_m = y_m = z_m = None
        if user_id:
            out.append(FurhatUser(user_id=str(user_id), x_m=x_m, y_m=y_m, z_m=z_m, raw=u))
    return out


async def run_users_monitor(cfg_path: str, interval_ms: int) -> None:
    cfg = load_config(cfg_path)
    logging.basicConfig(level=getattr(logging, cfg.logging.level.upper(), logging.INFO))
    logging.getLogger("websockets").setLevel(logging.INFO)
    log = logging.getLogger("furhat_asd.users_monitor")

    client = FurhatRealtimeClient(cfg.furhat.ip, cfg.furhat.ws_port, cfg.furhat.api_key)
    await client.connect()
    log.info("Connected to Furhat Realtime API at %s", client.ws_url)
    await client.start_users()
    try:
        await client.users_once()
    except Exception:
        pass

    last_print = 0.0
    async for msg in client.messages():
        payload = msg.data.get("users")
        if not isinstance(payload, list):
            continue

        users = _parse_users_payload(payload)
        now = time.time()
        if (now - last_print) * 1000.0 < float(interval_ms):
            continue
        last_print = now

        rows: list[dict[str, Any]] = []
        for u in users:
            az = user_azimuth_deg(u)
            cam = u.raw.get("camera") if isinstance(u.raw, dict) else None
            rows.append(
                {
                    "id": u.user_id,
                    "x": u.x_m,
                    "z": u.z_m,
                    "az_deg": az,
                    "camera": cam if isinstance(cam, dict) else None,
                }
            )
        log.info("users=%d %s", len(rows), rows)


def main() -> None:
    parser = argparse.ArgumentParser(prog="furhat-asd-users-monitor")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--interval-ms", type=int, default=1000, help="Print interval (default: 1000ms)")
    args = parser.parse_args()

    try:
        asyncio.run(run_users_monitor(args.config, args.interval_ms))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()

