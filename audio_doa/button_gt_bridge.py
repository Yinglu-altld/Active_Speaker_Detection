import argparse
import json
import socket
import time
from pathlib import Path


def _to_binary(value) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "on", "pressed", "down"}:
            return 1
        if text in {"0", "false", "off", "released", "up"}:
            return 0
        try:
            return 1 if float(text) >= 0.5 else 0
        except ValueError:
            return 0
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive ESP8266 button UDP packets and emit GT JSONL for plots."
    )
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=5005)
    parser.add_argument("--output-jsonl", default="/tmp/gt_live.jsonl")
    parser.add_argument("--poll-hz", type=float, default=20.0)
    parser.add_argument(
        "--stale-sec",
        type=float,
        default=0.7,
        help="If no UDP packet arrives within this window, force both buttons to 0.",
    )
    parser.add_argument("--user0-id", default="user-0")
    parser.add_argument("--user1-id", default="user-1")
    parser.add_argument(
        "--button0-key",
        default="u0",
        help="JSON key for button 0 (e.g., u0 for D2).",
    )
    parser.add_argument(
        "--button1-key",
        default="u1",
        help="JSON key for button 1 (e.g., u1 for D6).",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def _extract_buttons(
    payload: dict,
    key0: str,
    key1: str,
    user0_id: str,
    user1_id: str,
) -> tuple[int | None, int | None]:
    source = payload.get("buttons")
    if not isinstance(source, dict):
        source = payload
    if not isinstance(source, dict):
        return None, None
    value0 = source.get(key0, source.get(user0_id))
    value1 = source.get(key1, source.get(user1_id))
    parsed0 = None if value0 is None else _to_binary(value0)
    parsed1 = None if value1 is None else _to_binary(value1)
    return parsed0, parsed1


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((str(args.listen_host), int(args.listen_port)))
    sock.setblocking(False)

    state = {
        str(args.user0_id): 0,
        str(args.user1_id): 0,
    }
    interval = 1.0 / max(float(args.poll_hz), 1e-3)
    stale_sec = max(0.0, float(args.stale_sec))
    max_frames = max(0, int(args.max_frames))
    emitted = 0
    last_rx_ts = 0.0
    next_emit_ts = time.time()

    print(
        f"[button-gt-bridge] listening udp {args.listen_host}:{args.listen_port} "
        f"-> {output_path}"
    )

    with open(output_path, "a", encoding="utf-8", buffering=1) as handle:
        try:
            while True:
                changed = False
                while True:
                    try:
                        raw_bytes, _addr = sock.recvfrom(4096)
                    except BlockingIOError:
                        break

                    raw = raw_bytes.decode("utf-8", errors="ignore").strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue

                    b0, b1 = _extract_buttons(
                        payload=payload,
                        key0=str(args.button0_key),
                        key1=str(args.button1_key),
                        user0_id=str(args.user0_id),
                        user1_id=str(args.user1_id),
                    )
                    if b0 is not None and b0 != state[str(args.user0_id)]:
                        state[str(args.user0_id)] = b0
                        changed = True
                    if b1 is not None and b1 != state[str(args.user1_id)]:
                        state[str(args.user1_id)] = b1
                        changed = True
                    last_rx_ts = time.time()

                now_ts = time.time()
                if stale_sec > 0.0 and last_rx_ts > 0.0 and (now_ts - last_rx_ts) > stale_sec:
                    if state[str(args.user0_id)] != 0 or state[str(args.user1_id)] != 0:
                        state[str(args.user0_id)] = 0
                        state[str(args.user1_id)] = 0
                        changed = True

                if changed or now_ts >= next_emit_ts:
                    line = {
                        "t": float(now_ts),
                        "gt": {
                            str(args.user0_id): int(state[str(args.user0_id)]),
                            str(args.user1_id): int(state[str(args.user1_id)]),
                        },
                    }
                    handle.write(json.dumps(line, separators=(",", ":")) + "\n")
                    handle.flush()
                    emitted += 1
                    next_emit_ts = now_ts + interval
                    print(
                        f"[button-gt-bridge] {args.user0_id}={state[str(args.user0_id)]} "
                        f"{args.user1_id}={state[str(args.user1_id)]}"
                    )
                    if max_frames and emitted >= max_frames:
                        break
                time.sleep(0.002)
            print("[button-gt-bridge] reached --max-frames, exiting.")
        except KeyboardInterrupt:
            print("\n[button-gt-bridge] interrupted by user.")
        finally:
            sock.close()


if __name__ == "__main__":
    main()
