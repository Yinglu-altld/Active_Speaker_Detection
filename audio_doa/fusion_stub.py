import argparse
import json
import os
import sys
import time
from typing import Iterable

try:
    import websocket
except ImportError:
    websocket = None

try:
    from .fusion import FusionConfig, UserEvidence, score_users_for_frame
except ImportError:
    from fusion import FusionConfig, UserEvidence, score_users_for_frame


class FurhatWS:
    def __init__(
        self,
        ip: str,
        port: int,
        auth_key: str = "",
        speed: str = "medium",
        slack_pitch: float = 15.0,
        slack_yaw: float = 10.0,
        slack_timeout: int = 3000,
    ):
        if websocket is None:
            raise ImportError(
                "websocket-client is required for --attend-furhat. Install with: pip install websocket-client"
            )
        self.url = f"ws://{ip}:{port}/v1/events"
        self.auth_key = auth_key
        self.speed = speed
        self.slack_pitch = float(slack_pitch)
        self.slack_yaw = float(slack_yaw)
        self.slack_timeout = int(slack_timeout)
        self.ws = None

    def _connect(self) -> None:
        self.ws = websocket.create_connection(self.url, timeout=0.6)
        auth_payload = {"type": "request.auth"}
        if self.auth_key:
            auth_payload["key"] = self.auth_key
        self.ws.send(json.dumps(auth_payload))

    def _send(self, payload: dict) -> None:
        if self.ws is None:
            self._connect()
        try:
            self.ws.send(json.dumps(payload))
        except Exception:
            try:
                if self.ws is not None:
                    self.ws.close()
            finally:
                self.ws = None
            self._connect()
            self.ws.send(json.dumps(payload))

    def attend_user(self, user_id: str) -> None:
        self._send(
            {
                "type": "request.attend.user",
                "user_id": str(user_id),
                "speed": self.speed,
                "slack_pitch": self.slack_pitch,
                "slack_yaw": self.slack_yaw,
                "slack_timeout": self.slack_timeout,
            }
        )


class LiveCNNSnapshots:
    def __init__(self, path: str):
        self.path = path
        self.handle = None
        self.offset = 0
        self.latest = None

    def _ensure_handle(self) -> None:
        if self.handle is not None:
            return
        if not os.path.exists(self.path):
            return
        self.handle = open(self.path, "r", encoding="utf-8")
        self.handle.seek(self.offset)

    def poll(self) -> None:
        if os.path.exists(self.path) and os.path.getsize(self.path) < self.offset:
            if self.handle is not None:
                self.handle.close()
            self.handle = None
            self.offset = 0

        self._ensure_handle()
        if self.handle is None:
            return

        while True:
            line = self.handle.readline()
            if not line:
                break
            self.offset = self.handle.tell()
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            t_value = item.get("t")
            users = _parse_users_list(item.get("users"))
            if t_value is None or not users:
                continue
            self.latest = {"t": float(t_value), "users": users}

    def users_for_time(self, t_value: float | None, max_staleness_sec: float) -> list[UserEvidence]:
        self.poll()
        if self.latest is None:
            return []
        if t_value is not None:
            age = abs(float(t_value) - float(self.latest["t"]))
            if age > max_staleness_sec:
                return []
        return self.latest["users"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime fusion: consume DOA JSON lines and produce speaker scores (optional Furhat attend)."
    )
    parser.add_argument(
        "--doa-jsonl",
        default="-",
        help="DOA JSONL input path; use '-' for stdin.",
    )
    parser.add_argument(
        "--cnn-jsonl",
        default=None,
        help="Optional JSONL snapshots with {'t':..., 'users':[...]} (offline replay mode).",
    )
    parser.add_argument(
        "--cnn-jsonl-live",
        default=None,
        help="Live-growing CNN JSONL path (tail mode).",
    )
    parser.add_argument(
        "--max-cnn-staleness-sec",
        type=float,
        default=0.8,
        help="Max |t_doa - t_cnn| allowed for live snapshot sync.",
    )
    parser.add_argument(
        "--users",
        action="append",
        default=[],
        help="Static user: user_id,bearing_deg,cnn_prob[,face_conf[,track_conf]]",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many per-user rows to print in text mode.",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Print compact text lines instead of full JSON.",
    )
    parser.add_argument("--low-conf-th", type=float, default=0.03)
    parser.add_argument("--mid-conf-th", type=float, default=0.07)
    parser.add_argument("--low-srp-th", type=float, default=0.08)
    parser.add_argument("--low-audio-th", type=float, default=0.25)
    parser.add_argument("--weak-doa-weight", type=float, default=0.10)
    parser.add_argument("--mid-doa-weight", type=float, default=0.25)
    parser.add_argument("--strong-doa-weight", type=float, default=0.40)
    parser.add_argument("--default-sigma-deg", type=float, default=25.0)
    parser.add_argument("--min-sigma-deg", type=float, default=8.0)
    parser.add_argument("--attend-furhat", action="store_true")
    parser.add_argument("--furhat-ip", default="192.168.1.109")
    parser.add_argument("--furhat-port", type=int, default=9000)
    parser.add_argument("--furhat-auth-key", default="")
    parser.add_argument("--attend-speed", default="medium")
    parser.add_argument("--slack-pitch", type=float, default=15.0)
    parser.add_argument("--slack-yaw", type=float, default=10.0)
    parser.add_argument("--slack-timeout", type=int, default=3000)
    parser.add_argument("--min-speaker-score", type=float, default=0.30)
    parser.add_argument("--switch-hits", type=int, default=3)
    parser.add_argument("--switch-margin", type=float, default=0.03)
    parser.add_argument("--send-hz", type=float, default=4.0)
    return parser.parse_args()


def _open_lines(path: str):
    if path == "-":
        return sys.stdin
    return open(path, "r", encoding="utf-8")


def _parse_static_users(entries: Iterable[str]) -> list[UserEvidence]:
    users: list[UserEvidence] = []
    for item in entries:
        parts = [part.strip() for part in item.split(",")]
        if len(parts) < 3:
            raise ValueError(f"Invalid --users entry: {item}")
        user_id = parts[0]
        bearing_deg = float(parts[1])
        cnn_prob = float(parts[2])
        face_conf = float(parts[3]) if len(parts) >= 4 else 1.0
        track_conf = float(parts[4]) if len(parts) >= 5 else 1.0
        users.append(
            UserEvidence(
                user_id=user_id,
                bearing_deg=bearing_deg,
                cnn_prob=cnn_prob,
                face_conf=face_conf,
                track_conf=track_conf,
            )
        )
    return users


def _parse_users_list(users_data) -> list[UserEvidence]:
    users: list[UserEvidence] = []
    if not isinstance(users_data, list):
        return users
    for item in users_data:
        if not isinstance(item, dict):
            continue
        user_id = str(item.get("user_id") or item.get("track_id") or item.get("id") or "")
        if not user_id:
            continue
        if "bearing_deg" not in item:
            continue
        users.append(
            UserEvidence(
                user_id=user_id,
                bearing_deg=float(item["bearing_deg"]),
                cnn_prob=float(item.get("cnn_prob", item.get("prob", 0.0))),
                face_conf=float(item.get("face_conf", 1.0)),
                track_conf=float(item.get("track_conf", 1.0)),
            )
        )
    return users


def _read_cnn_snapshots(path: str | None) -> list[dict]:
    if not path:
        return []
    snapshots = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            t_value = item.get("t")
            users = _parse_users_list(item.get("users"))
            if t_value is None or not users:
                continue
            snapshots.append({"t": float(t_value), "users": users})
    snapshots.sort(key=lambda snap: snap["t"])
    return snapshots


def _select_users_for_time(
    t_value: float | None,
    static_users: list[UserEvidence],
    snapshots: list[dict],
    index: int,
) -> tuple[list[UserEvidence], int]:
    if static_users:
        return static_users, index
    if not snapshots:
        return [], index
    if t_value is None:
        return snapshots[index]["users"], index
    while index + 1 < len(snapshots) and snapshots[index + 1]["t"] <= float(t_value):
        index += 1
    return snapshots[index]["users"], index


def _top_stats(result: dict) -> tuple[str | None, float, float]:
    per_user = result.get("per_user") or []
    if not per_user:
        return None, 0.0, 0.0
    top_id = str(per_user[0]["user_id"])
    top_score = float(per_user[0]["score"])
    second_score = float(per_user[1]["score"]) if len(per_user) > 1 else 0.0
    gap = top_score - second_score
    return top_id, top_score, gap


def _print_text(result: dict, top_k: int, source: str) -> None:
    speaker = result.get("speaker_id")
    score = result.get("speaker_score")
    doa = result.get("doa", {})
    weights = result.get("weights", {})
    if speaker is None:
        print(f"speaker=None src={source}")
        return
    header = (
        f"speaker={speaker} score={float(score):.3f} src={source} "
        f"az={doa.get('azimuth_deg')} conf={float(doa.get('conf_doa', 0.0)):.3f} "
        f"w_doa={float(weights.get('doa', 0.0)):.2f}"
    )
    print(header)
    for item in result.get("per_user", [])[: max(1, int(top_k))]:
        print(
            f"  {item['user_id']}: score={item['score']:.3f} cnn={item['score_cnn']:.3f} "
            f"doa={item['score_doa']:.3f} d={item['delta_deg']:.1f}"
        )


def main() -> None:
    args = parse_args()
    cfg = FusionConfig(
        min_sigma_deg=float(args.min_sigma_deg),
        default_sigma_deg=float(args.default_sigma_deg),
        weak_doa_weight=float(args.weak_doa_weight),
        mid_doa_weight=float(args.mid_doa_weight),
        strong_doa_weight=float(args.strong_doa_weight),
        low_conf_th=float(args.low_conf_th),
        mid_conf_th=float(args.mid_conf_th),
        low_srp_th=float(args.low_srp_th),
        low_audio_th=float(args.low_audio_th),
    )
    static_users = _parse_static_users(args.users)
    cnn_snapshots = _read_cnn_snapshots(args.cnn_jsonl)
    live_cnn = LiveCNNSnapshots(args.cnn_jsonl_live) if args.cnn_jsonl_live else None
    cnn_index = 0

    furhat = None
    if args.attend_furhat:
        furhat = FurhatWS(
            ip=args.furhat_ip,
            port=args.furhat_port,
            auth_key=args.furhat_auth_key,
            speed=args.attend_speed,
            slack_pitch=args.slack_pitch,
            slack_yaw=args.slack_yaw,
            slack_timeout=args.slack_timeout,
        )

    current_speaker = None
    pending_speaker = None
    pending_hits = 0
    last_send_ts = 0.0
    min_send_period = 1.0 / max(float(args.send_hz), 1e-3)

    input_handle = _open_lines(args.doa_jsonl)
    close_after = input_handle is not None and input_handle is not sys.stdin
    try:
        for line in input_handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                doa_obs = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(doa_obs, dict):
                continue

            source = "none"
            if static_users:
                users = static_users
                source = "static"
            elif live_cnn is not None:
                users = live_cnn.users_for_time(doa_obs.get("t"), float(args.max_cnn_staleness_sec))
                source = "live-cnn"
            else:
                users, cnn_index = _select_users_for_time(
                    t_value=doa_obs.get("t"),
                    static_users=static_users,
                    snapshots=cnn_snapshots,
                    index=cnn_index,
                )
                source = "cnn-jsonl"

            if not users:
                continue

            result = score_users_for_frame(doa_obs=doa_obs, users=users, cfg=cfg)
            if args.text:
                _print_text(result, args.top_k, source)
            else:
                print(json.dumps(result, separators=(",", ":")))

            if furhat is None:
                continue

            top_id, top_score, top_gap = _top_stats(result)
            if top_id is None:
                pending_speaker = None
                pending_hits = 0
                continue
            if top_score < float(args.min_speaker_score) or top_gap < float(args.switch_margin):
                pending_speaker = None
                pending_hits = 0
                continue

            now = time.time()
            if current_speaker is None:
                current_speaker = top_id
                pending_speaker = None
                pending_hits = 0
                furhat.attend_user(current_speaker)
                last_send_ts = now
                continue

            if top_id == current_speaker:
                pending_speaker = None
                pending_hits = 0
                if now - last_send_ts >= min_send_period:
                    furhat.attend_user(current_speaker)
                    last_send_ts = now
                continue

            if pending_speaker == top_id:
                pending_hits += 1
            else:
                pending_speaker = top_id
                pending_hits = 1

            if pending_hits >= max(1, int(args.switch_hits)):
                current_speaker = top_id
                pending_speaker = None
                pending_hits = 0
                furhat.attend_user(current_speaker)
                last_send_ts = now
    finally:
        if close_after:
            input_handle.close()


if __name__ == "__main__":
    main()
