import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import math
from pathlib import Path

try:
    import websocket
except ImportError:
    websocket = None

try:
    from .fusion import FusionConfig, UserEvidence, score_users_for_frame
except ImportError:
    from fusion import FusionConfig, UserEvidence, score_users_for_frame


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
            if t_value is None:
                continue
            # Persist empty user lists too so "out of camera" can be detected in realtime.
            self.latest = {"t": float(t_value), "users": users}

    def users_for_time(
        self, t_value: float | None, max_staleness_sec: float
    ) -> list[UserEvidence]:
        self.poll()
        if self.latest is None:
            return []
        if t_value is not None:
            age = abs(float(t_value) - float(self.latest["t"]))
            if age > max_staleness_sec:
                return []
        return self.latest["users"]


def _top_stats(result: dict) -> tuple[str | None, float, float]:
    # Respect fusion policy modes which intentionally output speaker_id=None.
    if result.get("speaker_id") is None:
        return None, 0.0, 0.0
    per_user = result.get("per_user") or []
    if not per_user:
        return None, 0.0, 0.0
    top_id = str(per_user[0]["user_id"])
    top_score = float(per_user[0]["score"])
    second_score = float(per_user[1]["score"]) if len(per_user) > 1 else 0.0
    gap = top_score - second_score
    return top_id, top_score, gap


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

    def attend_location(self, x: float, y: float, z: float) -> None:
        self._send(
            {
                "type": "request.attend.location",
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "speed": self.speed,
                "slack_pitch": self.slack_pitch,
                "slack_yaw": self.slack_yaw,
                "slack_timeout": self.slack_timeout,
            }
        )


def _azimuth_to_xyz(
    azimuth_deg: float,
    distance_m: float,
    y_m: float,
    az_gain: float,
    max_az_deg: float,
    flip_x: bool,
) -> tuple[float, float, float]:
    lim = abs(float(max_az_deg))
    az_cmd = max(-lim, min(lim, float(azimuth_deg) * float(az_gain)))
    ang = math.radians(az_cmd)
    x = float(distance_m) * math.sin(ang)
    if flip_x:
        x = -x
    y = float(y_m)
    z = float(distance_m) * math.cos(ang)
    return x, y, z


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run true realtime CNN+DOA fusion in one command."
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--cnn-jsonl-live", default="/tmp/cnn_live.jsonl")
    parser.add_argument("--doa-jsonl-live", default="/tmp/doa_live.jsonl")
    parser.add_argument("--truncate-cnn-jsonl", action="store_true", default=True)
    parser.add_argument(
        "--no-truncate-cnn-jsonl", action="store_false", dest="truncate_cnn_jsonl"
    )
    parser.add_argument("--truncate-doa-jsonl", action="store_true", default=True)
    parser.add_argument(
        "--no-truncate-doa-jsonl", action="store_false", dest="truncate_doa_jsonl"
    )

    parser.add_argument("--cnn-source", choices=["furhat", "opencv", "file", "stream"], default="furhat")
    parser.add_argument("--furhat-ip", default="192.168.1.108")
    parser.add_argument("--furhat-auth", default=None)
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "data" / "models" / "cnn_vvad"),
    )
    parser.add_argument(
        "--step6-extra",
        action="append",
        default=["--show --window-width 960 --window-height 540"],
    )

    parser.add_argument("--audio-device", type=int, default=1)
    parser.add_argument("--audio-channels", type=int, default=6)
    parser.add_argument("--mic-channels", default="1,2,3,4")
    parser.add_argument("--doa-extra", action="append", default=[])

    parser.add_argument("--max-cnn-staleness-sec", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json-output", action="store_true")

    parser.add_argument("--low-conf-th", type=float, default=0.03)
    parser.add_argument("--mid-conf-th", type=float, default=0.07)
    parser.add_argument("--low-srp-th", type=float, default=0.08)
    parser.add_argument("--low-audio-th", type=float, default=0.25)
    parser.add_argument("--weak-doa-weight", type=float, default=0.10)
    parser.add_argument("--mid-doa-weight", type=float, default=0.25)
    parser.add_argument("--strong-doa-weight", type=float, default=0.40)
    parser.add_argument("--default-sigma-deg", type=float, default=25.0)
    parser.add_argument("--min-sigma-deg", type=float, default=8.0)
    parser.add_argument("--max-sigma-deg", type=float, default=45.0)
    parser.add_argument("--cnn-ambiguous-margin", type=float, default=0.12)
    parser.add_argument("--ambiguous-doa-boost", type=float, default=0.12)
    parser.add_argument("--cnn-dominant-prob-th", type=float, default=0.80)
    parser.add_argument("--cnn-dominant-margin", type=float, default=0.20)
    parser.add_argument("--dominance-doa-suppression", type=float, default=0.70)
    parser.add_argument(
        "--doa-disagreement-penalty",
        type=float,
        default=0.85,
        help="Penalty strength when DOA strongly disagrees with a visually dominant CNN winner (0..1).",
    )
    parser.add_argument(
        "--min-conf-doa-srp-for-fusion",
        type=float,
        default=0.10,
        help="Minimum SRP confidence needed before DOA can influence fusion in-frame.",
    )
    parser.add_argument(
        "--min-cnn-speech-th",
        type=float,
        default=0.30,
        help="If users are in-frame and BOTH audio speech is inactive and max CNN is below this, output speaker=None.",
    )
    parser.add_argument("--allow-single-user-doa", action="store_true")
    parser.add_argument("--single-user-doa-weight-cap", type=float, default=0.20)
    parser.add_argument("--single-user-min-conf", type=float, default=0.03)
    parser.add_argument("--single-user-max-delta-deg", type=float, default=35.0)

    parser.add_argument("--attend-furhat", action="store_true", default=True)
    parser.add_argument("--no-attend-furhat", action="store_false", dest="attend_furhat")
    parser.add_argument("--attend-furhat-ip", default=None)
    parser.add_argument("--attend-furhat-port", type=int, default=9000)
    parser.add_argument("--attend-furhat-auth-key", default="")
    parser.add_argument("--attend-speed", default="medium")
    parser.add_argument("--slack-pitch", type=float, default=15.0)
    parser.add_argument("--slack-yaw", type=float, default=10.0)
    parser.add_argument("--slack-timeout", type=int, default=3000)
    parser.add_argument("--min-speaker-score", type=float, default=0.30)
    parser.add_argument("--switch-hits", type=int, default=3)
    parser.add_argument("--switch-margin", type=float, default=0.03)
    parser.add_argument("--send-hz", type=float, default=4.0)
    parser.add_argument(
        "--doa-attend-location-fallback",
        action="store_true",
        default=True,
        help="When no visual users are available but DOA is active, send Furhat attend.location from DOA azimuth.",
    )
    parser.add_argument(
        "--no-doa-attend-location-fallback",
        action="store_false",
        dest="doa_attend_location_fallback",
    )
    parser.add_argument("--doa-target-distance-m", type=float, default=1.2)
    parser.add_argument("--doa-target-y-m", type=float, default=0.0)
    parser.add_argument("--doa-az-gain", type=float, default=1.0)
    parser.add_argument("--doa-max-az-deg", type=float, default=120.0)
    parser.add_argument(
        "--doa-flip-x",
        action="store_true",
        default=True,
        help="Mirror left/right when mapping DOA azimuth to Furhat x for attend.location.",
    )
    parser.add_argument(
        "--no-doa-flip-x",
        action="store_false",
        dest="doa_flip_x",
    )

    parser.add_argument("--show-child-logs", action="store_true", default=True)
    parser.add_argument("--no-show-child-logs", action="store_false", dest="show_child_logs")
    return parser.parse_args()


def _build_step6_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python_exe,
        str(PROJECT_ROOT / "scripts" / "step6_realtime_infer.py"),
        "--source",
        args.cnn_source,
        "--model-dir",
        args.model_dir,
        "--emit-cnn-jsonl",
        args.cnn_jsonl_live,
        "--doa-jsonl-live",
        args.doa_jsonl_live,
    ]
    if args.cnn_source == "furhat":
        cmd.extend(["--furhat-ip", args.furhat_ip])
        if args.furhat_auth:
            cmd.extend(["--furhat-auth", args.furhat_auth])
    for extra in args.step6_extra:
        cmd.extend(shlex.split(extra))
    return cmd


def _build_doa_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python_exe,
        str(PROJECT_ROOT / "audio_doa" / "doa_core.py"),
        "--device",
        str(args.audio_device),
        "--channels",
        str(args.audio_channels),
        "--mic-channels",
        str(args.mic_channels),
        "--no-emit-idle",
    ]
    for extra in args.doa_extra:
        cmd.extend(shlex.split(extra))
    return cmd


def _print_text(result: dict, top_k: int) -> None:
    speaker = result.get("speaker_id")
    score = result.get("speaker_score")
    mode = result.get("mode", "unknown")
    speech_active = bool(result.get("speech_active") or False)
    doa = result.get("doa", {})
    weights = result.get("weights", {})
    conf_value = float(doa.get("conf_doa_used", doa.get("conf_doa", 0.0)))
    if speaker is None:
        print(
            f"speaker=None mode={mode} speech={int(speech_active)} az={doa.get('azimuth_deg')} "
            f"conf={conf_value:.3f} w_doa={float(weights.get('doa', 0.0)):.2f}"
        )
        return
    print(
        f"speaker={speaker} score={float(score):.3f} mode={mode} speech={int(speech_active)} az={doa.get('azimuth_deg')} "
        f"conf={conf_value:.3f} w_doa={float(weights.get('doa', 0.0)):.2f}"
    )
    for item in result.get("per_user", [])[: max(1, int(top_k))]:
        print(
            f"  {item['user_id']}: score={item['score']:.3f} "
            f"cnn={item['score_cnn']:.3f} doa={item['score_doa']:.3f} d={item['delta_deg']:.1f}"
        )


def _terminate_process(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)
    print(f"[run-live-fusion] stopped {name}")


def main() -> None:
    args = parse_args()
    if args.attend_furhat and args.cnn_source != "furhat":
        raise ValueError(
            "--attend-furhat requires --cnn-source furhat so speaker_id maps to Furhat user IDs."
        )
    cfg = FusionConfig(
        min_sigma_deg=float(args.min_sigma_deg),
        default_sigma_deg=float(args.default_sigma_deg),
        max_sigma_deg=float(args.max_sigma_deg),
        weak_doa_weight=float(args.weak_doa_weight),
        mid_doa_weight=float(args.mid_doa_weight),
        strong_doa_weight=float(args.strong_doa_weight),
        low_conf_th=float(args.low_conf_th),
        mid_conf_th=float(args.mid_conf_th),
        low_srp_th=float(args.low_srp_th),
        low_audio_th=float(args.low_audio_th),
        cnn_ambiguous_margin=float(args.cnn_ambiguous_margin),
        ambiguous_doa_boost=float(args.ambiguous_doa_boost),
        cnn_dominant_prob_th=float(args.cnn_dominant_prob_th),
        cnn_dominant_margin=float(args.cnn_dominant_margin),
        dominance_doa_suppression=float(args.dominance_doa_suppression),
        min_cnn_speech_th=float(args.min_cnn_speech_th),
        allow_single_user_doa=bool(args.allow_single_user_doa),
        single_user_doa_weight_cap=float(args.single_user_doa_weight_cap),
        single_user_min_conf=float(args.single_user_min_conf),
        single_user_max_delta_deg=float(args.single_user_max_delta_deg),
        doa_disagreement_penalty=float(args.doa_disagreement_penalty),
        min_conf_doa_srp_for_fusion=float(args.min_conf_doa_srp_for_fusion),
    )

    cnn_path = Path(args.cnn_jsonl_live)
    doa_path = Path(args.doa_jsonl_live)
    cnn_path.parent.mkdir(parents=True, exist_ok=True)
    doa_path.parent.mkdir(parents=True, exist_ok=True)
    if args.truncate_cnn_jsonl:
        cnn_path.write_text("")
    if args.truncate_doa_jsonl:
        doa_path.write_text("")

    step6_cmd = _build_step6_cmd(args)
    doa_cmd = _build_doa_cmd(args)

    print("[run-live-fusion] step6 cmd:", " ".join(step6_cmd))
    print("[run-live-fusion] doa cmd:", " ".join(doa_cmd))

    step6_stdout = None if args.show_child_logs else subprocess.DEVNULL
    step6_stderr = None if args.show_child_logs else subprocess.STDOUT
    step6_proc = subprocess.Popen(
        step6_cmd,
        cwd=str(PROJECT_ROOT),
        stdout=step6_stdout,
        stderr=step6_stderr,
        text=True,
        bufsize=1,
    )
    doa_proc = subprocess.Popen(
        doa_cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=None if args.show_child_logs else subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    live_cnn = LiveCNNSnapshots(str(cnn_path))
    furhat = None
    if args.attend_furhat:
        furhat = FurhatWS(
            ip=args.attend_furhat_ip or args.furhat_ip,
            port=args.attend_furhat_port,
            auth_key=args.attend_furhat_auth_key,
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
    doa_live_handle = None

    try:
        doa_live_handle = open(doa_path, "a", encoding="utf-8", buffering=1)
        if doa_proc.stdout is None:
            raise RuntimeError("DOA process stdout is unavailable.")
        for line in doa_proc.stdout:
            if step6_proc.poll() is not None:
                raise RuntimeError(
                    f"step6 process exited early with code {step6_proc.returncode}."
                )

            raw = line.strip()
            if not raw:
                continue
            doa_live_handle.write(raw + "\n")
            doa_live_handle.flush()
            try:
                doa_obs = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(doa_obs, dict):
                continue

            users = live_cnn.users_for_time(
                doa_obs.get("t"), float(args.max_cnn_staleness_sec)
            )
            result = score_users_for_frame(doa_obs=doa_obs, users=users, cfg=cfg)
            if args.json_output:
                print(json.dumps(result, separators=(",", ":")))
            else:
                _print_text(result, args.top_k)

            if furhat is None:
                continue

            top_id, top_score, top_gap = _top_stats(result)
            mode = str(result.get("mode") or "")
            if top_id is None:
                pending_speaker = None
                pending_hits = 0
                if (
                    bool(args.doa_attend_location_fallback)
                    and mode == "doa_only_no_users"
                ):
                    doa_info = result.get("doa", {})
                    azimuth_deg = (
                        doa_info.get("azimuth_deg")
                        if isinstance(doa_info, dict)
                        else None
                    )
                    if azimuth_deg is not None:
                        now = time.time()
                        if now - last_send_ts >= min_send_period:
                            x, y, z = _azimuth_to_xyz(
                                azimuth_deg=float(azimuth_deg),
                                distance_m=float(args.doa_target_distance_m),
                                y_m=float(args.doa_target_y_m),
                                az_gain=float(args.doa_az_gain),
                                max_az_deg=float(args.doa_max_az_deg),
                                flip_x=bool(args.doa_flip_x),
                            )
                            furhat.attend_location(x, y, z)
                            last_send_ts = now
                continue
            if top_score < float(args.min_speaker_score) or top_gap < float(
                args.switch_margin
            ):
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
    except KeyboardInterrupt:
        print("\n[run-live-fusion] interrupted by user.")
    finally:
        if doa_live_handle is not None:
            try:
                doa_live_handle.close()
            except Exception:
                pass
        _terminate_process(doa_proc, "doa_core")
        _terminate_process(step6_proc, "step6_realtime_infer")


if __name__ == "__main__":
    main()
