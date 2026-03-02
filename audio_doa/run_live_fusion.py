import argparse
import json
import math
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import websocket
except ImportError:
    websocket = None

try:
    from .fusion import FusionConfig, score_users_cnn_only, score_users_for_frame
    from .fusion_stub import LiveCNNSnapshots, _top_stats
except ImportError:
    from fusion import FusionConfig, score_users_cnn_only, score_users_for_frame
    from fusion_stub import LiveCNNSnapshots, _top_stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# DOA azimuth_deg is expected to already be in robot bearing convention:
# 0 = forward (+Z), + = right (+X), - = left (-X).
DOA_DEADBAND_DEG = 4.0


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


class DOAStreamPump:
    def __init__(self, proc: subprocess.Popen, output_path: Path):
        self.proc = proc
        self.output_path = output_path
        self.handle = None
        self.latest = None
        self.latest_seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        if self.proc.stdout is None:
            raise RuntimeError("DOA process stdout is unavailable.")
        self.handle = open(self.output_path, "a", encoding="utf-8", buffering=1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            if self._stop.is_set():
                break
            raw = line.strip()
            if not raw:
                continue
            if self.handle is not None:
                self.handle.write(raw + "\n")
                self.handle.flush()
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            with self._lock:
                self.latest = item
                self.latest_seq += 1

    def latest_obs(self):
        with self._lock:
            if self.latest is None:
                return None
            return dict(self.latest)

    def latest_snapshot(self) -> tuple[dict | None, int]:
        with self._lock:
            if self.latest is None:
                return None, self.latest_seq
            return dict(self.latest), self.latest_seq

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.handle is not None:
            try:
                self.handle.close()
            except Exception:
                pass


@dataclass
class AttendState:
    current_speaker: str | None = None
    pending_speaker: str | None = None
    pending_hits: int = 0
    last_send_ts: float = 0.0
    last_location_xyz: tuple[float, float, float] | None = None
    last_doa_cmd_az_deg: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run true realtime CNN+DOA fusion in one command."
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--cnn-jsonl-live", default="/tmp/cnn_live.jsonl")
    parser.add_argument("--doa-jsonl-live", default="/tmp/doa_live.jsonl")
    parser.add_argument("--gt-jsonl-live", default="/tmp/gt_live.jsonl")
    parser.add_argument("--truncate-cnn-jsonl", action="store_true", default=True)
    parser.add_argument(
        "--no-truncate-cnn-jsonl", action="store_false", dest="truncate_cnn_jsonl"
    )
    parser.add_argument("--truncate-doa-jsonl", action="store_true", default=True)
    parser.add_argument(
        "--no-truncate-doa-jsonl", action="store_false", dest="truncate_doa_jsonl"
    )
    parser.add_argument("--truncate-gt-jsonl", action="store_true", default=True)
    parser.add_argument(
        "--no-truncate-gt-jsonl", action="store_false", dest="truncate_gt_jsonl"
    )

    parser.add_argument(
        "--cnn-source",
        choices=["furhat", "opencv", "file", "stream"],
        default="furhat",
    )
    parser.add_argument("--furhat-ip", default="192.168.1.108")
    parser.add_argument("--furhat-auth", default=None)
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "data" / "models" / "cnn_vvad"),
    )
    parser.add_argument("--step6-extra", action="append", default=[])

    parser.add_argument("--doa-extra", action="append", default=[])
    parser.add_argument(
        "--use-button-gt",
        action="store_true",
        default=False,
        help="Enable UDP button ground-truth bridge and plot stream.",
    )
    parser.add_argument("--gt-listen-host", default="0.0.0.0")
    parser.add_argument("--gt-listen-port", type=int, default=5005)
    parser.add_argument("--gt-user0-id", default="user-0")
    parser.add_argument("--gt-user1-id", default="user-1")
    parser.add_argument("--gt-button0-key", default="u0")
    parser.add_argument("--gt-button1-key", default="u1")
    parser.add_argument("--gt-poll-hz", type=float, default=20.0)
    parser.add_argument("--gt-stale-sec", type=float, default=0.7)

    parser.add_argument("--max-cnn-staleness-sec", type=float, default=0.8)
    parser.add_argument(
        "--idle-after-no-speech-sec",
        type=float,
        default=5.0,
        help="After this much continuous NO_SPEECH, send Furhat to center idle pose.",
    )
    parser.add_argument("--tick-hz", type=float, default=12.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json-output", action="store_true")

    parser.add_argument("--default-sigma-deg", type=float, default=25.0)
    parser.add_argument("--min-sigma-deg", type=float, default=8.0)

    parser.add_argument(
        "--attend-furhat",
        action="store_true",
        default=True,
        help="Enable Furhat attend control (default: enabled).",
    )
    parser.add_argument(
        "--no-attend-furhat",
        action="store_false",
        dest="attend_furhat",
        help="Disable Furhat attend control.",
    )
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
    parser.add_argument("--doa-target-distance-m", type=float, default=1.2)
    parser.add_argument("--doa-target-y-m", type=float, default=0.0)

    parser.add_argument("--show-child-logs", action="store_true")
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
        "--min-speaker-score",
        str(float(args.min_speaker_score)),
    ]
    if args.use_button_gt:
        cmd.extend(["--gt-jsonl-live", args.gt_jsonl_live])
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
        "--no-emit-idle",
    ]
    for extra in args.doa_extra:
        cmd.extend(shlex.split(extra))
    return cmd


def _build_gt_bridge_cmd(args: argparse.Namespace) -> list[str]:
    return [
        args.python_exe,
        str(PROJECT_ROOT / "audio_doa" / "button_gt_bridge.py"),
        "--listen-host",
        str(args.gt_listen_host),
        "--listen-port",
        str(int(args.gt_listen_port)),
        "--output-jsonl",
        str(args.gt_jsonl_live),
        "--poll-hz",
        str(float(args.gt_poll_hz)),
        "--stale-sec",
        str(float(args.gt_stale_sec)),
        "--user0-id",
        str(args.gt_user0_id),
        "--user1-id",
        str(args.gt_user1_id),
        "--button0-key",
        str(args.gt_button0_key),
        "--button1-key",
        str(args.gt_button1_key),
    ]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _wrap_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _doa_reliability(doa_obs: dict) -> float:
    conf_srp = _clip01(float(doa_obs.get("conf_doa_srp") or 0.0))
    audio_conf = _clip01(float(doa_obs.get("audio_conf") or 0.0))
    return 0.6 * conf_srp + 0.4 * audio_conf


def _obs_speech_active(doa_obs: dict | None) -> bool:
    if doa_obs is None:
        return False
    return bool(doa_obs.get("speech_active") or doa_obs.get("speech_detected"))


def _azimuth_to_xyz(azimuth_deg: float, distance_m: float, y_m: float) -> tuple[float, float, float]:
    az = _wrap_deg(float(azimuth_deg))
    ang = math.radians(az)
    x = float(distance_m * math.sin(ang))
    z = float(distance_m * math.cos(ang))
    y = float(y_m)
    return x, y, z


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


def _reset_attend_state(state: AttendState) -> None:
    state.current_speaker = None
    state.pending_speaker = None
    state.pending_hits = 0


def _handle_user_attend(
    result: dict,
    mode: str,
    furhat: FurhatWS,
    state: AttendState,
    now_ts: float,
    min_send_period: float,
    min_speaker_score: float,
    switch_margin: float,
    switch_hits: int,
) -> None:
    top_id, top_score, top_gap = _top_stats(result)
    if top_id is None:
        _reset_attend_state(state)
        return
    use_min_score = mode in ("SPEECH_AV", "SPEECH_CNN_ONLY")
    if use_min_score and top_score < float(min_speaker_score):
        _reset_attend_state(state)
        return
    if top_gap < float(switch_margin):
        state.pending_speaker = None
        state.pending_hits = 0
        return

    if state.current_speaker is None:
        state.current_speaker = top_id
        state.pending_speaker = None
        state.pending_hits = 0
        furhat.attend_user(state.current_speaker)
        state.last_location_xyz = None
        state.last_doa_cmd_az_deg = None
        state.last_send_ts = now_ts
        return

    if top_id == state.current_speaker:
        state.pending_speaker = None
        state.pending_hits = 0
        if now_ts - state.last_send_ts >= min_send_period:
            furhat.attend_user(state.current_speaker)
            state.last_location_xyz = None
            state.last_doa_cmd_az_deg = None
            state.last_send_ts = now_ts
        return

    if state.pending_speaker == top_id:
        state.pending_hits += 1
    else:
        state.pending_speaker = top_id
        state.pending_hits = 1

    if state.pending_hits >= max(1, int(switch_hits)):
        state.current_speaker = top_id
        state.pending_speaker = None
        state.pending_hits = 0
        furhat.attend_user(state.current_speaker)
        state.last_location_xyz = None
        state.last_doa_cmd_az_deg = None
        state.last_send_ts = now_ts


def _handle_doa_attend(
    doa_obs: dict,
    furhat: FurhatWS,
    state: AttendState,
    now_ts: float,
    min_send_period: float,
    distance_m: float,
    y_m: float,
) -> None:
    azimuth = doa_obs.get("azimuth_deg")
    if azimuth is None:
        return
    cmd_az = _wrap_deg(float(azimuth))
    if state.last_doa_cmd_az_deg is not None:
        delta_cmd = abs(_wrap_deg(cmd_az - float(state.last_doa_cmd_az_deg)))
        if delta_cmd < float(DOA_DEADBAND_DEG):
            return
    if now_ts - state.last_send_ts < min_send_period:
        return
    _reset_attend_state(state)
    x, y, z = _azimuth_to_xyz(cmd_az, float(distance_m), float(y_m))
    furhat.attend_location(x, y, z)
    state.last_location_xyz = (float(x), float(y), float(z))
    state.last_doa_cmd_az_deg = float(cmd_az)
    state.last_send_ts = now_ts


def _print_text(payload: dict, top_k: int) -> None:
    mode = str(payload.get("mode", "UNKNOWN"))
    if mode == "NO_SPEECH":
        print("state=NO_SPEECH no speech detected")
        return
    if mode == "SPEECH_DOA_ONLY":
        doa = payload.get("doa", {})
        print(
            f"state=SPEECH_DOA_ONLY az={doa.get('azimuth_deg')} "
            f"reliability={float(doa.get('reliability', 0.0)):.3f}"
        )
        return

    speaker = payload.get("speaker_id")
    score = payload.get("speaker_score")
    doa = payload.get("doa", {})
    weights = payload.get("weights", {})
    if speaker is None:
        print(f"state={mode} speaker=None")
        return
    print(
        f"state={mode} speaker={speaker} score={float(score):.3f} "
        f"az={doa.get('azimuth_deg')} rel={float(doa.get('reliability', 0.0)):.3f} "
        f"w_doa={float(weights.get('doa', 0.0)):.2f}"
    )
    for item in payload.get("per_user", [])[: max(1, int(top_k))]:
        print(
            f"  {item['user_id']}: score={item['score']:.3f} "
            f"cnn={item['score_cnn']:.3f} doa={item['score_doa']:.3f} d={item['delta_deg']:.1f}"
        )


def _no_speech_payload(now_ts: float, doa_obs: dict | None) -> dict:
    return {
        "mode": "NO_SPEECH",
        "t": now_ts,
        "speaker_id": None,
        "speaker_score": None,
        "weights": {"cnn": 0.0, "doa": 0.0},
        "doa": {
            "azimuth_deg": None if doa_obs is None else doa_obs.get("azimuth_deg"),
            "conf_doa": 0.0 if doa_obs is None else float(doa_obs.get("conf_doa") or 0.0),
            "conf_doa_srp": 0.0 if doa_obs is None else float(doa_obs.get("conf_doa_srp") or 0.0),
            "audio_conf": 0.0 if doa_obs is None else float(doa_obs.get("audio_conf") or 0.0),
            "sigma_deg": None if doa_obs is None else doa_obs.get("sigma_deg"),
            "reliability": 0.0 if doa_obs is None else float(_doa_reliability(doa_obs)),
        },
        "per_user": [],
    }


def main() -> None:
    args = parse_args()
    cfg = FusionConfig(
        min_sigma_deg=float(args.min_sigma_deg),
        default_sigma_deg=float(args.default_sigma_deg),
        fixed_doa_weight=0.35,
    )

    cnn_path = Path(args.cnn_jsonl_live)
    doa_path = Path(args.doa_jsonl_live)
    gt_path = Path(args.gt_jsonl_live)
    cnn_path.parent.mkdir(parents=True, exist_ok=True)
    doa_path.parent.mkdir(parents=True, exist_ok=True)
    if args.use_button_gt:
        gt_path.parent.mkdir(parents=True, exist_ok=True)
    if args.truncate_cnn_jsonl:
        cnn_path.write_text("")
    if args.truncate_doa_jsonl:
        doa_path.write_text("")
    if args.use_button_gt and args.truncate_gt_jsonl:
        gt_path.write_text("")

    step6_cmd = _build_step6_cmd(args)
    doa_cmd = _build_doa_cmd(args)
    gt_cmd = _build_gt_bridge_cmd(args) if args.use_button_gt else None
    print("[run-live-fusion] step6 cmd:", " ".join(step6_cmd))
    print("[run-live-fusion] doa cmd:", " ".join(doa_cmd))
    if gt_cmd is not None:
        print("[run-live-fusion] gt cmd:", " ".join(gt_cmd))

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
    gt_proc = None
    if gt_cmd is not None:
        gt_proc = subprocess.Popen(
            gt_cmd,
            cwd=str(PROJECT_ROOT),
            stdout=None if args.show_child_logs else subprocess.DEVNULL,
            stderr=None if args.show_child_logs else subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    live_cnn = LiveCNNSnapshots(str(cnn_path))
    doa_pump = DOAStreamPump(doa_proc, doa_path)
    doa_pump.start()

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

    attend_state = AttendState()
    min_send_period = 1.0 / max(float(args.send_hz), 1e-3)
    tick_period = 1.0 / max(float(args.tick_hz), 1e-3)
    next_tick = time.time()
    last_quiet_log_ts = 0.0
    no_speech_since_ts = None
    idle_sent = False
    last_doa_seq_used = 0

    try:
        while True:
            if step6_proc.poll() is not None:
                raise RuntimeError(
                    f"step6 process exited early with code {step6_proc.returncode}."
                )
            if doa_proc.poll() is not None:
                raise RuntimeError(
                    f"doa process exited early with code {doa_proc.returncode}."
                )
            if gt_proc is not None and gt_proc.poll() is not None:
                raise RuntimeError(
                    f"button_gt_bridge process exited early with code {gt_proc.returncode}."
                )

            now_ts = time.time()
            doa_latest, doa_seq = doa_pump.latest_snapshot()
            if doa_latest is not None and doa_seq > last_doa_seq_used:
                doa_obs = doa_latest
                last_doa_seq_used = doa_seq
            else:
                doa_obs = None

            doa_available = doa_obs is not None
            speech_active = doa_available and _obs_speech_active(doa_obs)
            users = live_cnn.users_for_time(now_ts, float(args.max_cnn_staleness_sec))

            payload = None
            mode = None
            if not speech_active:
                mode = "NO_SPEECH"
                payload = _no_speech_payload(now_ts=now_ts, doa_obs=doa_obs)
            elif users:
                if doa_available and doa_obs.get("azimuth_deg") is not None:
                    mode = "SPEECH_AV"
                    payload = score_users_for_frame(doa_obs=doa_obs, users=users, cfg=cfg)
                else:
                    mode = "SPEECH_CNN_ONLY"
                    t_value = now_ts if doa_obs is None else doa_obs.get("t")
                    payload = score_users_cnn_only(users=users, t_value=t_value)
            elif doa_available and doa_obs.get("azimuth_deg") is not None:
                mode = "SPEECH_DOA_ONLY"
                payload = {
                    "mode": mode,
                    "t": doa_obs.get("t"),
                    "speaker_id": None,
                    "speaker_score": None,
                    "weights": {"cnn": 0.0, "doa": 1.0},
                    "doa": {
                        "azimuth_deg": doa_obs.get("azimuth_deg"),
                        "conf_doa": float(doa_obs.get("conf_doa") or 0.0),
                        "conf_doa_srp": float(doa_obs.get("conf_doa_srp") or 0.0),
                        "audio_conf": float(doa_obs.get("audio_conf") or 0.0),
                        "sigma_deg": doa_obs.get("sigma_deg"),
                        "reliability": float(_doa_reliability(doa_obs)),
                    },
                    "per_user": [],
                }
            else:
                mode = "NO_SPEECH"
                payload = _no_speech_payload(now_ts=now_ts, doa_obs=doa_obs)

            if "mode" not in payload:
                payload = dict(payload)
                payload["mode"] = mode

            if mode == "NO_SPEECH":
                if no_speech_since_ts is None:
                    no_speech_since_ts = now_ts
                    idle_sent = False
            else:
                no_speech_since_ts = None
                idle_sent = False

            if args.json_output:
                print(json.dumps(payload, separators=(",", ":")))
            else:
                quiet_mode = mode == "NO_SPEECH"
                if (not quiet_mode) or (now_ts - last_quiet_log_ts >= 1.0):
                    _print_text(payload, args.top_k)
                    if quiet_mode:
                        last_quiet_log_ts = now_ts

            if furhat is not None:
                if mode in ("SPEECH_AV", "SPEECH_CNN_ONLY"):
                    _handle_user_attend(
                        result=payload,
                        mode=mode,
                        furhat=furhat,
                        state=attend_state,
                        now_ts=now_ts,
                        min_send_period=min_send_period,
                        min_speaker_score=float(args.min_speaker_score),
                        switch_margin=float(args.switch_margin),
                        switch_hits=int(args.switch_hits),
                    )
                elif mode == "SPEECH_DOA_ONLY":
                    _handle_doa_attend(
                        doa_obs=payload["doa"],
                        furhat=furhat,
                        state=attend_state,
                        now_ts=now_ts,
                        min_send_period=min_send_period,
                        distance_m=float(args.doa_target_distance_m),
                        y_m=float(args.doa_target_y_m),
                    )
                else:
                    _reset_attend_state(attend_state)
                    idle_after = max(0.0, float(args.idle_after_no_speech_sec))
                    no_speech_elapsed = (
                        0.0
                        if no_speech_since_ts is None
                        else max(0.0, now_ts - float(no_speech_since_ts))
                    )
                    # Keep holding the last DOA target during short no-speech gaps
                    # to avoid abrupt drift-to-center behavior on natural pauses.
                    if (
                        attend_state.last_location_xyz is not None
                        and no_speech_elapsed < idle_after
                        and (now_ts - attend_state.last_send_ts) >= min_send_period
                    ):
                        hold_x, hold_y, hold_z = attend_state.last_location_xyz
                        furhat.attend_location(hold_x, hold_y, hold_z)
                        attend_state.last_send_ts = now_ts
                    if (
                        no_speech_since_ts is not None
                        and not idle_sent
                        and no_speech_elapsed >= idle_after
                    ):
                        furhat.attend_location(
                            0.0,
                            float(args.doa_target_y_m),
                            float(args.doa_target_distance_m),
                        )
                        attend_state.last_location_xyz = None
                        attend_state.last_doa_cmd_az_deg = None
                        attend_state.last_send_ts = now_ts
                        idle_sent = True

            next_tick += tick_period
            sleep_sec = next_tick - time.time()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
            else:
                next_tick = time.time()
    except KeyboardInterrupt:
        print("\n[run-live-fusion] interrupted by user.")
    finally:
        doa_pump.close()
        _terminate_process(doa_proc, "doa_core")
        _terminate_process(gt_proc, "button_gt_bridge")
        _terminate_process(step6_proc, "step6_realtime_infer")


if __name__ == "__main__":
    main()
