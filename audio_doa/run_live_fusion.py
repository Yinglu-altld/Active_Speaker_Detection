import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

try:
    import websocket
except ImportError:
    websocket = None

try:
    from .fusion import FusionConfig, score_users_for_frame
    from .fusion_stub import LiveCNNSnapshots, _top_stats
except ImportError:
    from fusion import FusionConfig, score_users_for_frame
    from fusion_stub import LiveCNNSnapshots, _top_stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    parser.add_argument("--furhat-ip", default="192.168.1.109")
    parser.add_argument("--furhat-auth", default=None)
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "data" / "models" / "cnn_vvad"),
    )
    parser.add_argument("--step6-extra", action="append", default=[])

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

    parser.add_argument("--attend-furhat", action="store_true")
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
    doa = result.get("doa", {})
    weights = result.get("weights", {})
    if speaker is None:
        print("speaker=None")
        return
    print(
        f"speaker={speaker} score={float(score):.3f} az={doa.get('azimuth_deg')} "
        f"conf={float(doa.get('conf_doa', 0.0)):.3f} w_doa={float(weights.get('doa', 0.0)):.2f}"
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
            if not users:
                continue

            result = score_users_for_frame(doa_obs=doa_obs, users=users, cfg=cfg)
            if args.json_output:
                print(json.dumps(result, separators=(",", ":")))
            else:
                _print_text(result, args.top_k)

            if furhat is None:
                continue

            top_id, top_score, top_gap = _top_stats(result)
            if top_id is None:
                pending_speaker = None
                pending_hits = 0
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
