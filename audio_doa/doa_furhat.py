import argparse
import json
import math
import queue
import time

import numpy as np
import sounddevice as sd
import websocket

try:
    from .doa_core import DOAConfig, DOAEstimator, SileroGate
    from .srp_phat import SRPPhatDOA
except ImportError:
    from doa_core import DOAConfig, DOAEstimator, SileroGate
    from srp_phat import SRPPhatDOA


MIC_XY = np.array([[0.028, 0.0], [0.0, 0.028], [-0.028, 0.0], [0.0, -0.028]], dtype=np.float64)


def wrap(a): return ((a + 180.0) % 360.0) - 180.0
def cdelta(a, b): return wrap(b - a)
def cblend(a, b, alpha): return wrap(a + alpha * cdelta(a, b))
def fold_front(az):
    az = wrap(az)
    if az > 90.0: return 180.0 - az
    if az < -90.0: return -180.0 - az
    return az


class FurhatWS:
    def __init__(self, ip, port, key="", speed="medium", slack_pitch=15.0, slack_yaw=5.0, slack_timeout=3000):
        self.url, self.key, self.speed, self.ws = f"ws://{ip}:{port}/v1/events", key, speed, None
        self.slack_pitch = float(slack_pitch)
        self.slack_yaw = float(slack_yaw)
        self.slack_timeout = int(slack_timeout)
        self.last_users = []
        self.users_stream_started = False

    def _connect(self):
        self.ws = websocket.create_connection(self.url, timeout=0.5)
        self.ws.settimeout(0.01)
        auth = {"type": "request.auth"}
        if self.key: auth["key"] = self.key
        self.ws.send(json.dumps(auth))

    def _send(self, payload):
        if self.ws is None: self._connect()
        try:
            self.ws.send(json.dumps(payload))
        except Exception:
            try:
                if self.ws is not None: self.ws.close()
            finally:
                self.ws = None
            self._connect()
            self.ws.send(json.dumps(payload))

    def _drain_messages(self, max_messages=20):
        if self.ws is None:
            return
        for _ in range(max_messages):
            try:
                msg = self.ws.recv()
            except Exception:
                break
            try:
                data = json.loads(msg)
            except Exception:
                continue
            if data.get("type") == "response.users.data":
                users = data.get("users")
                if isinstance(users, list):
                    self.last_users = users

    def start_users_stream(self):
        if self.users_stream_started:
            return
        self._send({"type": "request.users.start"})
        self.users_stream_started = True

    def request_users_once(self):
        self._send({"type": "request.users.once"})

    def get_users(self):
        self._drain_messages()
        return self.last_users

    def attend(self, x, y, z):
        payload = {
            "type": "request.attend.location",
            "x": x,
            "y": y,
            "z": z,
            "speed": self.speed,
            "slack_pitch": self.slack_pitch,
            "slack_yaw": self.slack_yaw,
            "slack_timeout": self.slack_timeout,
        }
        self._send(payload)

    def attend_closest_user(self):
        payload = {
            "type": "request.attend.user",
            "user_id": "closest",
            "speed": self.speed,
            "slack_pitch": self.slack_pitch,
            "slack_yaw": self.slack_yaw,
            "slack_timeout": self.slack_timeout,
        }
        self._send(payload)


def map_doa_to_az(doa, board_cw, offset, add_180, front_only, max_az):
    az1 = wrap(offset + ((-doa) if board_cw else doa) + (180.0 if add_180 else 0.0))
    az2 = wrap(az1 + 180.0)
    if front_only:
        az1 = fold_front(az1)
        az2 = fold_front(az2)
    lim = abs(max_az)
    az1 = max(-lim, min(lim, az1))
    az2 = max(-lim, min(lim, az2))
    return az1, az2


def user_bearing_deg(user):
    if not isinstance(user, dict):
        return None
    loc = user.get("location", {})
    if not isinstance(loc, dict):
        return None
    try:
        x = float(loc.get("x", 0.0))
        z = float(loc.get("z", 0.0))
    except Exception:
        return None
    if abs(x) < 1e-6 and abs(z) < 1e-6:
        return None
    return math.degrees(math.atan2(x, z))


def user_location_xyz(user):
    if not isinstance(user, dict):
        return None
    loc = user.get("location", {})
    if not isinstance(loc, dict):
        return None
    try:
        x = float(loc.get("x", 0.0))
        y = float(loc.get("y", 0.0))
        z = float(loc.get("z", 0.0))
    except Exception:
        return None
    return x, y, z


def main():
    p = argparse.ArgumentParser(description="ReSpeaker raw DOA + Silero VAD -> Furhat attend.location")
    p.add_argument("--furhat-ip", default="192.168.1.109")
    p.add_argument("--furhat-port", type=int, default=9000)
    p.add_argument("--furhat-auth-key", default="")
    p.add_argument("--fs", type=int, default=16000)
    p.add_argument("--channels", type=int, default=6)
    p.add_argument("--device", type=int, default=0, help="sounddevice input device index")
    p.add_argument("--mic-channels", default="1,2,3,4", help="0-based indices inside capture stream")
    p.add_argument("--vad-mic", type=int, default=0, help="index within --mic-channels used for VAD/energy (0..N-1)")
    p.add_argument("--frame-ms", type=int, default=80)
    p.add_argument("--srp-az-step-deg", type=float, default=2.0)
    p.add_argument("--srp-interp", type=int, default=4)
    p.add_argument("--srp-f-low-hz", type=float, default=300.0)
    p.add_argument("--srp-f-high-hz", type=float, default=3400.0)
    p.add_argument("--board-zero-offset", type=float, default=-45.0)
    p.add_argument("--board-cw", action="store_true", default=False)
    p.add_argument("--no-board-cw", action="store_false", dest="board_cw")
    p.add_argument("--add-180", action="store_true", help="add 180 deg to DOA mapping")
    p.add_argument("--attend-mode", choices=("location", "closest-user", "hybrid", "doa-user-match"), default="doa-user-match")
    p.add_argument("--smooth-alpha", type=float, default=0.18)
    p.add_argument("--lock-alpha", type=float, default=0.14, help="update rate for locked azimuth during one speech segment")
    p.add_argument("--max-jump-deg", type=float, default=90.0)
    p.add_argument("--speaker-switch-deg", type=float, default=40.0, help="force switch lock if candidate stays this far from current lock")
    p.add_argument("--speaker-switch-updates", type=int, default=2, help="consistent far-apart updates needed to force speaker switch")
    p.add_argument("--consistency-deg", type=float, default=10.0, help="max azimuth spread for consistent DOA updates")
    p.add_argument("--min-consistent-updates", type=int, default=3, help="required consistent DOA updates before lock update")
    p.add_argument("--doa-quality-threshold", type=float, default=0.20)
    p.add_argument("--vad-threshold", type=float, default=0.22)
    p.add_argument("--vad-smooth-alpha", type=float, default=0.80, help="EMA alpha for VAD prob (0 disables smoothing)")
    p.add_argument("--vad-update-threshold", type=float, default=0.30, help="minimum VAD prob required to update DOA")
    p.add_argument("--energy-threshold", type=float, default=150.0, help="fallback speech gate on mean abs mono int16")
    p.add_argument("--energy-update-threshold", type=float, default=250.0, help="fallback update gate when VAD is uncertain")
    p.add_argument("--noise-alpha", type=float, default=0.97, help="EMA factor for noise floor energy estimate (higher = slower)")
    p.add_argument("--snr-speech-ratio", type=float, default=1.8, help="speech gate: energy >= noise*ratio + add")
    p.add_argument("--snr-speech-add", type=float, default=35.0, help="speech gate: energy >= noise*ratio + add")
    p.add_argument("--snr-update-ratio", type=float, default=2.2, help="update gate (when VAD low): energy >= noise*ratio + add")
    p.add_argument("--snr-update-add", type=float, default=60.0, help="update gate (when VAD low): energy >= noise*ratio + add")
    p.add_argument("--speech-hold-ms", type=int, default=300, help="continue tracking this long after VAD drops")
    p.add_argument("--front-only", action="store_true", default=True)
    p.add_argument("--no-front-only", action="store_false", dest="front_only")
    p.add_argument("--use-mirror-branch", action="store_true", default=False, help="use mirrored azimuth candidate (legacy behavior)")
    p.add_argument("--max-az-deg", type=float, default=60.0)
    p.add_argument("--az-gain", type=float, default=0.70, help="scale commanded azimuth to reduce over-tilt at extremes")
    p.add_argument("--target-distance-m", type=float, default=1.2)
    p.add_argument("--target-y-m", type=float, default=0.0)
    p.add_argument("--flip-x", action="store_true", default=True, help="mirror left/right by negating x")
    p.add_argument("--no-flip-x", action="store_false", dest="flip_x")
    p.add_argument("--hybrid-doa-x-blend", type=float, default=0.20, help="blend DOA x intent into camera user x (0..1)")
    p.add_argument("--hybrid-side-threshold-m", type=float, default=0.08, help="minimum |x| to treat DOA/user side as left/right")
    p.add_argument("--hybrid-no-camera-policy", choices=("hold", "closest-user", "doa"), default="hold")
    p.add_argument("--hybrid-camera-hold-ms", type=int, default=1500, help="how long to hold last camera target if users data drops")
    p.add_argument("--doa-user-max-diff-deg", type=float, default=30.0, help="max angular difference to match DOA with a camera user")
    p.add_argument("--doa-user-hold-ms", type=int, default=0, help="hold last matched user location before DOA fallback")
    p.add_argument("--doa-user-min-hits", type=int, default=3, help="consecutive good matches required before using camera user target")
    p.add_argument("--doa-user-match-blend", type=float, default=0.20, help="blend camera user target with DOA target in doa-user-match mode")
    p.add_argument("--doa-user-min-z", type=float, default=0.35, help="ignore users closer than this distance (m)")
    p.add_argument("--doa-user-max-z", type=float, default=2.50, help="ignore users farther than this distance (m)")
    p.add_argument("--doa-user-side-threshold-m", type=float, default=0.08, help="minimum |x| to compare DOA/user left-right side")
    p.add_argument("--update-hz", type=float, default=6.0)
    p.add_argument("--attend-speed", default="medium")
    p.add_argument("--slack-pitch", type=float, default=15.0)
    p.add_argument("--slack-yaw", type=float, default=10.0)
    p.add_argument("--slack-timeout", type=int, default=3000)
    a = p.parse_args()

    idx = [int(s.strip()) for s in a.mic_channels.split(",")]
    vad_mic = int(a.vad_mic)
    if vad_mic < 0 or vad_mic >= len(idx):
        vad_mic = 0
    q = queue.Queue(maxsize=16)
    furhat = FurhatWS(
        a.furhat_ip,
        a.furhat_port,
        a.furhat_auth_key,
        a.attend_speed,
        a.slack_pitch,
        a.slack_yaw,
        a.slack_timeout,
    )
    srp = SRPPhatDOA(
        MIC_XY,
        fs=a.fs,
        az_step_deg=a.srp_az_step_deg,
        interp=a.srp_interp,
        f_low_hz=a.srp_f_low_hz,
        f_high_hz=a.srp_f_high_hz,
    )
    doa_cfg = DOAConfig(
        fs=a.fs,
        frame_ms=a.frame_ms,
        vad_mic=vad_mic,
        vad_threshold=a.vad_threshold,
        vad_smooth_alpha=a.vad_smooth_alpha,
        vad_update_threshold=a.vad_update_threshold,
        energy_threshold=a.energy_threshold,
        energy_update_threshold=a.energy_update_threshold,
        noise_alpha=a.noise_alpha,
        snr_speech_ratio=a.snr_speech_ratio,
        snr_speech_add=a.snr_speech_add,
        snr_update_ratio=a.snr_update_ratio,
        snr_update_add=a.snr_update_add,
        speech_hold_ms=a.speech_hold_ms,
        doa_quality_threshold=a.doa_quality_threshold,
    )
    doa_est = DOAEstimator(doa_cfg, srp, SileroGate(a.vad_threshold, a.fs))
    sm_az = None
    locked_az = None
    pending_az = None
    pending_count = 0
    switch_count = 0
    last_cam_xyz = None
    last_cam_ts = 0.0
    last_doa_match_xyz = None
    last_doa_match_ts = 0.0
    active_doa_user_id = None
    active_doa_user_hits = 0
    last_users_once_ts = 0.0
    last_send = 0.0
    last_idle_log = 0.0
    min_period = 1.0 / max(a.update_hz, 1e-3)
    if a.attend_mode in ("hybrid", "closest-user", "doa-user-match"):
        furhat.start_users_stream()

    def cb(indata, frames, time_info, status):
        if not q.full(): q.put_nowait(indata.copy())

    try:
        if a.attend_mode == "closest-user":
            print("[tracker] running. Ctrl+C to stop.")
            while True:
                now = time.time()
                if now - last_send >= min_period:
                    furhat.attend_closest_user()
                    furhat.get_users()
                    last_send = now
                    print("[track] mode=closest-user src=closest-user")
                time.sleep(0.02)

        with sd.InputStream(
            device=a.device,
            samplerate=a.fs,
            channels=a.channels,
            dtype="int16",
            blocksize=int(a.fs * a.frame_ms / 1000),
            callback=cb,
        ):
            print("[tracker] running. Ctrl+C to stop.")
            while True:
                frame = q.get()
                mics_i16 = frame[:, idx]
                now = time.time()
                obs = doa_est.process(mics_i16, now)
                if obs.speech_ended:
                    locked_az = None
                    sm_az = None
                    pending_az = None
                    pending_count = 0
                    switch_count = 0
                    active_doa_user_id = None
                    active_doa_user_hits = 0
                    last_doa_match_xyz = None
                    last_doa_match_ts = 0.0
                if not obs.speech_active:
                    if now - last_idle_log >= 1.0:
                        print(f"[idle] vad={float(obs.speech_prob):4.2f} e={obs.energy:6.1f} noise={float(obs.noise_energy):6.1f} gate={obs.speech_gate_energy:6.1f}")
                        last_idle_log = now
                    continue

                doa = float("nan")
                qd = float("nan")
                doa_updated = False
                if obs.doa_deg is not None:
                    doa = float(obs.doa_deg)
                if obs.doa_conf is not None:
                    qd = float(obs.doa_conf)
                doa_updated = bool(obs.doa_updated)
                if doa_updated:
                    az1, az2 = map_doa_to_az(
                        doa=doa,
                        board_cw=a.board_cw,
                        offset=a.board_zero_offset,
                        add_180=a.add_180,
                        front_only=a.front_only,
                        max_az=a.max_az_deg,
                    )
                    ref = locked_az if locked_az is not None else (sm_az if sm_az is not None else 0.0)
                    az_candidates = (az1, az2) if a.use_mirror_branch else (az1,)
                    az = min(az_candidates, key=lambda t: abs(cdelta(ref, t)))
                    if pending_az is None or abs(cdelta(pending_az, az)) > a.consistency_deg:
                        pending_az = az
                        pending_count = 1
                    else:
                        pending_az = cblend(pending_az, az, 0.5)
                        pending_count += 1

                    if pending_count >= max(1, a.min_consistent_updates):
                        if locked_az is None:
                            locked_az = pending_az
                            switch_count = 0
                        else:
                            jump = abs(cdelta(locked_az, pending_az))
                            if jump <= a.max_jump_deg:
                                locked_az = cblend(locked_az, pending_az, a.lock_alpha)
                                switch_count = 0
                            elif jump >= a.speaker_switch_deg:
                                # New active speaker on another side: unlock after a few consistent updates.
                                switch_count += 1
                                if switch_count >= max(1, a.speaker_switch_updates):
                                    locked_az = pending_az
                                    switch_count = 0
                            else:
                                switch_count = 0

                if sm_az is None:
                    if locked_az is not None:
                        sm_az = locked_az
                elif locked_az is not None:
                    sm_az = cblend(sm_az, locked_az, a.smooth_alpha)

                if sm_az is None and a.attend_mode != "hybrid":
                    continue

                if time.time() - last_send < min_period: continue
                doa_available = sm_az is not None
                if doa_available:
                    az_cmd = max(-abs(a.max_az_deg), min(abs(a.max_az_deg), float(sm_az) * float(a.az_gain)))
                    ang = math.radians(az_cmd)
                    x = float(a.target_distance_m * math.sin(ang))
                    if a.flip_x: x = -x
                    y = float(a.target_y_m)
                    z = float(a.target_distance_m * math.cos(ang))
                else:
                    x = 0.0
                    y = float(a.target_y_m)
                    z = float(a.target_distance_m)
                source = "doa"
                if a.attend_mode == "closest-user":
                    furhat.attend_closest_user()
                    source = "closest-user"
                elif a.attend_mode == "hybrid":
                    users = furhat.get_users()
                    if users:
                        doa_side = 0
                        if doa_available and abs(x) >= a.hybrid_side_threshold_m:
                            doa_side = 1 if x > 0 else -1
                        chosen = users[0]
                        if doa_side != 0:
                            side_users = []
                            for u in users:
                                loc = u.get("location", {}) if isinstance(u, dict) else {}
                                ux = float(loc.get("x", 0.0)) if isinstance(loc, dict) else 0.0
                                if abs(ux) >= a.hybrid_side_threshold_m and (1 if ux > 0 else -1) == doa_side:
                                    side_users.append(u)
                            if side_users:
                                chosen = side_users[0]
                        loc = chosen.get("location", {}) if isinstance(chosen, dict) else {}
                        ux = float(loc.get("x", x)) if isinstance(loc, dict) else x
                        uy = float(loc.get("y", y)) if isinstance(loc, dict) else y
                        uz = float(loc.get("z", z)) if isinstance(loc, dict) else z
                        b = max(0.0, min(1.0, a.hybrid_doa_x_blend)) if doa_available else 0.0
                        x = (1.0 - b) * ux + b * x
                        y = uy
                        z = uz
                        last_cam_xyz = (x, y, z)
                        last_cam_ts = time.time()
                        source = "hybrid-camera"
                        furhat.attend(x, y, z)
                    else:
                        age_ms = (time.time() - last_cam_ts) * 1000.0
                        if a.hybrid_no_camera_policy == "hold" and last_cam_xyz is not None and age_ms <= a.hybrid_camera_hold_ms:
                            x, y, z = last_cam_xyz
                            furhat.attend(x, y, z)
                            source = "hybrid-hold"
                        elif a.hybrid_no_camera_policy == "closest-user":
                            furhat.attend_closest_user()
                            source = "closest-user-fallback"
                        elif a.hybrid_no_camera_policy == "doa" and doa_available:
                            furhat.attend(x, y, z)
                            source = "doa-fallback"
                        else:
                            continue
                elif a.attend_mode == "doa-user-match":
                    if not doa_available:
                        continue
                    now = time.time()
                    users = furhat.get_users()
                    if (not users) or (now - last_users_once_ts >= 0.8):
                        furhat.request_users_once()
                        last_users_once_ts = now
                        users = furhat.get_users()

                    matched_uid = None
                    matched_xyz = None
                    best_diff = 1e9
                    if doa_updated:
                        doa_bearing = float(sm_az)
                        doa_side = 0
                        if abs(x) >= a.doa_user_side_threshold_m:
                            doa_side = 1 if x > 0 else -1
                        for i, u in enumerate(users):
                            xyz_u = user_location_xyz(u)
                            if xyz_u is None:
                                continue
                            ux, uy, uz = xyz_u
                            if uz < a.doa_user_min_z or uz > a.doa_user_max_z:
                                continue
                            ub = math.degrees(math.atan2(ux, uz))
                            user_side = 0
                            if abs(ux) >= a.doa_user_side_threshold_m:
                                user_side = 1 if ux > 0 else -1
                            if doa_side != 0 and user_side != 0 and doa_side != user_side:
                                continue
                            diff = abs(cdelta(doa_bearing, ub))
                            if diff < best_diff:
                                best_diff = diff
                                matched_xyz = (ux, uy, uz)
                                matched_uid = str(u.get("id", u.get("userId", u.get("user_id", i))))

                    if matched_xyz is not None and best_diff <= abs(a.doa_user_max_diff_deg):
                        if matched_uid == active_doa_user_id:
                            active_doa_user_hits += 1
                        else:
                            active_doa_user_id = matched_uid
                            active_doa_user_hits = 1

                        if active_doa_user_hits >= max(1, a.doa_user_min_hits):
                            ux, uy, uz = matched_xyz
                            b = max(0.0, min(1.0, a.doa_user_match_blend))
                            x = (1.0 - b) * x + b * ux
                            y = (1.0 - b) * y + b * uy
                            z = (1.0 - b) * z + b * uz
                            furhat.attend(x, y, z)
                            source = "doa-matched-user"
                            last_doa_match_xyz = (x, y, z)
                            last_doa_match_ts = now
                        else:
                            furhat.attend(x, y, z)
                            source = "doa-prelock"
                    else:
                        active_doa_user_id = None
                        active_doa_user_hits = 0
                        age_ms = (now - last_doa_match_ts) * 1000.0
                        if last_doa_match_xyz is not None and age_ms <= a.doa_user_hold_ms:
                            x, y, z = last_doa_match_xyz
                            furhat.attend(x, y, z)
                            source = "doa-match-hold"
                        else:
                            furhat.attend(x, y, z)
                            source = "doa-fallback"
                else:
                    furhat.attend(x, y, z)
                last_send = time.time()
                az_log = sm_az if sm_az is not None else float("nan")
                print(f"[track] mode={a.attend_mode} src={source} doa={doa:7.2f} conf={qd:4.2f} vad={obs.vad_prob:4.2f} e={obs.energy:6.1f} az={az_log:7.2f} xyz=({x:6.3f},{y:5.2f},{z:6.3f})")
    except KeyboardInterrupt:
        print("\n[tracker] stopped.")


if __name__ == "__main__":
    main()
