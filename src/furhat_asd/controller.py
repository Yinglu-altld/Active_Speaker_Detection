from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from furhat_asd.audio.audio_stream import SoundDeviceAudioInput
from furhat_asd.audio.doa_srp_phat import SrpPhatConfig, SrpPhatDoa
from furhat_asd.audio.vad_gate import VadGate, VadGateConfig
from furhat_asd.audio.vad_models import EnergyVad, EnergyVadConfig, VadBackend
from furhat_asd.config import AppConfig
from furhat_asd.fusion.user_selector import UserSelector, UserSelectorConfig
from furhat_asd.model import FurhatUser
from furhat_asd.output.segment_logger import SegmentDoa, SegmentLogger, SegmentRecord
from furhat_asd.realtime.client import FurhatRealtimeClient, FurhatMessage
from furhat_asd.utils.angles import (
    circular_dispersion_deg,
    circular_dispersion_weighted_deg,
    circular_mean_deg,
    circular_mean_weighted_deg,
    circular_mode_weighted_deg,
)
from furhat_asd.vision.addressing import FurhatUserPoseAddressingEstimator, NoopAddressingEstimator
from furhat_asd.net.udp_json import UdpJsonReceiver


@dataclass
class SharedState:
    users: list[FurhatUser]
    last_frame_ts_ms: int | None
    last_jpeg: bytes | None


def _parse_users(msg: FurhatMessage) -> list[FurhatUser]:
    raw_users = msg.data.get("users", [])
    out: list[FurhatUser] = []
    if not isinstance(raw_users, list):
        return out
    for u in raw_users:
        if not isinstance(u, dict):
            continue
        user_id = u.get("id") or u.get("user_id") or u.get("userId") or u.get("userid") or ""
        user_id = str(user_id)
        pos = u.get("pos") or u.get("position") or u.get("location") or u.get("loc")
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
            out.append(FurhatUser(user_id=user_id, x_m=x_m, y_m=y_m, z_m=z_m, raw=u))
    return out


async def _furhat_ingest(client: FurhatRealtimeClient, shared: SharedState) -> None:
    log = logging.getLogger("furhat_asd.furhat_ingest")
    last_users_log_ms = 0
    last_cam_log_ms = 0
    last_user_ids: tuple[str, ...] = ()
    async for msg in client.messages():
        # Different Furhat builds/versions use slightly different event type strings.
        # Be robust: detect payload shapes rather than relying only on msg.type.
        if isinstance(msg.data.get("users"), list):
            shared.users = _parse_users(msg)
            user_ids = tuple(u.user_id for u in shared.users)
            if user_ids != last_user_ids:
                last_user_ids = user_ids
                log.info("users_changed count=%d ids=%s", len(user_ids), list(user_ids))
            now_ms = int(time.time() * 1000)
            if now_ms - last_users_log_ms >= 2000:
                last_users_log_ms = now_ms
                sample = []
                for u in shared.users[:3]:
                    sample.append({"id": u.user_id, "x": u.x_m, "z": u.z_m})
                log.info("users=%d sample=%s", len(shared.users), sample)
        elif isinstance(msg.data.get("image"), str):
            b64 = msg.data.get("image")
            if isinstance(b64, str) and b64:
                try:
                    shared.last_jpeg = base64.b64decode(b64)
                    shared.last_frame_ts_ms = int(time.time() * 1000)
                    now_ms = shared.last_frame_ts_ms
                    if now_ms - last_cam_log_ms >= 5000:
                        last_cam_log_ms = now_ms
                        log.info("camera_frame bytes=%d", len(shared.last_jpeg))
                except Exception:
                    continue
        else:
            log.debug("furhat msg type=%s keys=%s", msg.type, list(msg.data.keys()))


async def run_controller(cfg: AppConfig) -> None:
    logging.basicConfig(level=getattr(logging, cfg.logging.level.upper(), logging.INFO))
    # Avoid drowning useful app logs in third-party debug noise.
    logging.getLogger("websockets").setLevel(logging.INFO)
    log = logging.getLogger("furhat_asd")

    client = FurhatRealtimeClient(cfg.furhat.ip, cfg.furhat.ws_port, cfg.furhat.api_key)
    await client.connect()
    log.info("Connected to Furhat Realtime API at %s", client.ws_url)
    await client.start_users()
    await client.start_camera()
    # Force an initial snapshot so we see data even if streams are idle.
    try:
        await client.users_once()
        await client.camera_once()
    except Exception:
        pass

    shared = SharedState(users=[], last_frame_ts_ms=None, last_jpeg=None)
    ingest_task = asyncio.create_task(_furhat_ingest(client, shared))

    audio_in = None
    vad_backend: VadBackend | None = None
    gate: VadGate | None = None
    doa_estimator: SrpPhatDoa | None = None
    udp_audio: UdpJsonReceiver | None = None

    if cfg.audio.mode == "local":
        audio_in = SoundDeviceAudioInput(
            device=cfg.audio.device,
            sample_rate=cfg.audio.sample_rate,
            channels=cfg.audio.channels,
            block_ms=cfg.audio.block_ms,
        )
        await audio_in.start()

        fs = audio_in.sample_rate

        if cfg.vad.backend == "energy":
            vad_backend = EnergyVad(EnergyVadConfig(threshold=cfg.vad.energy_threshold))
            log.warning("Using Energy VAD backend (fallback). Prefer silero/webrtc for real usage.")
        elif cfg.vad.backend == "webrtc":
            from furhat_asd.audio.vad_models import WebRtcVad, WebRtcVadConfig

            vad_backend = WebRtcVad(
                WebRtcVadConfig(aggressiveness=cfg.vad.webrtc_aggressiveness, sample_rate=fs)
            )
            log.info("Using WebRTC VAD backend.")
        elif cfg.vad.backend == "silero":
            from furhat_asd.audio.vad_models import SileroVad, SileroVadConfig

            vad_backend = SileroVad(SileroVadConfig(model_path=cfg.vad.silero_model_path, sample_rate=fs))
            log.info("Using Silero VAD backend.")
        else:
            raise RuntimeError(f"Unsupported VAD backend: {cfg.vad.backend}")

        gate = VadGate(
            VadGateConfig(
                speech_on_conf=cfg.control.speech_on_conf,
                speech_on_ms=cfg.control.speech_on_ms,
                speech_off_conf=cfg.control.speech_off_conf,
                speech_off_ms=cfg.control.speech_off_ms,
                min_burst_ms=cfg.control.min_burst_ms,
            )
        )

        if cfg.doa.enabled:
            if cfg.doa.mic_positions_m is None:
                log.warning("DOA enabled but mic_positions_m missing; disabling DOA.")
            else:
                mic_pos = np.asarray(cfg.doa.mic_positions_m, dtype=np.float64)
                doa_estimator = SrpPhatDoa(
                    SrpPhatConfig(
                        mic_positions_m=mic_pos,
                        sample_rate=fs,
                        search_step_deg=cfg.doa.search_step_deg,
                        gcc_interp=cfg.doa.gcc_interp,
                    )
                )
        channel_idx = cfg.audio.channel_indices
        if channel_idx is not None:
            if not isinstance(channel_idx, list) or not channel_idx:
                raise RuntimeError("audio.channel_indices must be a non-empty list or null")
            if any((not isinstance(i, int)) or i < 0 or i >= cfg.audio.channels for i in channel_idx):
                raise RuntimeError("audio.channel_indices contains invalid indices for configured audio.channels")
            if doa_estimator is not None and cfg.doa.mic_positions_m is not None and len(channel_idx) != len(cfg.doa.mic_positions_m):
                raise RuntimeError("len(audio.channel_indices) must match len(doa.mic_positions_m) for DOA")

        doa_frame_ms = max(20, int(cfg.doa.frame_ms))
        doa_frame_frames = int(fs * (doa_frame_ms / 1000.0))
        doa_buf: list[np.ndarray] = []
        doa_buf_frames = 0
    elif cfg.audio.mode == "udp":
        udp_audio = UdpJsonReceiver(cfg.udp_audio.listen_host, cfg.udp_audio.listen_port)
        await udp_audio.start()
        log.info("Listening for UDP audio events on %s:%d", cfg.udp_audio.listen_host, cfg.udp_audio.listen_port)
    else:
        raise RuntimeError(f"Unsupported audio.mode: {cfg.audio.mode}")

    selector = UserSelector(
        UserSelectorConfig(
            doa_sigma_deg=cfg.control.doa_sigma_deg,
            doa_offset_deg=cfg.control.doa_offset_deg,
            doa_sign=cfg.control.doa_sign,
            switch_margin_ratio=cfg.control.switch_margin_ratio,
            switch_hold_ms=cfg.control.switch_hold_ms,
        )
    )

    doa_window: list[float] = []
    doa_window_ts: list[int] = []
    doa_window_confs: list[float] = []
    doa_window_ms = max(200, int(cfg.control.doa_window_ms))

    last_attended: str | None = None
    last_speech_off_ts: int | None = None
    addressing = FurhatUserPoseAddressingEstimator() if cfg.vision.enabled else NoopAddressingEstimator()

    seg_logger = SegmentLogger(cfg.output.segments_jsonl)
    seg_counter = 0
    seg_id: str | None = None
    seg_start_ms: int | None = None
    seg_doa_degs_cam: list[float] = []
    seg_doa_confs: list[float] = []
    seg_users_snapshot: dict[str, tuple[float, float]] = {}

    last_state_log_ms = 0
    last_vad_log_ms = 0

    # UDP audio current state (when audio.mode == "udp")
    udp_speech_on = False
    udp_prev_speech_on = False
    udp_seg_id: str | None = None
    udp_doa_deg_mic_smooth: float | None = None
    udp_doa_conf: float | None = None
    udp_doa_spread_deg: float | None = None

    try:
        tick_s = 1.0 / max(1.0, float(cfg.control.loop_hz))
        while True:
            now_ms = int(time.time() * 1000)

            if cfg.audio.mode == "local":
                assert audio_in is not None and vad_backend is not None and gate is not None
                start = time.time()
                while time.time() - start < tick_s:
                    chunk = await audio_in.read()
                    ts_ms = chunk.ts_ms
                    pcm_full = chunk.pcm_f32
                    pcm = pcm_full[:, channel_idx] if channel_idx is not None else pcm_full

                    if cfg.audio.vad_channel_index is not None:
                        vi = int(cfg.audio.vad_channel_index)
                        if vi < 0 or vi >= cfg.audio.channels:
                            raise RuntimeError("audio.vad_channel_index is out of range for audio.channels")
                        mono = pcm_full[:, vi]
                    else:
                        # Default: average across selected channels for robustness at distance.
                        mono_avg = pcm.mean(axis=1) if pcm.ndim == 2 and pcm.shape[1] > 1 else pcm[:, 0]
                        rms_avg = float(np.sqrt(np.mean(np.square(mono_avg), axis=0) + 1e-12))
                        speech_prob_avg = float(vad_backend.speech_prob(mono_avg))

                        # If VAD is struggling (quiet speaker / directional capture), fall back to
                        # the single loudest channel for this chunk. This often boosts sensitivity
                        # without forcing a fixed channel in the config.
                        mono = mono_avg
                        rms = rms_avg
                        speech_prob = speech_prob_avg
                        if pcm.ndim == 2 and pcm.shape[1] > 1 and speech_prob_avg < 0.20:
                            rms_per_ch = np.sqrt(np.mean(np.square(pcm), axis=0) + 1e-12)
                            best_i = int(np.argmax(rms_per_ch))
                            mono_best = pcm[:, best_i]
                            speech_prob_best = float(vad_backend.speech_prob(mono_best))
                            if speech_prob_best > speech_prob_avg:
                                mono = mono_best
                                rms = float(rms_per_ch[best_i])
                                speech_prob = speech_prob_best
                                if log.isEnabledFor(logging.DEBUG):
                                    log.debug(
                                        "vad using max-rms channel=%d (avg_prob=%.2f best_prob=%.2f)",
                                        best_i,
                                        speech_prob_avg,
                                        speech_prob_best,
                                    )
                        else:
                            if log.isEnabledFor(logging.DEBUG):
                                speech_prob = speech_prob_avg
                                rms = rms_avg
                    if cfg.audio.vad_channel_index is not None:
                        rms = float(np.sqrt(np.mean(np.square(mono), axis=0) + 1e-12))
                        speech_prob = float(vad_backend.speech_prob(mono))
                    if log.isEnabledFor(logging.DEBUG) and (ts_ms - last_vad_log_ms) >= 1000:
                        last_vad_log_ms = ts_ms
                        log.debug("vad speech_prob=%.2f rms=%.4f", speech_prob, rms)
                    changed = gate.update(ts_ms, speech_prob)
                    if changed:
                        if gate.speech_on:
                            log.info("speech_on")
                            seg_counter += 1
                            seg_id = f"s{seg_counter:06d}"
                            seg_start_ms = ts_ms
                            seg_doa_degs_cam = []
                            seg_doa_confs = []
                            seg_users_snapshot = {}
                        else:
                            last_speech_off_ts = ts_ms
                            log.info("speech_off")
                            users_for_scoring = (
                                [FurhatUser(user_id=uid, x_m=x, z_m=z) for uid, (x, z) in seg_users_snapshot.items()]
                                if seg_users_snapshot
                                else shared.users
                            )
                            _finalize_segment(
                                cfg=cfg,
                                seg_logger=seg_logger,
                                shared_users=users_for_scoring,
                                seg_id=seg_id,
                                seg_start_ms=seg_start_ms,
                                seg_end_ms=ts_ms,
                                seg_doa_degs_cam=seg_doa_degs_cam,
                                seg_doa_confs=seg_doa_confs,
                            )
                            seg_id = None
                            seg_start_ms = None
                            seg_doa_degs_cam = []
                            seg_doa_confs = []
                            seg_users_snapshot = {}
                            doa_buf = []
                            doa_buf_frames = 0
                            doa_window = []
                            doa_window_ts = []
                            doa_window_confs = []

                    if gate.speech_on:
                        # Snapshot user positions during the segment so segment scoring is robust to movement
                        # and doesn't depend on the user list at the exact moment speech ends.
                        for u in shared.users:
                            if u.x_m is None or u.z_m is None:
                                continue
                            seg_users_snapshot[u.user_id] = (float(u.x_m), float(u.z_m))

                    if doa_estimator is not None and gate.speech_on:
                        doa_buf.append(pcm)
                        doa_buf_frames += int(pcm.shape[0])
                        while doa_buf and doa_buf_frames > doa_frame_frames:
                            removed = doa_buf.pop(0)
                            doa_buf_frames -= int(removed.shape[0])
                        est = doa_estimator.estimate_azimuth_deg(np.concatenate(doa_buf, axis=0)) if doa_buf else None
                        if est is not None:
                            doa_deg, doa_conf = est
                            # Keep a small floor so we can still localize quieter speakers,
                            # but avoid polluting the window with pure noise.
                            if doa_conf > 0.02:
                                doa_window.append(float(doa_deg))
                                doa_window_ts.append(ts_ms)
                                doa_window_confs.append(float(doa_conf))
                                while doa_window_ts and (ts_ms - doa_window_ts[0]) > doa_window_ms:
                                    doa_window_ts.pop(0)
                                    doa_window.pop(0)
                                    doa_window_confs.pop(0)
                                doa_cam = (cfg.control.doa_sign * float(doa_deg) + float(cfg.control.doa_offset_deg)) % 360.0
                                if seg_id is not None:
                                    seg_doa_degs_cam.append(float(doa_cam))
                                    seg_doa_confs.append(float(doa_conf))
            else:
                assert udp_audio is not None
                # Drain any pending UDP messages quickly.
                drain_start = time.time()
                while time.time() - drain_start < tick_s:
                    try:
                        msg = await asyncio.wait_for(udp_audio.recv(), timeout=tick_s)
                    except asyncio.TimeoutError:
                        break
                    payload = msg.payload
                    msg_type = str(payload.get("type", ""))
                    if msg_type == "audio.state":
                        udp_prev_speech_on = udp_speech_on
                        udp_speech_on = bool(payload.get("speech_on", False))
                        if udp_prev_speech_on and not udp_speech_on:
                            last_speech_off_ts = now_ms
                        udp_seg_id = payload.get("seg_id") if isinstance(payload.get("seg_id"), str) else udp_seg_id
                        doa = payload.get("doa", {})
                        if isinstance(doa, dict):
                            udp_doa_deg_mic_smooth = float(doa.get("azimuth_deg_mic_smooth")) if doa.get("azimuth_deg_mic_smooth") is not None else udp_doa_deg_mic_smooth
                            udp_doa_conf = float(doa.get("conf")) if doa.get("conf") is not None else udp_doa_conf
                            udp_doa_spread_deg = float(doa.get("spread_deg")) if doa.get("spread_deg") is not None else udp_doa_spread_deg
                    elif msg_type == "audio.segment_start":
                        udp_seg_id = str(payload.get("seg_id", udp_seg_id or ""))
                        log.info("udp segment_start seg_id=%s", udp_seg_id)
                    elif msg_type == "audio.segment_end":
                        seg_id_end = payload.get("seg_id")
                        if isinstance(seg_id_end, str):
                            last_speech_off_ts = now_ms
                            doa = payload.get("doa", None)
                            doa_summary = None
                            if isinstance(doa, dict):
                                try:
                                    az_mic = float(doa.get("azimuth_deg_mic"))
                                    az_cam = (cfg.control.doa_sign * az_mic + float(cfg.control.doa_offset_deg)) % 360.0
                                    doa_summary = SegmentDoa(
                                        azimuth_deg=float(az_cam),
                                        conf=float(max(0.0, min(1.0, float(doa.get("conf", 0.0))))),
                                        spread_deg=float(doa.get("spread_deg", 180.0)),
                                    )
                                except Exception:
                                    doa_summary = None
                            # write segment record (scores computed on PC using current users)
                            t_start_ms = int(payload.get("t_start_ms", now_ms))
                            t_end_ms = int(payload.get("t_end_ms", now_ms))
                            doa_scores = _doa_scores(cfg, doa_summary, shared.users)
                            seg_logger.write(
                                SegmentRecord(
                                    seg_id=seg_id_end,
                                    t_start_ms=t_start_ms,
                                    t_end_ms=t_end_ms,
                                    doa=doa_summary,
                                    doa_scores=doa_scores,
                                )
                            )
                            if doa_summary is not None:
                                log.info(
                                    "udp segment_end seg_id=%s doa_cam=%.1f conf=%.2f spread=%.1f",
                                    seg_id_end,
                                    doa_summary.azimuth_deg,
                                    doa_summary.conf,
                                    doa_summary.spread_deg,
                                )
                            else:
                                log.info("udp segment_end seg_id=%s doa_cam=none", seg_id_end)

            # Attention decision tick
            if cfg.audio.mode == "local":
                assert gate is not None
                speech_on = gate.speech_on
                doa_for_choice = (
                    circular_mean_weighted_deg(doa_window, doa_window_confs) if doa_window else None
                )
                doa_disp = (
                    circular_dispersion_weighted_deg(doa_window, doa_window_confs) if doa_window else None
                )
                doa_disp = float(doa_disp) if doa_disp is not None else 180.0
                doa_signal_conf = float(np.mean(doa_window_confs)) if doa_window_confs else 0.0
                doa_conf_smooth = float(
                    max(
                        0.0,
                        min(
                            1.0,
                            (doa_signal_conf / max(1e-6, float(cfg.control.doa_usable_min_conf)))
                            * (1.0 - (doa_disp / max(1e-6, float(cfg.control.doa_usable_max_spread_deg)))),
                        ),
                    )
                )
                # Decide "DOA usable" based on the smooth combined metric, not the raw mean confidence.
                # In real rooms, raw SRP-PHAT confidence can be conservative even when the angle is stable.
                doa_ok = (doa_for_choice is not None) and (doa_conf_smooth >= float(cfg.control.doa_usable_min_conf)) and (
                    doa_disp <= float(cfg.control.doa_usable_max_spread_deg)
                )
                doa_spread_log = doa_disp
                doa_signal_conf_log = doa_signal_conf
            else:
                speech_on = udp_speech_on
                doa_for_choice = udp_doa_deg_mic_smooth
                doa_conf_smooth = float(udp_doa_conf) if udp_doa_conf is not None else 0.0
                spread = float(udp_doa_spread_deg) if udp_doa_spread_deg is not None else 180.0
                doa_ok = (doa_for_choice is not None) and (doa_conf_smooth >= float(cfg.control.doa_usable_min_conf)) and (
                    spread <= float(cfg.control.doa_usable_max_spread_deg)
                )
                doa_spread_log = spread
                doa_signal_conf_log = doa_conf_smooth

            active_id: str | None = None
            active_conf = 0.0
            doa_cam: float | None = None
            combined_conf = 0.0

            if speech_on:
                if doa_for_choice is not None and doa_ok:
                    active_id, active_conf, doa_cam = selector.choose(now_ms, float(doa_for_choice), shared.users)
                    combined_conf = float(min(1.0, 0.6 * active_conf + 0.4 * doa_conf_smooth))
                    # Practical fallback: if Furhat sees exactly one user but DOA is unstable
                    # (common in noisy / reverberant spaces), don't block attention on DOA.
                    if len(shared.users) == 1 and (doa_conf_smooth < 0.2 or active_conf < 0.1):
                        active_id = shared.users[0].user_id
                        active_conf = 0.4
                        doa_cam = None
                        combined_conf = 0.4
                elif selector.active_user_id is not None and any(u.user_id == selector.active_user_id for u in shared.users):
                    # DOA is currently ambiguous/low-confidence: keep the last active speaker rather than flickering.
                    active_id = selector.active_user_id
                    active_conf = 0.15
                    doa_cam = None
                    combined_conf = 0.15
                elif len(shared.users) == 1:
                    # Fallback: if Furhat sees exactly one user but DOA isn't stable/available yet,
                    # still attend that user during speech.
                    active_id = shared.users[0].user_id
                    active_conf = 0.4
                    doa_cam = None
                    combined_conf = 0.4

                active_user = next((u for u in shared.users if u.user_id == active_id), None) if active_id else None
                addressing_prob = addressing.estimate(now_ms, shared.last_jpeg, active_user).prob

                if now_ms - last_state_log_ms >= 2000:
                    last_state_log_ms = now_ms
                    doa_str = f"{doa_cam:.1f}" if doa_cam is not None else "none"
                    log.info(
                        "state speech_on users=%d doa=%s doa_ok=%s doa_raw=%.2f doa_smooth=%.2f spread=%.1f active=%s conf=%.2f addressing=%.2f",
                        len(shared.users),
                        doa_str,
                        str(bool(doa_ok)),
                        float(doa_signal_conf_log),
                        float(doa_conf_smooth),
                        float(doa_spread_log),
                        active_id,
                        combined_conf,
                        addressing_prob,
                    )

                if active_id and combined_conf >= 0.2 and addressing_prob >= cfg.control.attend_addressing_threshold:
                    if last_attended != active_id:
                        await client.attend_user(active_id)
                        last_attended = active_id
                        if doa_cam is None:
                            log.info("attend.user=%s conf=%.2f doa_cam=none users=%d", active_id, combined_conf, len(shared.users))
                        else:
                            log.info("attend.user=%s conf=%.2f doa_cam=%.1f users=%d", active_id, combined_conf, doa_cam, len(shared.users))
                else:
                    # During speech_on, low-confidence DOA should not force us to "attend.nobody".
                    # Only drop attention if we are explicitly not addressing the robot.
                    if last_attended is not None and addressing_prob < cfg.control.attend_addressing_threshold:
                        await client.attend_nobody()
                        last_attended = None
                        log.info("attend.nobody (not addressing)")
            else:
                if last_speech_off_ts is not None and now_ms - last_speech_off_ts > 1000:
                    if last_attended is not None:
                        await client.attend_nobody()
                        last_attended = None
                        log.info("attend.nobody (silence)")

            await asyncio.sleep(tick_s)
    except asyncio.CancelledError:
        return
    finally:
        ingest_task.cancel()
        if audio_in is not None:
            await audio_in.stop()
        if udp_audio is not None:
            await udp_audio.stop()
        await client.stop_camera()
        await client.stop_users()
        await client.close()


def _doa_scores(cfg: AppConfig, doa_summary: SegmentDoa | None, users: list[FurhatUser]) -> list[dict[str, Any]]:
    doa_scores: list[dict[str, Any]] = []
    if doa_summary is None:
        return doa_scores
    for u in users:
        if u.x_m is None or u.z_m is None:
            continue
        user_az = np.degrees(np.arctan2(u.x_m, u.z_m))
        user_az = float((user_az + 360.0) % 360.0)
        diff = abs((doa_summary.azimuth_deg - user_az + 180.0) % 360.0 - 180.0)
        sigma = max(1e-3, float(cfg.control.doa_sigma_deg))
        d = float(np.exp(-(diff * diff) / (2.0 * sigma * sigma)))
        doa_scores.append({"userID": u.user_id, "D": d, "az_deg": user_az})
    doa_scores.sort(key=lambda x: x["D"], reverse=True)
    return doa_scores[:10]


def _finalize_segment(
    *,
    cfg: AppConfig,
    seg_logger: SegmentLogger,
    shared_users: list[FurhatUser],
    seg_id: str | None,
    seg_start_ms: int | None,
    seg_end_ms: int,
    seg_doa_degs_cam: list[float],
    seg_doa_confs: list[float],
) -> None:
    if seg_id is None or seg_start_ms is None:
        return
    doa_summary: SegmentDoa | None = None
    if seg_doa_degs_cam:
        doa_center = (
            circular_mode_weighted_deg(
                seg_doa_degs_cam,
                seg_doa_confs if seg_doa_confs else [1.0] * len(seg_doa_degs_cam),
                bin_deg=float(cfg.doa.search_step_deg) if cfg.doa.search_step_deg else 5.0,
            )
            if seg_doa_confs
            else circular_mean_deg(seg_doa_degs_cam)
        )
        doa_spread = (
            circular_dispersion_weighted_deg(seg_doa_degs_cam, seg_doa_confs)
            if seg_doa_confs
            else (circular_dispersion_deg(seg_doa_degs_cam) or 180.0)
        )
        doa_spread = float(doa_spread if doa_spread is not None else 180.0)
        doa_conf_raw = float(np.mean(seg_doa_confs)) if seg_doa_confs else 0.0
        # Penalize unstable segments where DOA jumps around a lot.
        doa_conf = float(max(0.0, min(1.0, doa_conf_raw * (1.0 - min(1.0, doa_spread / 120.0)))))
        if doa_center is not None:
            doa_summary = SegmentDoa(
                azimuth_deg=float(doa_center),
                conf=doa_conf,
                spread_deg=doa_spread,
            )
    doa_scores = _doa_scores(cfg, doa_summary, shared_users)
    seg_logger.write(
        SegmentRecord(
            seg_id=seg_id,
            t_start_ms=seg_start_ms,
            t_end_ms=seg_end_ms,
            doa=doa_summary,
            doa_scores=doa_scores,
        )
    )
