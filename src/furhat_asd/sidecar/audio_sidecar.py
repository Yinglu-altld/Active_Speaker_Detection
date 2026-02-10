from __future__ import annotations

import argparse
import asyncio
import logging
import time

import numpy as np

from furhat_asd.audio.audio_stream import SoundDeviceAudioInput
from furhat_asd.audio.doa_srp_phat import SrpPhatConfig, SrpPhatDoa
from furhat_asd.audio.vad_gate import VadGate, VadGateConfig
from furhat_asd.audio.vad_models import EnergyVad, EnergyVadConfig, VadBackend
from furhat_asd.config import AppConfig, load_config
from furhat_asd.net.udp_json import UdpJsonSender
from furhat_asd.utils.angles import circular_dispersion_deg, circular_mean_deg


async def run_sidecar(cfg: AppConfig) -> None:
    logging.basicConfig(level=getattr(logging, cfg.logging.level.upper(), logging.INFO))
    log = logging.getLogger("furhat_asd.sidecar")

    if not cfg.udp_audio.target_host:
        raise RuntimeError("udp_audio.target_host must be set on the sidecar")

    sender = UdpJsonSender(cfg.udp_audio.target_host, cfg.udp_audio.target_port)

    audio_in = SoundDeviceAudioInput(
        device=cfg.audio.device,
        sample_rate=cfg.audio.sample_rate,
        channels=cfg.audio.channels,
        block_ms=cfg.audio.block_ms,
    )
    await audio_in.start()

    fs = audio_in.sample_rate

    vad_backend: VadBackend
    if cfg.vad.backend == "energy":
        vad_backend = EnergyVad(EnergyVadConfig(threshold=cfg.vad.energy_threshold))
        log.warning("Using Energy VAD backend (fallback). Prefer silero/webrtc for real usage.")
    elif cfg.vad.backend == "webrtc":
        from furhat_asd.audio.vad_models import WebRtcVad, WebRtcVadConfig

        vad_backend = WebRtcVad(WebRtcVadConfig(aggressiveness=cfg.vad.webrtc_aggressiveness, sample_rate=fs))
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

    doa_estimator: SrpPhatDoa | None = None
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

    seg_counter = 0
    seg_id: str | None = None
    seg_start_ms: int | None = None
    seg_doa_degs: list[float] = []
    seg_doa_confs: list[float] = []

    doa_window: list[float] = []
    doa_window_ts: list[int] = []
    doa_window_ms = 1000

    doa_frame_ms = max(20, int(cfg.doa.frame_ms))
    doa_frame_frames = int(fs * (doa_frame_ms / 1000.0))
    doa_buf: list[np.ndarray] = []
    doa_buf_frames = 0

    send_interval_ms = int(1000 / max(1, cfg.udp_audio.send_hz))
    last_state_sent_ms = 0

    try:
        while True:
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
                # Use an average across selected channels for more robust VAD at distance.
                mono = pcm.mean(axis=1) if pcm.ndim == 2 and pcm.shape[1] > 1 else pcm[:, 0]
            speech_prob = float(vad_backend.speech_prob(mono))
            changed = gate.update(ts_ms, speech_prob)

            if changed:
                if gate.speech_on:
                    seg_counter += 1
                    seg_id = f"s{seg_counter:06d}"
                    seg_start_ms = ts_ms
                    seg_doa_degs = []
                    seg_doa_confs = []
                    sender.send({"type": "audio.segment_start", "t_ms": ts_ms, "seg_id": seg_id, "t_start_ms": ts_ms})
                    log.info("speech_on seg_id=%s", seg_id)
                else:
                    if seg_id is not None and seg_start_ms is not None:
                        doa_summary = None
                        if seg_doa_degs:
                            doa_mean = circular_mean_deg(seg_doa_degs)
                            doa_spread = circular_dispersion_deg(seg_doa_degs) or 180.0
                            doa_conf = float(np.mean(seg_doa_confs)) if seg_doa_confs else 0.0
                            if doa_mean is not None:
                                doa_summary = {
                                    "azimuth_deg_mic": float(doa_mean),
                                    "conf": float(max(0.0, min(1.0, doa_conf))),
                                    "spread_deg": float(doa_spread),
                                }
                        sender.send(
                            {
                                "type": "audio.segment_end",
                                "t_ms": ts_ms,
                                "seg_id": seg_id,
                                "t_start_ms": seg_start_ms,
                                "t_end_ms": ts_ms,
                                "doa": doa_summary,
                            }
                        )
                        log.info("speech_off seg_id=%s", seg_id)
                    seg_id = None
                    seg_start_ms = None
                    seg_doa_degs = []
                    seg_doa_confs = []
                    doa_window = []
                    doa_window_ts = []
                    doa_buf = []
                    doa_buf_frames = 0

            if doa_estimator is not None and gate.speech_on:
                doa_buf.append(pcm)
                doa_buf_frames += int(pcm.shape[0])
                while doa_buf and doa_buf_frames > doa_frame_frames:
                    removed = doa_buf.pop(0)
                    doa_buf_frames -= int(removed.shape[0])
                est = doa_estimator.estimate_azimuth_deg(np.concatenate(doa_buf, axis=0)) if doa_buf else None
                if est is not None:
                    doa_deg, doa_conf = est
                    if doa_conf > 0.05:
                        doa_window.append(float(doa_deg))
                        doa_window_ts.append(ts_ms)
                        while doa_window_ts and (ts_ms - doa_window_ts[0]) > doa_window_ms:
                            doa_window_ts.pop(0)
                            doa_window.pop(0)
                        if seg_id is not None:
                            seg_doa_degs.append(float(doa_deg))
                            seg_doa_confs.append(float(doa_conf))

            # Periodically send current state
            if ts_ms - last_state_sent_ms >= send_interval_ms:
                last_state_sent_ms = ts_ms
                doa_smooth = circular_mean_deg(doa_window) if doa_window else None
                doa_spread = circular_dispersion_deg(doa_window) if doa_window else None
                doa_payload = None
                if doa_smooth is not None:
                    disp = float(doa_spread) if doa_spread is not None else 180.0
                    doa_conf_smooth = max(0.0, min(1.0, 1.0 - (disp / 90.0)))
                    doa_payload = {
                        "azimuth_deg_mic_smooth": float(doa_smooth),
                        "conf": float(doa_conf_smooth),
                        "spread_deg": disp,
                    }
                sender.send(
                    {
                        "type": "audio.state",
                        "t_ms": ts_ms,
                        "speech_on": gate.speech_on,
                        "seg_id": seg_id,
                        "speech_prob": speech_prob,
                        "doa": doa_payload,
                    }
                )
    finally:
        await audio_in.stop()


def main() -> None:
    parser = argparse.ArgumentParser(prog="furhat-asd-sidecar")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    asyncio.run(run_sidecar(cfg))


if __name__ == "__main__":
    main()
