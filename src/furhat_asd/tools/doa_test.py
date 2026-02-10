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
from furhat_asd.output.segment_logger import SegmentDoa, SegmentLogger, SegmentRecord
from furhat_asd.utils.angles import (
    circular_dispersion_deg,
    circular_dispersion_weighted_deg,
    circular_mean_deg,
    circular_mean_weighted_deg,
)


async def run_doa_test(cfg: AppConfig) -> None:
    logging.basicConfig(level=getattr(logging, cfg.logging.level.upper(), logging.INFO))
    log = logging.getLogger("furhat_asd.doa_test")

    if cfg.audio.mode != "local":
        raise RuntimeError("DOA test requires audio.mode=local")
    if not cfg.doa.enabled:
        raise RuntimeError("DOA test requires doa.enabled=true")
    if cfg.doa.mic_positions_m is None:
        raise RuntimeError("DOA test requires doa.mic_positions_m")

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
        log.info("Using Energy VAD backend.")
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
        if len(channel_idx) != mic_pos.shape[0]:
            raise RuntimeError("len(audio.channel_indices) must match len(doa.mic_positions_m) for DOA")

    seg_logger = SegmentLogger(cfg.output.segments_jsonl)
    seg_counter = 0
    seg_id: str | None = None
    seg_start_ms: int | None = None
    seg_doa_degs: list[float] = []
    seg_doa_confs: list[float] = []

    doa_window: list[float] = []
    doa_window_ts: list[int] = []
    doa_window_confs: list[float] = []
    doa_window_ms = max(200, int(cfg.control.doa_window_ms))

    doa_frame_ms = max(20, int(cfg.doa.frame_ms))
    doa_frame_frames = int(fs * (doa_frame_ms / 1000.0))
    doa_buf: list[np.ndarray] = []
    doa_buf_frames = 0

    last_print_ms = 0

    def _to_cam(az_mic_deg: float) -> float:
        return (cfg.control.doa_sign * float(az_mic_deg) + float(cfg.control.doa_offset_deg)) % 360.0

    try:
        if channel_idx is None:
            log.info(
                "Starting DOA test (device=%s sr=%dHz dev_channels=%d doa_frame_ms=%d). Speak while facing the array; watch azimuth + spread.",
                cfg.audio.device,
                fs,
                cfg.audio.channels,
                doa_frame_ms,
            )
        else:
            log.info(
                "Starting DOA test (device=%s sr=%dHz dev_channels=%d use_dev_channels=%s doa_frame_ms=%d). Speak while facing the array; watch azimuth + spread.",
                cfg.audio.device,
                fs,
                cfg.audio.channels,
                channel_idx,
                doa_frame_ms,
            )
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
                mono = pcm.mean(axis=1) if pcm.ndim == 2 and pcm.shape[1] > 1 else pcm[:, 0]

            speech_prob = float(vad_backend.speech_prob(mono))
            changed = gate.update(ts_ms, speech_prob)

            if changed and gate.speech_on:
                seg_counter += 1
                seg_id = f"s{seg_counter:06d}"
                seg_start_ms = ts_ms
                seg_doa_degs = []
                seg_doa_confs = []
                log.info("speech_on seg_id=%s", seg_id)

            if gate.speech_on:
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
                        doa_window_confs.append(float(doa_conf))
                        while doa_window_ts and (ts_ms - doa_window_ts[0]) > doa_window_ms:
                            doa_window_ts.pop(0)
                            doa_window.pop(0)
                            doa_window_confs.pop(0)
                        if seg_id is not None:
                            seg_doa_degs.append(float(doa_deg))
                            seg_doa_confs.append(float(doa_conf))

            if changed and not gate.speech_on:
                if seg_id is not None and seg_start_ms is not None:
                    doa_summary = None
                    if seg_doa_degs:
                        doa_mean = circular_mean_deg(seg_doa_degs)
                        doa_spread = circular_dispersion_deg(seg_doa_degs) or 180.0
                        doa_conf = float(np.mean(seg_doa_confs)) if seg_doa_confs else 0.0
                        if doa_mean is not None:
                            doa_summary = SegmentDoa(
                                azimuth_deg=float(doa_mean),
                                conf=float(doa_conf),
                                spread_deg=float(doa_spread),
                            )
                    seg_logger.write(
                        SegmentRecord(
                            seg_id=seg_id,
                            t_start_ms=seg_start_ms,
                            t_end_ms=ts_ms,
                            doa=doa_summary,
                            doa_scores=[],
                        )
                    )
                    if doa_summary is not None:
                        log.info(
                            "speech_off seg_id=%s doa_mic=%.1fdeg doa_cam=%.1fdeg conf=%.2f spread=%.1fdeg",
                            seg_id,
                            doa_summary.azimuth_deg,
                            _to_cam(doa_summary.azimuth_deg),
                            doa_summary.conf,
                            doa_summary.spread_deg,
                        )
                    else:
                        log.info("speech_off seg_id=%s doa=none", seg_id)

                seg_id = None
                seg_start_ms = None
                seg_doa_degs = []
                seg_doa_confs = []
                doa_window = []
                doa_window_ts = []
                doa_buf = []
                doa_buf_frames = 0

            if gate.speech_on and ts_ms - last_print_ms >= 1000:
                last_print_ms = ts_ms
                doa_smooth = circular_mean_deg(doa_window) if doa_window else None
                doa_smooth = (
                    circular_mean_weighted_deg(doa_window, doa_window_confs) if doa_window else None
                )
                disp = (
                    circular_dispersion_weighted_deg(doa_window, doa_window_confs) if doa_window else None
                )
                if doa_smooth is not None:
                    log.info(
                        "doa_smooth_mic=%.1fdeg doa_smooth_cam=%.1fdeg spread=%.1fdeg",
                        float(doa_smooth),
                        _to_cam(float(doa_smooth)),
                        float(disp) if disp is not None else 180.0,
                    )
    finally:
        await audio_in.stop()


def main() -> None:
    parser = argparse.ArgumentParser(prog="furhat-asd-doa-test")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    try:
        asyncio.run(run_doa_test(cfg))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
