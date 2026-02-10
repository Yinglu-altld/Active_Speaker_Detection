from __future__ import annotations

import argparse
import asyncio
import logging
import time

import numpy as np

from furhat_asd.audio.audio_stream import SoundDeviceAudioInput
from furhat_asd.audio.vad_gate import VadGate, VadGateConfig
from furhat_asd.audio.vad_models import EnergyVad, EnergyVadConfig, VadBackend
from furhat_asd.config import AppConfig, load_config


async def run_vad_test(cfg: AppConfig, *, interval_ms: int) -> None:
    logging.basicConfig(level=getattr(logging, cfg.logging.level.upper(), logging.INFO))
    log = logging.getLogger("furhat_asd.vad_test")

    if cfg.audio.mode != "local":
        raise RuntimeError("vad-test requires audio.mode=local")

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

    channel_idx = cfg.audio.channel_indices
    if channel_idx is not None:
        if not isinstance(channel_idx, list) or not channel_idx:
            raise RuntimeError("audio.channel_indices must be a non-empty list or null")
        if any((not isinstance(i, int)) or i < 0 or i >= cfg.audio.channels for i in channel_idx):
            raise RuntimeError("audio.channel_indices contains invalid indices for configured audio.channels")

    interval_s = max(0.05, float(interval_ms) / 1000.0)
    last_print = 0.0

    try:
        log.info(
            "VAD test started (device=%s sr=%dHz dev_channels=%d vad_channel_index=%s).",
            cfg.audio.device,
            fs,
            cfg.audio.channels,
            str(cfg.audio.vad_channel_index) if cfg.audio.vad_channel_index is not None else "auto",
        )
        log.info(
            "Gate: on_conf=%.2f on_ms=%d off_conf=%.2f off_ms=%d",
            cfg.control.speech_on_conf,
            cfg.control.speech_on_ms,
            cfg.control.speech_off_conf,
            cfg.control.speech_off_ms,
        )
        log.info("Speak normally for 2-3 seconds and watch `speech_prob`.")

        while True:
            chunk = await audio_in.read()
            ts_ms = chunk.ts_ms
            pcm_full = chunk.pcm_f32
            pcm_sel = pcm_full[:, channel_idx] if channel_idx is not None else pcm_full

            best_i: int | None = None
            best_devch: int | None = None
            rms_best: float | None = None
            speech_prob_best: float | None = None

            # Always compute "loudest channel" for visibility when debugging.
            if pcm_sel.ndim == 2 and pcm_sel.shape[1] > 1:
                rms_per_ch = np.sqrt(np.mean(np.square(pcm_sel), axis=0) + 1e-12)
                best_i = int(np.argmax(rms_per_ch))
                rms_best = float(rms_per_ch[best_i])
                if channel_idx is not None and 0 <= best_i < len(channel_idx):
                    best_devch = int(channel_idx[best_i])
                else:
                    best_devch = best_i
            if cfg.audio.vad_channel_index is not None:
                vi = int(cfg.audio.vad_channel_index)
                if vi < 0 or vi >= cfg.audio.channels:
                    raise RuntimeError("audio.vad_channel_index is out of range for audio.channels")
                mono = pcm_full[:, vi]
                rms = float(np.sqrt(np.mean(np.square(mono), axis=0) + 1e-12))
                speech_prob = float(vad_backend.speech_prob(mono))
            else:
                mono_avg = pcm_sel.mean(axis=1) if pcm_sel.ndim == 2 and pcm_sel.shape[1] > 1 else pcm_sel[:, 0]
                rms_avg = float(np.sqrt(np.mean(np.square(mono_avg), axis=0) + 1e-12))
                speech_prob_avg = float(vad_backend.speech_prob(mono_avg))

                mono = mono_avg
                rms = rms_avg
                speech_prob = speech_prob_avg

                if pcm_sel.ndim == 2 and pcm_sel.shape[1] > 1 and speech_prob_avg < 0.20:
                    assert best_i is not None
                    mono_best = pcm_sel[:, best_i]
                    speech_prob_best = float(vad_backend.speech_prob(mono_best))
                    if speech_prob_best > speech_prob_avg:
                        mono = mono_best
                        rms = rms_best
                        speech_prob = speech_prob_best
                        log.debug(
                            "vad using max-rms channel=%d (avg_prob=%.2f best_prob=%.2f)",
                            best_i,
                            speech_prob_avg,
                            speech_prob_best,
                        )
            changed = gate.update(ts_ms, speech_prob)

            if changed:
                log.info("gate_change speech_on=%s (speech_prob=%.2f rms=%.4f)", gate.speech_on, speech_prob, rms)

            now = time.time()
            if now - last_print >= interval_s:
                last_print = now
                extra = ""
                if best_devch is not None and rms_best is not None:
                    extra = f" best_devch={best_devch} best_rms={rms_best:.4f}"
                if speech_prob_best is not None:
                    extra += f" best_prob={speech_prob_best:.2f}"
                log.info("speech_prob=%.2f rms=%.4f speech_on=%s%s", speech_prob, rms, gate.speech_on, extra)
    finally:
        await audio_in.stop()


def main() -> None:
    parser = argparse.ArgumentParser(prog="furhat-asd-vad-test")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--interval-ms", type=int, default=250, help="Print interval (default: 250ms)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    try:
        asyncio.run(run_vad_test(cfg, interval_ms=args.interval_ms))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
