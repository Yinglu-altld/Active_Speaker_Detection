from __future__ import annotations

import argparse
import asyncio
import logging
import time

import numpy as np

from furhat_asd.audio.audio_stream import SoundDeviceAudioInput
from furhat_asd.config import AppConfig, load_config


async def run_audio_levels(cfg: AppConfig, *, interval_ms: int) -> None:
    logging.basicConfig(level=getattr(logging, cfg.logging.level.upper(), logging.INFO))
    log = logging.getLogger("furhat_asd.audio_levels")

    if cfg.audio.mode != "local":
        raise RuntimeError("audio-levels requires audio.mode=local")

    audio_in = SoundDeviceAudioInput(
        device=cfg.audio.device,
        sample_rate=cfg.audio.sample_rate,
        channels=cfg.audio.channels,
        block_ms=cfg.audio.block_ms,
    )
    await audio_in.start()

    channel_idx = cfg.audio.channel_indices
    if channel_idx is not None:
        if not isinstance(channel_idx, list) or not channel_idx:
            raise RuntimeError("audio.channel_indices must be a non-empty list or null")
        if any((not isinstance(i, int)) or i < 0 or i >= cfg.audio.channels for i in channel_idx):
            raise RuntimeError("audio.channel_indices contains invalid indices for configured audio.channels")

    interval_s = max(0.05, float(interval_ms) / 1000.0)
    last_print = 0.0
    rms_sum = None
    rms_peak = None
    n_accum = 0

    try:
        log.info(
            "Audio levels started (device=%s sr=%dHz channels=%d). Speak or tap near each mic; watch which channels respond.",
            cfg.audio.device,
            audio_in.sample_rate,
            cfg.audio.channels,
        )
        while True:
            chunk = await audio_in.read()
            pcm = chunk.pcm_f32[:, channel_idx] if channel_idx is not None else chunk.pcm_f32

            # RMS per channel for this block
            block_rms = np.sqrt(np.mean(np.square(pcm), axis=0) + 1e-12)
            if rms_sum is None:
                rms_sum = block_rms
            else:
                rms_sum += block_rms
            if rms_peak is None:
                rms_peak = block_rms
            else:
                rms_peak = np.maximum(rms_peak, block_rms)
            n_accum += 1

            now = time.time()
            if now - last_print >= interval_s:
                last_print = now
                assert rms_sum is not None
                assert rms_peak is not None
                avg_rms = rms_sum / max(1, n_accum)
                peak_rms = rms_peak
                rms_sum = None
                rms_peak = None
                n_accum = 0

                max_avg_ch = int(np.argmax(avg_rms))
                max_peak_ch = int(np.argmax(peak_rms))

                if channel_idx is None:
                    labels = [f"devch{i}" for i in range(avg_rms.shape[0])]
                    max_avg_label = f"devch{max_avg_ch}"
                    max_peak_label = f"devch{max_peak_ch}"
                else:
                    labels = [f"devch{i}" for i in channel_idx]
                    max_avg_label = f"devch{channel_idx[max_avg_ch]}"
                    max_peak_label = f"devch{channel_idx[max_peak_ch]}"

                avg_str = " ".join([f"{labels[i]}:{v:.4f}" for i, v in enumerate(avg_rms.tolist())])
                peak_str = " ".join([f"{labels[i]}:{v:.4f}" for i, v in enumerate(peak_rms.tolist())])
                # Dominance helps mapping: if max_peak is only slightly larger than the others, the
                # sound source is not isolated enough (all mics hear it similarly).
                order = np.argsort(peak_rms)[::-1]
                best_i = int(order[0])
                second_i = int(order[1]) if peak_rms.shape[0] > 1 else int(order[0])
                best_v = float(peak_rms[best_i])
                second_v = float(peak_rms[second_i]) if second_i != best_i else 1e-12
                peak_ratio = best_v / (second_v + 1e-12)
                top2 = f"{labels[best_i]}/{labels[second_i]}"

                log.info("RMS avg  %s  (max_avg=%s)", avg_str, max_avg_label)
                log.info("RMS peak %s  (max_peak=%s top2=%s ratio=%.2f)", peak_str, max_peak_label, top2, peak_ratio)
    finally:
        await audio_in.stop()


def main() -> None:
    parser = argparse.ArgumentParser(prog="furhat-asd-audio-levels")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--interval-ms", type=int, default=250, help="Print interval in milliseconds (default: 250)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    try:
        asyncio.run(run_audio_levels(cfg, interval_ms=args.interval_ms))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
