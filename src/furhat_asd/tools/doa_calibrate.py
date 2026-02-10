from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import time

import numpy as np

from furhat_asd.audio.audio_stream import SoundDeviceAudioInput
from furhat_asd.audio.doa_srp_phat import SrpPhatConfig, SrpPhatDoa
from furhat_asd.config import AppConfig, load_config
from furhat_asd.utils.angles import circular_mean_deg, smallest_angle_diff_deg, wrap_deg


def _circular_mean_of_diffs_deg(diffs: list[float]) -> float:
    """
    Compute mean angle of differences (in degrees), wrapped to [0, 360).
    """
    mean = circular_mean_deg(diffs)
    return float(wrap_deg(mean if mean is not None else 0.0))


def _mean_abs_error_deg(pred: list[float], target: list[float]) -> float:
    errs = [abs(float(smallest_angle_diff_deg(p, t))) for p, t in zip(pred, target, strict=True)]
    return float(np.mean(errs)) if errs else 180.0


async def _capture(
    *,
    audio_in: SoundDeviceAudioInput,
    duration_s: float,
) -> np.ndarray:
    """
    Capture `duration_s` seconds of audio and return pcm (frames, dev_channels).
    """
    end = time.time() + max(0.1, float(duration_s))
    chunks: list[np.ndarray] = []
    while time.time() < end:
        chunk = await audio_in.read()
        chunks.append(chunk.pcm_f32)
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, audio_in.channels), dtype=np.float32)


def _estimate_mean_doa(
    *,
    doa: SrpPhatDoa,
    pcm: np.ndarray,
    frame_frames: int,
) -> float | None:
    """
    Estimate a robust mean DOA for a PCM segment by running SRP-PHAT on sliding frames.
    """
    if pcm.size == 0 or pcm.shape[0] < frame_frames:
        return None
    azs: list[float] = []
    for start in range(0, pcm.shape[0] - frame_frames + 1, frame_frames):
        frame = pcm[start : start + frame_frames]
        est = doa.estimate_azimuth_deg(frame)
        if est is None:
            continue
        az, conf = est
        if conf >= 0.05:
            azs.append(float(az))
    return circular_mean_deg(azs) if azs else None


async def run_doa_calibrate(cfg: AppConfig, *, duration_s: float) -> None:
    logging.basicConfig(level=getattr(logging, cfg.logging.level.upper(), logging.INFO))
    log = logging.getLogger("furhat_asd.doa_calibrate")

    if cfg.audio.mode != "local":
        raise RuntimeError("doa-calibrate requires audio.mode=local")
    if not cfg.doa.enabled:
        raise RuntimeError("doa-calibrate requires doa.enabled=true")
    if cfg.doa.mic_positions_m is None:
        raise RuntimeError("doa-calibrate requires doa.mic_positions_m")
    if cfg.audio.channel_indices is None or len(cfg.audio.channel_indices) != 4:
        raise RuntimeError("doa-calibrate requires audio.channel_indices with exactly 4 raw mic channels (e.g. [1,2,3,4])")

    mic_pos = np.asarray(cfg.doa.mic_positions_m, dtype=np.float64)
    if mic_pos.shape != (4, 3):
        raise RuntimeError("doa-calibrate currently supports 4 mics only (doa.mic_positions_m must be 4x3)")

    audio_in = SoundDeviceAudioInput(
        device=cfg.audio.device,
        sample_rate=cfg.audio.sample_rate,
        channels=cfg.audio.channels,
        block_ms=cfg.audio.block_ms,
    )
    await audio_in.start()
    fs = audio_in.sample_rate

    doa_frame_ms = max(50, int(cfg.doa.frame_ms))
    frame_frames = int(fs * (doa_frame_ms / 1000.0))

    # Canonical geometry order in cfg.doa.mic_positions_m is assumed to be:
    #   index 0: +x (right)
    #   index 1: +z (top)
    #   index 2: -x (left)
    #   index 3: -z (bottom)
    #
    # We ask you to place a strong sound source at those 4 sides. Then we brute-force
    # the mapping of device channels -> geometry indices by trying all 24 permutations.
    steps = [
        ("RIGHT (+x)", 90.0),   # In our DOA math, 90° corresponds to +x
        ("TOP (+z)", 0.0),      # 0° corresponds to +z
        ("LEFT (-x)", 270.0),   # 270° corresponds to -x
        ("BOTTOM (-z)", 180.0), # 180° corresponds to -z
    ]

    log.info(
        "DOA calibration starting (device=%s sr=%dHz dev_channels=%d raw=%s frame_ms=%d capture=%.1fs).",
        cfg.audio.device,
        fs,
        cfg.audio.channels,
        cfg.audio.channel_indices,
        doa_frame_ms,
        duration_s,
    )
    log.info("Use a LOUD phone noise (pink/white) held 0.5–1 cm above the board (don’t touch the PCB).")

    try:
        measurements: list[tuple[str, float, np.ndarray]] = []
        for name, expected_deg in steps:
            input(f"\nStep: place sound at {name}, then press Enter to record {duration_s:.1f}s… ")
            # Warm-up / flush previous queue samples.
            t0 = time.time()
            while time.time() - t0 < 0.2:
                _ = await audio_in.read()
            pcm_full = await _capture(audio_in=audio_in, duration_s=duration_s)
            pcm_raw = pcm_full[:, cfg.audio.channel_indices]
            measurements.append((name, expected_deg, pcm_raw))
            log.info("Captured %s (%d frames).", name, int(pcm_raw.shape[0]))

        best = None
        raw_labels = [f"devch{c}" for c in cfg.audio.channel_indices]

        for perm in itertools.permutations(range(4)):
            # perm maps geometry-index -> raw-channel-index
            # i.e., geometry mic i uses raw channel perm[i]
            mapped_devchs = [cfg.audio.channel_indices[i] for i in perm]
            doa = SrpPhatDoa(
                SrpPhatConfig(mic_positions_m=mic_pos, sample_rate=fs, search_step_deg=cfg.doa.search_step_deg, gcc_interp=cfg.doa.gcc_interp)
            )

            ests: list[float] = []
            exps: list[float] = []
            ok = True
            for _, exp_deg, pcm_raw in measurements:
                pcm_mapped = pcm_raw[:, perm]  # reorder columns
                az = _estimate_mean_doa(doa=doa, pcm=pcm_mapped, frame_frames=frame_frames)
                if az is None:
                    ok = False
                    break
                ests.append(float(az))
                exps.append(float(exp_deg))
            if not ok:
                continue

            # Allow a flip (mirror) because mic coordinate handedness may be inverted depending on channel order.
            for sign in (1.0, -1.0):
                # Best offset is circular-mean of (expected - sign*measured)
                diffs = [wrap_deg(e - (sign * m)) for e, m in zip(exps, ests, strict=True)]
                offset = _circular_mean_of_diffs_deg(diffs)
                pred = [wrap_deg(sign * m + offset) for m in ests]
                mae = _mean_abs_error_deg(pred, exps)
                score = mae

                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "perm": perm,
                        "mapped_devchs": mapped_devchs,
                        "sign": int(1 if sign >= 0 else -1),
                        "offset": float(offset),
                        "pred": pred,
                        "exp": exps,
                    }

        if best is None:
            raise RuntimeError("Calibration failed: could not get stable DOA estimates. Try louder/closer sound or longer duration.")

        perm = best["perm"]
        mapped_devchs = best["mapped_devchs"]
        sign = best["sign"]
        offset = best["offset"]

        # Explain mapping in plain terms
        mapping_str = ", ".join([f"geom{i}<=devch{mapped_devchs[i]}" for i in range(4)])
        log.info("Best mapping: %s", mapping_str)
        log.info("Best sign/offset for this setup: doa_sign=%d doa_offset_deg=%.1f (MAE=%.1f°)", sign, offset, float(best["score"]))
        log.info(
            "Recommended config edits:\n"
            '  "audio": { "channel_indices": %s },\n'
            '  "control": { "doa_sign": %d, "doa_offset_deg": %.1f }',
            mapped_devchs,
            sign,
            offset,
        )

        # Per-step output
        for (name, exp_deg, _), pred_deg in zip(measurements, best["pred"], strict=True):
            log.info("  %s expected=%.0f° predicted=%.1f°", name, float(exp_deg), float(pred_deg))

        # Also print a human-readable “which devch is which side”
        # because that’s what you actually want for debugging.
        side_names = ["RIGHT(+x)", "TOP(+z)", "LEFT(-x)", "BOTTOM(-z)"]
        for i, side in enumerate(side_names):
            log.info("Side %s uses %s", side, f"devch{mapped_devchs[i]}")

        # Helpful hint: if you later mount the array on Furhat with a different rotation,
        # you'll re-tune doa_offset_deg (but keep channel_indices).
        log.info("Note: keep this channel mapping; later you’ll re-tune doa_offset_deg after mounting on Furhat.")
    finally:
        await audio_in.stop()


def main() -> None:
    parser = argparse.ArgumentParser(prog="furhat-asd-doa-calibrate")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--duration-s", type=float, default=2.5, help="Seconds to record per direction (default: 2.5)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    try:
        asyncio.run(run_doa_calibrate(cfg, duration_s=args.duration_s))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()

