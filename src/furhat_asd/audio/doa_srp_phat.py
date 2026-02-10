from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from furhat_asd.utils.angles import wrap_deg


SPEED_OF_SOUND_M_S = 343.0


@dataclass(frozen=True)
class SrpPhatConfig:
    mic_positions_m: np.ndarray  # (M, 3)
    sample_rate: int
    search_step_deg: int = 5
    gcc_interp: int = 1


@dataclass(frozen=True)
class DoaEstimate:
    azimuth_deg: float
    conf: float
    peak_ratio: float
    second_azimuth_deg: float | None


def _gcc_phat(sig: np.ndarray, refsig: np.ndarray, sample_rate: int, max_tau: float, interp: int = 1) -> float:
    """
    Returns time delay (seconds) between sig and refsig via GCC-PHAT.
    """
    interp = max(1, int(interp))
    n = interp * (sig.shape[0] + refsig.shape[0])
    sig_fft = np.fft.rfft(sig, n=n)
    ref_fft = np.fft.rfft(refsig, n=n)
    r = sig_fft * np.conj(ref_fft)
    denom = np.abs(r)
    denom[denom < 1e-12] = 1e-12
    r /= denom
    cc = np.fft.irfft(r, n=n)
    max_shift = int(min(int(interp * sample_rate * max_tau), n // 2))
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc)) - max_shift)
    return float(shift) / float(interp * sample_rate)


def _gcc_phat_cc(
    sig: np.ndarray, refsig: np.ndarray, sample_rate: int, max_tau: float, interp: int = 1
) -> tuple[np.ndarray, int, int]:
    """
    Returns GCC-PHAT cross-correlation around [-max_shift, +max_shift] and max_shift.
    """
    interp = max(1, int(interp))
    n = interp * (sig.shape[0] + refsig.shape[0])
    sig_fft = np.fft.rfft(sig, n=n)
    ref_fft = np.fft.rfft(refsig, n=n)
    r = sig_fft * np.conj(ref_fft)
    denom = np.abs(r)
    denom[denom < 1e-12] = 1e-12
    r /= denom
    cc = np.fft.irfft(r, n=n)
    max_shift = int(min(int(interp * sample_rate * max_tau), n // 2))
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    return cc, max_shift, interp

class SrpPhatDoa:
    """
    SRP-PHAT over a 2D azimuth grid (0..360). Requires microphone positions.
    """

    def __init__(self, cfg: SrpPhatConfig) -> None:
        self._mic_pos = cfg.mic_positions_m.astype(np.float64)
        self._fs = int(cfg.sample_rate)
        self._step = int(cfg.search_step_deg)
        self._interp = max(1, int(cfg.gcc_interp))
        if self._mic_pos.ndim != 2 or self._mic_pos.shape[0] < 2 or self._mic_pos.shape[1] != 3:
            raise ValueError("mic_positions_m must be (M,3) with M>=2")
        self._pairs = [(i, j) for i in range(self._mic_pos.shape[0]) for j in range(i + 1, self._mic_pos.shape[0])]

    @staticmethod
    def _clamp01(x: float) -> float:
        return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)

    def estimate(self, pcm_f32: np.ndarray) -> DoaEstimate | None:
        """
        Returns a richer DOA estimate including a peak ratio heuristic.

        `conf` is a conservative [0,1] score intended for gating, combining:
          - how much the best angle stands out vs the average score, and
          - how strongly the best peak dominates the second-best peak.
        """
        if pcm_f32.ndim != 2 or pcm_f32.shape[1] != self._mic_pos.shape[0]:
            return None
        frames = int(pcm_f32.shape[0])
        if frames < int(0.01 * self._fs):
            return None

        pair_cc: dict[tuple[int, int], tuple[np.ndarray, int, int]] = {}
        for i, j in self._pairs:
            dist = float(np.linalg.norm(self._mic_pos[i] - self._mic_pos[j]))
            max_tau = dist / SPEED_OF_SOUND_M_S
            pair_cc[(i, j)] = _gcc_phat_cc(pcm_f32[:, i], pcm_f32[:, j], self._fs, max_tau=max_tau, interp=self._interp)

        best_deg = 0.0
        best_score = -1.0
        second_deg: float | None = None
        second_score = -1.0
        scores: list[float] = []

        for deg in range(0, 360, self._step):
            theta = math.radians(float(deg))
            direction = np.array([math.sin(theta), 0.0, math.cos(theta)], dtype=np.float64)
            taus = (self._mic_pos @ direction) / SPEED_OF_SOUND_M_S
            score = 0.0
            for i, j in self._pairs:
                pred = float(taus[i] - taus[j])
                cc, max_shift, interp = pair_cc[(i, j)]
                shift = int(round(pred * self._fs * interp))
                if shift < -max_shift:
                    shift = -max_shift
                elif shift > max_shift:
                    shift = max_shift
                score += float(abs(cc[shift + max_shift]))
            scores.append(score)

            if score > best_score:
                second_score = best_score
                second_deg = best_deg if best_score >= 0 else None
                best_score = score
                best_deg = float(deg)
            elif score > second_score:
                second_score = score
                second_deg = float(deg)

        if not scores or best_score <= 0:
            return None

        mean_score = float(np.mean(scores))
        # Peak-vs-average confidence (0..1-ish).
        conf_mean = 0.0 if mean_score <= 1e-9 else float(min(1.0, (best_score - mean_score) / (best_score + 1e-9)))

        # Peak ratio confidence: if best and second-best are similar, we treat DOA as ambiguous.
        if second_score <= 0:
            peak_ratio = float("inf")
            conf_ratio = 1.0
        else:
            peak_ratio = float(best_score / (second_score + 1e-12))
            # Map ratio to [0,1]. Ratio 1.0 => 0.0, ratio 1.5 => ~1.0.
            conf_ratio = self._clamp01((peak_ratio - 1.0) / 0.5)

        conf = float(self._clamp01(conf_mean * conf_ratio))
        return DoaEstimate(
            azimuth_deg=wrap_deg(best_deg),
            conf=conf,
            peak_ratio=float(peak_ratio if peak_ratio != float("inf") else 999.0),
            second_azimuth_deg=wrap_deg(float(second_deg)) if second_deg is not None else None,
        )

    def estimate_azimuth_deg(self, pcm_f32: np.ndarray) -> tuple[float, float] | None:
        """
        Args:
          pcm_f32: (frames, channels)
        Returns:
          (azimuth_deg, confidence) or None if insufficient data.
        """
        est = self.estimate(pcm_f32)
        if est is None:
            return None
        return float(est.azimuth_deg), float(est.conf)
