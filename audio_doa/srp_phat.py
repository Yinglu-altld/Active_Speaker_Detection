
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


C_SOUND = 343.0


def _next_pow2(n: int) -> int:
    return 1 << max(1, (n - 1).bit_length())


def _gcc_phat_curve(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    fs: int,
    max_tau_s: float,
    interp: int,
    f_low_hz: Optional[float] = None,
    f_high_hz: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n_fft = _next_pow2(sig_a.shape[0] + sig_b.shape[0])
    a = np.fft.rfft(sig_a, n=n_fft)
    b = np.fft.rfft(sig_b, n=n_fft)
    r = a * np.conj(b)
    if f_low_hz is not None or f_high_hz is not None:
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        f_low = 0.0 if f_low_hz is None else max(0.0, float(f_low_hz))
        f_high = (fs * 0.5) if f_high_hz is None else min(fs * 0.5, float(f_high_hz))
        if f_high > f_low:
            band = (freqs >= f_low) & (freqs <= f_high)
            if np.any(band):
                r *= band.astype(np.float64)
    r /= np.abs(r) + 1e-12
    cc = np.fft.irfft(r, n=n_fft * interp)
    max_shift = int(interp * fs * max_tau_s)
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    lags = np.arange(-max_shift, max_shift + 1, dtype=np.float64) / (interp * fs)
    return lags, np.abs(cc)


class SRPPhatDOA:
    """2D far-field SRP-PHAT DOA estimator.

    Angles are in degrees, +CCW from +X axis.
    """

    def __init__(
        self,
        mic_xy_m: Sequence[Tuple[float, float]],
        fs: int = 16000,
        az_min_deg: float = -180.0,
        az_max_deg: float = 180.0,
        az_step_deg: float = 2.0,
        interp: int = 4,
        f_low_hz: Optional[float] = 300.0,
        f_high_hz: Optional[float] = 3400.0,
    ):
        self.fs = fs
        self.interp = interp
        self.f_low_hz = f_low_hz
        self.f_high_hz = f_high_hz
        self.mic_xy = np.asarray(mic_xy_m, dtype=np.float64)
        self.az_grid = np.arange(az_min_deg, az_max_deg + 0.5 * az_step_deg, az_step_deg, dtype=np.float64)
        az_rad = np.deg2rad(self.az_grid)
        self.dir_xy = np.stack([np.cos(az_rad), np.sin(az_rad)], axis=1)

        self.pairs: List[Tuple[int, int, float, float, float]] = []
        n_mics = self.mic_xy.shape[0]
        for i in range(n_mics):
            for j in range(i + 1, n_mics):
                dx = float(self.mic_xy[i, 0] - self.mic_xy[j, 0])
                dy = float(self.mic_xy[i, 1] - self.mic_xy[j, 1])
                dist = math.hypot(dx, dy)
                if dist > 1e-6:
                    self.pairs.append((i, j, dx, dy, dist))

    def estimate(self, mics: np.ndarray) -> Optional["DOAResult"]:
        if mics.ndim != 2 or mics.shape[1] < 2 or not self.pairs:
            return None

        scores = np.zeros(self.az_grid.shape[0], dtype=np.float64)
        pair_best_az_deg: List[float] = []
        for i, j, dx, dy, dist in self.pairs:
            lags, cc_abs = _gcc_phat_curve(
                mics[:, i],
                mics[:, j],
                self.fs,
                dist / C_SOUND,
                self.interp,
                self.f_low_hz,
                self.f_high_hz,
            )
            tau_pred = (dx * self.dir_xy[:, 0] + dy * self.dir_xy[:, 1]) / C_SOUND
            pair_score = np.interp(tau_pred, lags, cc_abs, left=0.0, right=0.0)
            scores += pair_score
            pair_best_az_deg.append(float(self.az_grid[int(np.argmax(pair_score))]))

        best_idx = int(np.argmax(scores))
        doa_deg = float(self.az_grid[best_idx])

        best = float(scores[best_idx])
        if self.az_grid.size > 1:
            step_deg = float(self.az_grid[1] - self.az_grid[0])
        else:
            step_deg = 1.0
        second_idx, second = _second_peak_excluding_window(
            scores=scores,
            best_idx=best_idx,
            window_bins=max(1, int(round(12.0 / max(step_deg, 1e-6)))),
        )

        mean = float(np.mean(scores)) + 1e-12
        median = float(np.median(scores)) + 1e-12
        sharpness = max(0.0, min(1.0, (best - second) / (best + 1e-12)))
        contrast = max(0.0, min(1.0, (best - mean) / (best + 1e-12)))
        entropy = _entropy01(scores)
        sigma_deg = _half_max_sigma(scores, best_idx, step_deg)
        peak_ratio = best / (second + 1e-12)
        peak_ratio_score = max(
            0.0,
            min(1.0, 1.0 - math.exp(-max(0.0, peak_ratio - 1.0))),
        )
        entropy_score = 1.0 - entropy
        sigma_score = (
            0.0
            if sigma_deg is None
            else max(0.0, min(1.0, math.exp(-sigma_deg / 25.0)))
        )
        prominence = max(0.0, min(1.0, (best - median) / (best + 1e-12)))
        consensus_r, consensus_std_deg = _circular_consensus(pair_best_az_deg)
        consensus_score = max(0.0, min(1.0, consensus_r))
        peak_sep_deg = (
            0.0
            if second_idx is None
            else abs(_circular_deg_diff(doa_deg, float(self.az_grid[second_idx])))
        )
        second_ratio = second / (best + 1e-12)

        confidence = max(
            0.0,
            min(
                1.0,
                0.24 * peak_ratio_score
                + 0.16 * contrast
                + 0.14 * entropy_score
                + 0.14 * sigma_score
                + 0.20 * consensus_score
                + 0.12 * prominence,
            ),
        )
        if second_ratio > 0.85 and peak_sep_deg >= 120.0:
            confidence *= 0.80
        if consensus_score < 0.25:
            confidence *= 0.85

        conf_components = {
            "peak_ratio_score": float(peak_ratio_score),
            "contrast_score": float(contrast),
            "entropy_score": float(entropy_score),
            "sigma_score": float(sigma_score),
            "sharpness_score": float(sharpness),
            "prominence_score": float(prominence),
            "pair_consensus_score": float(consensus_score),
            "pair_consensus_std_deg": float(consensus_std_deg),
            "peak_separation_deg": float(peak_sep_deg),
            "peak_ratio_raw": float(peak_ratio),
            "second_ratio_raw": float(second_ratio),
        }
        peaks = _top_peaks(scores, self.az_grid, k=3)
        return DOAResult(
            doa_deg=doa_deg,
            conf=confidence,
            sigma_deg=sigma_deg,
            peaks=peaks,
            entropy=entropy,
            conf_components=conf_components,
        )


@dataclass(frozen=True)
class DOAResult:
    doa_deg: float
    conf: float
    sigma_deg: Optional[float]
    peaks: List[dict]
    entropy: float
    conf_components: Dict[str, float]


def _entropy01(scores: np.ndarray) -> float:
    total = float(np.sum(scores))
    if total <= 0.0 or scores.size == 0:
        return 1.0
    if scores.size == 1:
        return 0.0
    p = scores / total
    h = float(-np.sum(p * np.log(p + 1e-12)))
    return max(0.0, min(1.0, h / math.log(scores.size)))


def _half_max_sigma(scores: np.ndarray, best_idx: int, step_deg: float) -> Optional[float]:
    if scores.size == 0:
        return None
    best = float(scores[best_idx])
    if best <= 0.0:
        return None
    half = 0.5 * best
    left = best_idx
    while left > 0 and scores[left - 1] >= half:
        left -= 1
    right = best_idx
    while right < scores.size - 1 and scores[right + 1] >= half:
        right += 1
    width_bins = max(1, right - left + 1)
    width_deg = float(width_bins) * float(step_deg)
    return width_deg / 2.355


def _top_peaks(scores: np.ndarray, az_grid: np.ndarray, k: int = 3) -> List[dict]:
    if scores.size == 0:
        return []
    k = max(1, min(int(k), scores.size))
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    total = float(np.sum(scores)) + 1e-12
    best = float(np.max(scores)) + 1e-12
    peaks = []
    for i in idx:
        s = float(scores[int(i)])
        peaks.append(
            {
                "azimuth_deg": float(az_grid[int(i)]),
                "score": s,
                "score_norm": float(s / best),
                "score_prob": float(s / total),
            }
        )
    return peaks


def _second_peak_excluding_window(
    scores: np.ndarray,
    best_idx: int,
    window_bins: int,
) -> tuple[Optional[int], float]:
    if scores.size <= 1:
        return None, 0.0
    window_bins = max(0, int(window_bins))
    mask = np.ones(scores.shape[0], dtype=bool)
    lo = max(0, best_idx - window_bins)
    hi = min(scores.shape[0], best_idx + window_bins + 1)
    mask[lo:hi] = False
    if not np.any(mask):
        return None, 0.0
    masked_idx = np.where(mask)[0]
    rel = int(np.argmax(scores[mask]))
    idx = int(masked_idx[rel])
    return idx, float(scores[idx])


def _circular_deg_diff(a_deg: float, b_deg: float) -> float:
    return ((b_deg - a_deg + 180.0) % 360.0) - 180.0


def _circular_consensus(angles_deg: Sequence[float]) -> tuple[float, float]:
    if not angles_deg:
        return 0.0, 180.0
    ang = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    c = float(np.mean(np.cos(ang)))
    s = float(np.mean(np.sin(ang)))
    r = max(0.0, min(1.0, math.hypot(c, s)))
    if r < 1e-12:
        return 0.0, 180.0
    circ_std_rad = math.sqrt(max(0.0, -2.0 * math.log(r)))
    circ_std_deg = float(np.degrees(circ_std_rad))
    return r, circ_std_deg
