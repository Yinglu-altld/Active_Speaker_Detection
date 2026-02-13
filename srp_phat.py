import math
from typing import List, Optional, Sequence, Tuple

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

    def estimate(self, mics: np.ndarray) -> Optional[Tuple[float, float]]:
        if mics.ndim != 2 or mics.shape[1] < 2 or not self.pairs:
            return None

        scores = np.zeros(self.az_grid.shape[0], dtype=np.float64)
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

        best_idx = int(np.argmax(scores))
        doa_deg = float(self.az_grid[best_idx])

        best = float(scores[best_idx])
        if scores.shape[0] > 1:
            second = float(np.max(np.delete(scores, best_idx)))
        else:
            second = 0.0
        mean = float(np.mean(scores)) + 1e-12
        sharpness = max(0.0, min(1.0, (best - second) / (best + 1e-12)))
        contrast = max(0.0, min(1.0, (best - mean) / (best + 1e-12)))
        confidence = 0.5 * sharpness + 0.5 * contrast
        return doa_deg, confidence
