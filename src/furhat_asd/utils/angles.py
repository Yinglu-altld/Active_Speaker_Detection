from __future__ import annotations

import math


def wrap_deg(deg: float) -> float:
    deg = deg % 360.0
    return deg + 360.0 if deg < 0 else deg


def smallest_angle_diff_deg(a: float, b: float) -> float:
    """
    Returns signed smallest difference from b -> a in degrees, in [-180, 180).
    """
    a = wrap_deg(a)
    b = wrap_deg(b)
    diff = (a - b + 180.0) % 360.0 - 180.0
    return diff


def circular_mean_deg(values: list[float]) -> float | None:
    if not values:
        return None
    sum_sin = 0.0
    sum_cos = 0.0
    for v in values:
        r = math.radians(v)
        sum_sin += math.sin(r)
        sum_cos += math.cos(r)
    if sum_sin == 0.0 and sum_cos == 0.0:
        return None
    return wrap_deg(math.degrees(math.atan2(sum_sin, sum_cos)))


def circular_mean_weighted_deg(values: list[float], weights: list[float]) -> float | None:
    """
    Weighted circular mean in degrees.

    Args:
      values: angles in degrees
      weights: non-negative weights (same length as values)
    """
    if not values or not weights or len(values) != len(weights):
        return None
    sum_w = 0.0
    sum_sin = 0.0
    sum_cos = 0.0
    for v, w in zip(values, weights, strict=True):
        w = float(w)
        if w <= 0:
            continue
        r = math.radians(v)
        sum_w += w
        sum_sin += w * math.sin(r)
        sum_cos += w * math.cos(r)
    if sum_w <= 0.0 or (sum_sin == 0.0 and sum_cos == 0.0):
        return None
    return wrap_deg(math.degrees(math.atan2(sum_sin, sum_cos)))


def circular_mode_weighted_deg(values: list[float], weights: list[float], *, bin_deg: float) -> float | None:
    """
    "Mode-like" angle estimate for noisy / jittery sequences.

    Implementation:
    - Bucket angles into bins of width `bin_deg` over [0, 360)
    - Choose the bin with the highest total weight
    - Return the weighted circular mean of angles that fell into that bin

    This is more robust than a plain mean when the distribution is multi-modal
    (e.g., DOA estimator flickers between a few directions).
    """
    if not values or not weights or len(values) != len(weights):
        return None
    bin_deg_f = float(bin_deg)
    if not (bin_deg_f > 0.0):
        return None

    n_bins = int(round(360.0 / bin_deg_f))
    n_bins = max(1, min(360, n_bins))

    bin_w = [0.0] * n_bins
    bin_sin = [0.0] * n_bins
    bin_cos = [0.0] * n_bins

    for v, w in zip(values, weights, strict=True):
        w = float(w)
        if w <= 0.0:
            continue
        v = wrap_deg(float(v))
        idx = int(v // bin_deg_f) % n_bins
        r = math.radians(v)
        bin_w[idx] += w
        bin_sin[idx] += w * math.sin(r)
        bin_cos[idx] += w * math.cos(r)

    best_idx = max(range(n_bins), key=lambda i: bin_w[i])
    if bin_w[best_idx] <= 0.0:
        return None
    if bin_sin[best_idx] == 0.0 and bin_cos[best_idx] == 0.0:
        return None
    return wrap_deg(math.degrees(math.atan2(bin_sin[best_idx], bin_cos[best_idx])))


def circular_dispersion_deg(values: list[float]) -> float | None:
    """
    Approx dispersion proxy: maps resultant vector length R in [0,1] to degrees.
    Larger means less stable. Returns None if undefined.
    """
    if not values:
        return None
    sum_sin = 0.0
    sum_cos = 0.0
    for v in values:
        r = math.radians(v)
        sum_sin += math.sin(r)
        sum_cos += math.cos(r)
    n = float(len(values))
    r_len = math.sqrt(sum_sin * sum_sin + sum_cos * sum_cos) / n
    r_len = max(0.0, min(1.0, r_len))
    return math.degrees(math.acos(r_len)) if r_len > 0 else 180.0


def circular_dispersion_weighted_deg(values: list[float], weights: list[float]) -> float | None:
    """
    Weighted dispersion proxy: maps weighted resultant vector length R in [0,1] to degrees.
    Larger means less stable. Returns None if undefined.
    """
    if not values or not weights or len(values) != len(weights):
        return None
    sum_w = 0.0
    sum_sin = 0.0
    sum_cos = 0.0
    for v, w in zip(values, weights, strict=True):
        w = float(w)
        if w <= 0:
            continue
        r = math.radians(v)
        sum_w += w
        sum_sin += w * math.sin(r)
        sum_cos += w * math.cos(r)
    if sum_w <= 0:
        return None
    r_len = math.sqrt(sum_sin * sum_sin + sum_cos * sum_cos) / sum_w
    r_len = max(0.0, min(1.0, r_len))
    return math.degrees(math.acos(r_len)) if r_len > 0 else 180.0
