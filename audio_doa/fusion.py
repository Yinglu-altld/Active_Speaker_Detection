from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _wrap_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _angle_diff_deg(a_deg: float, b_deg: float) -> float:
    return abs(_wrap_deg(float(a_deg) - float(b_deg)))


@dataclass(frozen=True)
class UserEvidence:
    user_id: str
    bearing_deg: float
    cnn_prob: float
    face_conf: float = 1.0
    track_conf: float = 1.0


@dataclass(frozen=True)
class FusionConfig:
    min_sigma_deg: float = 8.0
    max_sigma_deg: float = 45.0
    default_sigma_deg: float = 25.0
    sigma_trust_decay_deg: float = 30.0
    min_align_gap: float = 0.20
    doa_delta_hard_gate_deg: float = 85.0
    min_reliability_for_doa: float = 0.05
    reliability_gamma: float = 0.65
    fixed_doa_weight: float = 0.35


def _cnn_score(user: UserEvidence) -> float:
    return _clip01(user.cnn_prob) * _clip01(user.face_conf) * _clip01(user.track_conf)


def _doa_score(
    azimuth_deg: float | None,
    sigma_deg: float | None,
    user_bearing_deg: float,
    cfg: FusionConfig,
) -> tuple[float, float]:
    if azimuth_deg is None:
        return 0.0, 180.0
    sigma_raw = cfg.default_sigma_deg if sigma_deg is None else float(sigma_deg)
    sigma_max = max(float(cfg.min_sigma_deg), float(cfg.max_sigma_deg))
    sigma = min(sigma_max, max(float(cfg.min_sigma_deg), sigma_raw))
    delta = _angle_diff_deg(float(azimuth_deg), float(user_bearing_deg))
    if delta > float(cfg.doa_delta_hard_gate_deg):
        return 0.0, float(delta)
    align = math.exp(-0.5 * (delta / sigma) ** 2)
    return float(align), delta


def _resolve_azimuth_from_faces(
    azimuth_deg: float | None,
    users: Iterable[UserEvidence],
) -> float | None:
    if azimuth_deg is None:
        return None
    az0 = _wrap_deg(float(azimuth_deg))
    az1 = _wrap_deg(az0 + 180.0)
    users = list(users)
    if not users:
        return az0
    # Pick the hemisphere whose direction is, on average, closer to visible face bearings.
    # This removes 180-degree flip ambiguity before per-user scoring.
    cost0 = 0.0
    cost1 = 0.0
    for user in users:
        w = max(0.05, _clip01(user.face_conf) * _clip01(user.track_conf))
        b = float(user.bearing_deg)
        cost0 += w * _angle_diff_deg(az0, b)
        cost1 += w * _angle_diff_deg(az1, b)
    return az0 if cost0 <= cost1 else az1


def _reliability_weight(reliability: float, cfg: FusionConfig) -> float:
    # Smooth mapping (no hard cut): reliability -> [0,1] DOA weight scale.
    r = _clip01(reliability)
    if r <= 0.0:
        return 0.0
    knee = max(1e-3, float(cfg.min_reliability_for_doa))
    gamma = max(0.25, float(cfg.reliability_gamma))
    base = r / (r + knee)
    return _clip01(base**gamma)


def score_users_cnn_only(
    users: Iterable[UserEvidence],
    t_value: float | None = None,
) -> dict:
    per_user = []
    for user in users:
        s_cnn = _cnn_score(user)
        per_user.append(
            {
                "user_id": user.user_id,
                "score": float(s_cnn),
                "score_cnn": float(s_cnn),
                "score_doa": 0.0,
                "delta_deg": 180.0,
                "bearing_deg": float(user.bearing_deg),
            }
        )
    per_user.sort(key=lambda item: item["score"], reverse=True)
    top = per_user[0] if per_user else None
    return {
        "t": t_value,
        "speaker_id": None if top is None else top["user_id"],
        "speaker_score": None if top is None else top["score"],
        "weights": {"cnn": 1.0, "doa": 0.0},
        "doa": {
            "azimuth_deg": None,
            "conf_doa": 0.0,
            "conf_doa_srp": 0.0,
            "audio_conf": 0.0,
            "sigma_deg": None,
            "reliability": 0.0,
        },
        "per_user": per_user,
    }


def score_users_for_frame(
    doa_obs: Mapping[str, object],
    users: Iterable[UserEvidence],
    cfg: FusionConfig = FusionConfig(),
) -> dict:
    users = list(users)
    azimuth_raw = doa_obs.get("azimuth_deg")
    azimuth_deg = _resolve_azimuth_from_faces(
        azimuth_deg=None if azimuth_raw is None else float(azimuth_raw),
        users=users,
    )
    sigma_deg = doa_obs.get("sigma_deg")
    conf_doa = _clip01(float(doa_obs.get("conf_doa") or 0.0))
    conf_doa_srp = _clip01(float(doa_obs.get("conf_doa_srp") or 0.0))
    audio_conf = _clip01(float(doa_obs.get("audio_conf") or 0.0))
    reliability = _clip01(0.6 * conf_doa_srp + 0.4 * audio_conf)
    reliability_eff = _reliability_weight(reliability, cfg)
    doa_available = azimuth_deg is not None
    sigma_f = None if sigma_deg is None else float(sigma_deg)

    doa_enabled = doa_available
    # Continuous DOA weight: fades with reliability instead of hard on/off gating.
    w_doa = _clip01(float(cfg.fixed_doa_weight) * float(reliability_eff)) if doa_enabled else 0.0
    w_cnn = 1.0 - w_doa

    sigma_used = sigma_f
    raw_rows = []
    for user in users:
        s_cnn = _cnn_score(user)
        if doa_enabled:
            align_raw, delta_deg = _doa_score(
                azimuth_deg=float(azimuth_deg) if azimuth_deg is not None else None,
                sigma_deg=sigma_used,
                user_bearing_deg=float(user.bearing_deg),
                cfg=cfg,
            )
        else:
            align_raw, delta_deg = 0.0, 180.0
        raw_rows.append(
            {
                "user": user,
                "score_cnn": float(s_cnn),
                "align_raw": float(align_raw),
                "delta_deg": float(delta_deg),
            }
        )

    if doa_enabled and raw_rows:
        aligns = sorted((float(row["align_raw"]) for row in raw_rows), reverse=True)
        top_align = aligns[0]
        second_align = aligns[1] if len(aligns) > 1 else 0.0
        if len(aligns) > 1:
            gap = max(0.0, top_align - second_align)
            separability = _clip01(gap / max(1e-6, float(cfg.min_align_gap)))
        else:
            separability = 1.0
        # Keep sigma influence only inside align_raw (Gaussian delta/sigma term).
        sigma_quality = 1.0
        doa_trust = _clip01(float(separability))
    else:
        top_align = 0.0
        second_align = 0.0
        separability = 0.0
        sigma_quality = 0.0
        doa_trust = 0.0

    per_user = []
    for row in raw_rows:
        user = row["user"]
        s_cnn = float(row["score_cnn"])
        s_doa = float(row["align_raw"]) * float(doa_trust)
        if w_doa <= 0.0:
            s_doa = 0.0
        mix = w_cnn * s_cnn + w_doa * s_doa
        score = mix
        part_cnn = w_cnn * s_cnn
        part_doa = w_doa * s_doa
        floor_applied = False
        per_user.append(
            {
                "user_id": user.user_id,
                "score": float(score),
                "score_cnn": float(s_cnn),
                "score_doa_raw": float(row["align_raw"]),
                "score_doa": float(s_doa),
                "part_cnn": float(part_cnn),
                "part_doa": float(part_doa),
                "mix_score": float(mix),
                "doa_used": bool(part_doa > 1e-6),
                "floor_applied": bool(floor_applied),
                "delta_deg": float(row["delta_deg"]),
                "bearing_deg": float(user.bearing_deg),
            }
        )

    per_user.sort(key=lambda item: item["score"], reverse=True)
    top = per_user[0] if per_user else None
    return {
        "t": doa_obs.get("t"),
        "speaker_id": None if top is None else top["user_id"],
        "speaker_score": None if top is None else top["score"],
        "weights": {"cnn": float(w_cnn), "doa": float(w_doa)},
        "doa": {
            "azimuth_deg": azimuth_deg,
            "azimuth_raw_deg": azimuth_raw,
            "conf_doa": conf_doa,
            "conf_doa_srp": conf_doa_srp,
            "audio_conf": audio_conf,
            "sigma_deg": sigma_deg,
            "reliability": float(reliability),
            "reliability_effective": float(reliability_eff),
            "doa_trust": float(doa_trust),
            "doa_separability": float(separability),
            "doa_sigma_quality": float(sigma_quality),
            "doa_top_align": float(top_align),
            "doa_second_align": float(second_align),
            "valid_for_fusion": bool(doa_enabled and w_doa > 0.0),
        },
        "per_user": per_user,
    }
