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
    default_sigma_deg: float = 25.0
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
    sigma = cfg.default_sigma_deg if sigma_deg is None else max(cfg.min_sigma_deg, float(sigma_deg))
    delta = _angle_diff_deg(float(azimuth_deg), float(user_bearing_deg))
    align = math.exp(-0.5 * (delta / sigma) ** 2)
    return float(align), delta


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
    azimuth_deg = doa_obs.get("azimuth_deg")
    sigma_deg = doa_obs.get("sigma_deg")
    conf_doa = _clip01(float(doa_obs.get("conf_doa") or 0.0))
    conf_doa_srp = _clip01(float(doa_obs.get("conf_doa_srp") or 0.0))
    audio_conf = _clip01(float(doa_obs.get("audio_conf") or 0.0))
    reliability = _clip01(0.6 * conf_doa_srp + 0.4 * audio_conf)
    w_doa = _clip01(float(cfg.fixed_doa_weight))
    w_cnn = 1.0 - w_doa

    per_user = []
    for user in users:
        s_cnn = _cnn_score(user)
        s_doa, delta_deg = _doa_score(
            azimuth_deg=float(azimuth_deg) if azimuth_deg is not None else None,
            sigma_deg=float(sigma_deg) if sigma_deg is not None else None,
            user_bearing_deg=float(user.bearing_deg),
            cfg=cfg,
        )
        score = w_cnn * s_cnn + w_doa * s_doa
        per_user.append(
            {
                "user_id": user.user_id,
                "score": float(score),
                "score_cnn": float(s_cnn),
                "score_doa": float(s_doa),
                "delta_deg": float(delta_deg),
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
            "conf_doa": conf_doa,
            "conf_doa_srp": conf_doa_srp,
            "audio_conf": audio_conf,
            "sigma_deg": sigma_deg,
            "reliability": float(reliability),
        },
        "per_user": per_user,
    }
