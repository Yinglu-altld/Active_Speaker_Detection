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
    max_sigma_deg: float = 45.0
    weak_doa_weight: float = 0.10
    mid_doa_weight: float = 0.25
    strong_doa_weight: float = 0.40
    low_conf_th: float = 0.03
    mid_conf_th: float = 0.07
    low_srp_th: float = 0.08
    low_audio_th: float = 0.25
    cnn_ambiguous_margin: float = 0.12
    ambiguous_doa_boost: float = 0.12
    cnn_dominant_prob_th: float = 0.80
    cnn_dominant_margin: float = 0.20
    dominance_doa_suppression: float = 0.70
    # If users are visible but visually silent (max CNN below this) and audio speech is inactive,
    # do not output any active speaker (mode 4).
    min_cnn_speech_th: float = 0.30
    allow_single_user_doa: bool = True
    single_user_doa_weight_cap: float = 0.20
    single_user_min_conf: float = 0.03
    single_user_max_delta_deg: float = 35.0
    # Adaptive real-time reliability: reduce DOA influence when it disagrees with strong visual evidence.
    doa_disagreement_penalty: float = 0.85
    # Minimum DOA confidence to enable CNN+DOA fusion in-frame.
    min_conf_doa_srp_for_fusion: float = 0.10


def doa_weight(
    conf_doa_used: float,
    cfg: FusionConfig,
) -> float:
    # Raw mode: use DOA confidence directly as fusion weight.
    return _clip01(conf_doa_used)


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
    if sigma_deg is None:
        sigma = float(cfg.default_sigma_deg)
    else:
        sigma = max(cfg.min_sigma_deg, min(cfg.max_sigma_deg, float(sigma_deg)))
    delta = _angle_diff_deg(float(azimuth_deg), float(user_bearing_deg))
    align = math.exp(-0.5 * (delta / sigma) ** 2)
    return float(align), delta


def score_users_for_frame(
    doa_obs: Mapping[str, object],
    users: Iterable[UserEvidence],
    cfg: FusionConfig = FusionConfig(),
) -> dict:
    users = list(users)
    azimuth_deg = doa_obs.get("azimuth_deg")
    sigma_deg = doa_obs.get("sigma_deg")
    conf_doa = _clip01(float(doa_obs.get("conf_doa") or 0.0))
    conf_doa_srp = _clip01(float(doa_obs.get("conf_doa_srp") or 0.0))
    audio_conf = _clip01(float(doa_obs.get("audio_conf") or 0.0))
    speech_active = bool(doa_obs.get("speech_active") or doa_obs.get("speech_detected") or False)
    # Use SRP confidence for fusion weighting; do not downscale by VAD/audio confidence.
    conf_doa_used = conf_doa_srp
    has_users = len(users) > 0
    max_cnn_score = max((_cnn_score(u) for u in users), default=0.0)
    visual_speech = bool(max_cnn_score >= float(cfg.min_cnn_speech_th))

    # 4-mode policy:
    # 1) Users in frame but speech not good enough -> CNN only
    # 2) Users out of camera -> DOA only (no per-user speaker_id)
    # 3) Users in frame and speech good -> CNN+DOA
    # 4) Users in frame and no one talks -> no active speaker (speaker_id=None)
    if has_users and (not speech_active) and (not visual_speech):
        # Mode 4: users in-frame but nobody is talking (visually + audio).
        mode = "silent_in_frame"
        base_w_doa = 0.0
    elif has_users and (not speech_active) and visual_speech:
        # Mode 1: users in-frame, CNN indicates speech but audio isn't good enough -> CNN only.
        mode = "cnn_only_low_speech"
        base_w_doa = 0.0
    elif has_users and (azimuth_deg is not None) and (conf_doa_srp >= float(cfg.min_conf_doa_srp_for_fusion)):
        mode = "cnn_doa_in_frame"
        base_w_doa = doa_weight(conf_doa_used, cfg)
    elif has_users:
        mode = "cnn_only_low_speech"
        base_w_doa = 0.0
    else:
        # No users visible.
        if speech_active and azimuth_deg is not None:
            mode = "doa_only_no_users"
        else:
            mode = "idle_no_users"
        base_w_doa = doa_weight(conf_doa_used, cfg)
        if azimuth_deg is None:
            base_w_doa = 0.0

    per_user = []
    for user in users:
        per_user.append(
            {
                "user_id": user.user_id,
                "score_cnn": float(_cnn_score(user)),
                "score_doa": 0.0,
                "delta_deg": 180.0,
                "bearing_deg": float(user.bearing_deg),
            }
        )

    for item in per_user:
        s_doa, delta_deg = _doa_score(
            azimuth_deg=float(azimuth_deg) if azimuth_deg is not None else None,
            sigma_deg=float(sigma_deg) if sigma_deg is not None else None,
            user_bearing_deg=float(item["bearing_deg"]),
            cfg=cfg,
        )
        item["score_doa"] = float(s_doa)
        item["delta_deg"] = float(delta_deg)

    # Adaptive DOA reliability from real-time agreement with visual evidence.
    w_doa = float(base_w_doa)
    if per_user and w_doa > 0.0:
        cnn_sorted = sorted(float(item["score_cnn"]) for item in per_user)
        top_cnn = cnn_sorted[-1]
        second_cnn = cnn_sorted[-2] if len(cnn_sorted) > 1 else 0.0
        margin = max(0.0, top_cnn - second_cnn)
        top_visual = max(per_user, key=lambda it: float(it["score_cnn"]))
        top_delta = float(top_visual["delta_deg"])
        sigma_ref = float(cfg.default_sigma_deg if sigma_deg is None else sigma_deg)
        sigma_ref = max(12.0, min(float(cfg.max_sigma_deg), sigma_ref))
        agree = math.exp(-0.5 * (top_delta / sigma_ref) ** 2)
        visual_certainty = _clip01(math.sqrt(max(0.0, top_cnn)) * max(0.0, margin))
        disagreement = 1.0 - float(agree)
        penalty = _clip01(float(cfg.doa_disagreement_penalty) * visual_certainty * disagreement)
        w_doa = _clip01(w_doa * (1.0 - penalty))

    w_cnn = 1.0 - w_doa
    for item in per_user:
        item["score"] = float(w_cnn * float(item["score_cnn"]) + w_doa * float(item["score_doa"]))

    per_user.sort(key=lambda item: item["score"], reverse=True)
    top = per_user[0] if per_user else None
    speaker_id = None if top is None else top["user_id"]
    speaker_score = None if top is None else top["score"]
    if mode == "silent_in_frame":
        speaker_id = None
        speaker_score = None
    if mode in ("doa_only_no_users", "idle_no_users"):
        speaker_id = None
        speaker_score = None
    return {
        "t": doa_obs.get("t"),
        "mode": mode,
        "speech_active": bool(speech_active),
        "speaker_id": speaker_id,
        "speaker_score": speaker_score,
        "weights": {"cnn": float(w_cnn), "doa": float(w_doa)},
        "doa": {
            "azimuth_deg": azimuth_deg,
            "azimuth_deg_raw": doa_obs.get("azimuth_deg_raw"),
            "azimuth_offset_deg": doa_obs.get("azimuth_offset_deg"),
            "azimuth_branch_shift_deg": doa_obs.get("azimuth_branch_shift_deg"),
            "conf_doa": conf_doa,
            "conf_doa_srp": conf_doa_srp,
            "conf_doa_used": conf_doa_used,
            "audio_conf": audio_conf,
            "sigma_deg": sigma_deg,
        },
        "per_user": per_user,
    }
