from __future__ import annotations

import math
from dataclasses import dataclass

from furhat_asd.model import FurhatUser
from furhat_asd.utils.angles import smallest_angle_diff_deg, wrap_deg


def user_azimuth_deg(user: FurhatUser) -> float | None:
    """
    Furhat coordinates: x to robot's left, z to the front.
    Azimuth: 0 = front, +90 = left, right becomes 270.
    """
    if user.x_m is None or user.z_m is None:
        return None
    az = math.degrees(math.atan2(user.x_m, user.z_m))
    return wrap_deg(az)


@dataclass(frozen=True)
class UserSelectorConfig:
    doa_sigma_deg: float
    doa_offset_deg: float
    doa_sign: int
    switch_margin_ratio: float
    switch_hold_ms: int


class UserSelector:
    def __init__(self, cfg: UserSelectorConfig) -> None:
        self._cfg = cfg
        self._active_user_id: str | None = None
        self._candidate_user_id: str | None = None
        self._candidate_since_ms: int | None = None

    @property
    def active_user_id(self) -> str | None:
        return self._active_user_id

    def _doa_to_camera(self, doa_deg: float) -> float:
        return wrap_deg(self._cfg.doa_sign * doa_deg + self._cfg.doa_offset_deg)

    def choose(self, ts_ms: int, doa_deg: float, users: list[FurhatUser]) -> tuple[str | None, float, float]:
        """
        Returns (active_user_id, confidence, doa_deg_camera).
        """
        doa_cam = self._doa_to_camera(doa_deg)
        scored: list[tuple[str, float]] = []
        for u in users:
            az = user_azimuth_deg(u)
            if az is None:
                continue
            diff = abs(smallest_angle_diff_deg(doa_cam, az))
            sigma = max(1e-3, self._cfg.doa_sigma_deg)
            score = math.exp(-(diff * diff) / (2.0 * sigma * sigma))
            scored.append((u.user_id, score))
        if not scored:
            self._active_user_id = None
            self._candidate_user_id = None
            self._candidate_since_ms = None
            return None, 0.0, doa_cam

        scored.sort(key=lambda x: x[1], reverse=True)
        best_id, best_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0
        margin_ratio = (best_score / (second_score + 1e-9)) if second_score > 0 else float("inf")
        confidence = float(min(1.0, best_score * (1.0 if margin_ratio == float("inf") else (margin_ratio - 1.0))))

        if self._active_user_id is None:
            self._active_user_id = best_id
            self._candidate_user_id = None
            self._candidate_since_ms = None
            return self._active_user_id, confidence, doa_cam

        if best_id == self._active_user_id:
            self._candidate_user_id = None
            self._candidate_since_ms = None
            return self._active_user_id, confidence, doa_cam

        # Consider switching.
        if margin_ratio < self._cfg.switch_margin_ratio:
            self._candidate_user_id = None
            self._candidate_since_ms = None
            return self._active_user_id, confidence, doa_cam

        if self._candidate_user_id != best_id:
            self._candidate_user_id = best_id
            self._candidate_since_ms = ts_ms
            return self._active_user_id, confidence, doa_cam

        assert self._candidate_since_ms is not None
        if ts_ms - self._candidate_since_ms >= self._cfg.switch_hold_ms:
            self._active_user_id = best_id
            self._candidate_user_id = None
            self._candidate_since_ms = None
        return self._active_user_id, confidence, doa_cam

