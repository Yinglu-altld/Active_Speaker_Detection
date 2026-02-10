from __future__ import annotations

from dataclasses import dataclass

from furhat_asd.model import FurhatUser


@dataclass(frozen=True)
class AddressingEstimate:
    """
    `prob` is the likelihood the active speaker is addressing Furhat (camera/robot).
    """

    ts_ms: int
    prob: float


class AddressingEstimator:
    def estimate(self, ts_ms: int, jpeg: bytes | None, active_user: FurhatUser | None) -> AddressingEstimate:
        raise NotImplementedError


class NoopAddressingEstimator(AddressingEstimator):
    """
    Placeholder estimator until vision modules (head pose + iris) are wired in.
    """

    def estimate(self, ts_ms: int, jpeg: bytes | None, active_user: FurhatUser | None) -> AddressingEstimate:
        return AddressingEstimate(ts_ms=ts_ms, prob=1.0)


class FurhatUserPoseAddressingEstimator(AddressingEstimator):
    """
    Uses pose/rotation fields from Furhat `users` payload if available.

    This is a pragmatic MVP path because it requires no heavy vision dependencies.
    If the payload doesn't include rotation, it falls back to 0.5 ("unknown").
    """

    def __init__(self, max_yaw_deg: float = 20.0, max_pitch_deg: float = 20.0) -> None:
        self._max_yaw = float(max_yaw_deg)
        self._max_pitch = float(max_pitch_deg)

    def estimate(self, ts_ms: int, jpeg: bytes | None, active_user: FurhatUser | None) -> AddressingEstimate:
        if active_user is None or not active_user.raw:
            return AddressingEstimate(ts_ms=ts_ms, prob=0.5)
        rot = active_user.raw.get("rot") or active_user.raw.get("rotation")
        if not isinstance(rot, dict):
            return AddressingEstimate(ts_ms=ts_ms, prob=0.5)
        yaw = rot.get("y", rot.get("yaw", None))
        pitch = rot.get("x", rot.get("pitch", None))
        try:
            yaw = float(yaw)
            pitch = float(pitch)
        except Exception:
            return AddressingEstimate(ts_ms=ts_ms, prob=0.5)
        aligned = (abs(yaw) <= self._max_yaw) and (abs(pitch) <= self._max_pitch)
        return AddressingEstimate(ts_ms=ts_ms, prob=0.9 if aligned else 0.1)
