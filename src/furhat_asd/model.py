from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FurhatUser:
    user_id: str
    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class DoaEstimate:
    ts_ms: int
    doa_deg: float
    conf: float


@dataclass(frozen=True)
class VadFrame:
    ts_ms: int
    speech_prob: float


@dataclass(frozen=True)
class SpeechState:
    ts_ms: int
    speech_on: bool


@dataclass(frozen=True)
class ActiveSpeakerDecision:
    ts_ms: int
    active_user_id: str | None
    confidence: float
    doa_deg_camera: float | None

