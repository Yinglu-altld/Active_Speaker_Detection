from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VadGateConfig:
    speech_on_conf: float
    speech_on_ms: int
    speech_off_conf: float
    speech_off_ms: int
    min_burst_ms: int


class VadGate:
    """
    Converts per-frame speech probabilities into a stable speech_on/speech_off state
    using hysteresis and minimum durations to reduce false triggers.
    """

    def __init__(self, cfg: VadGateConfig) -> None:
        self._cfg = cfg
        self._speech_on = False
        self._above_since_ms: int | None = None
        self._below_since_ms: int | None = None
        self._last_on_ms: int | None = None

    @property
    def speech_on(self) -> bool:
        return self._speech_on

    def update(self, ts_ms: int, speech_prob: float) -> bool:
        """
        Update state, return True if state changed.
        """
        if self._speech_on:
            if speech_prob <= self._cfg.speech_off_conf:
                if self._below_since_ms is None:
                    self._below_since_ms = ts_ms
                if ts_ms - self._below_since_ms >= self._cfg.speech_off_ms:
                    # Ignore ultra-short bursts.
                    if self._last_on_ms is not None and ts_ms - self._last_on_ms < self._cfg.min_burst_ms:
                        self._below_since_ms = None
                        return False
                    self._speech_on = False
                    self._below_since_ms = None
                    self._above_since_ms = None
                    return True
            else:
                self._below_since_ms = None
            return False

        # Speech currently off.
        if speech_prob >= self._cfg.speech_on_conf:
            if self._above_since_ms is None:
                self._above_since_ms = ts_ms
            if ts_ms - self._above_since_ms >= self._cfg.speech_on_ms:
                self._speech_on = True
                self._last_on_ms = ts_ms
                self._above_since_ms = None
                self._below_since_ms = None
                return True
        else:
            self._above_since_ms = None
        return False

