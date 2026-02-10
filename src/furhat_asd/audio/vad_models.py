from __future__ import annotations

import math
import inspect
import importlib
from dataclasses import dataclass

import numpy as np


class VadBackend:
    def speech_prob(self, mono_pcm_f32: np.ndarray) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class EnergyVadConfig:
    threshold: float


class EnergyVad(VadBackend):
    """
    Lightweight fallback VAD that estimates speech probability from RMS energy.
    This is not robust in noise; it's meant to keep the pipeline runnable without
    heavyweight model dependencies.
    """

    def __init__(self, cfg: EnergyVadConfig) -> None:
        self._thr = float(cfg.threshold)

    def speech_prob(self, mono_pcm_f32: np.ndarray) -> float:
        if mono_pcm_f32.size == 0:
            return 0.0
        rms = float(math.sqrt(float(np.mean(np.square(mono_pcm_f32)))))
        if rms <= 0:
            return 0.0
        x = (rms - self._thr) / (self._thr * 0.5 + 1e-6)
        return float(1.0 / (1.0 + math.exp(-x)))


@dataclass(frozen=True)
class WebRtcVadConfig:
    aggressiveness: int = 2  # 0..3
    sample_rate: int = 16000


class WebRtcVad(VadBackend):
    """
    Wrapper around `webrtcvad` that returns a pseudo-probability in {0.0, 1.0}.
    Use VadGate for stability/hysteresis.
    """

    def __init__(self, cfg: WebRtcVadConfig) -> None:
        try:
            import webrtcvad
        except Exception as e:  # pragma: no cover
            raise RuntimeError("webrtcvad is required for WebRtcVad backend") from e
        if cfg.sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("webrtcvad supports 8/16/32/48 kHz only")
        self._vad = webrtcvad.Vad(int(cfg.aggressiveness))
        self._sr = int(cfg.sample_rate)

    def speech_prob(self, mono_pcm_f32: np.ndarray) -> float:
        if mono_pcm_f32.size == 0:
            return 0.0
        # WebRTC VAD expects 16-bit PCM bytes, 10/20/30ms frames.
        pcm_i16 = np.clip(mono_pcm_f32, -1.0, 1.0)
        pcm_i16 = (pcm_i16 * 32767.0).astype(np.int16)
        is_speech = self._vad.is_speech(pcm_i16.tobytes(), self._sr)
        return 1.0 if is_speech else 0.0


@dataclass(frozen=True)
class SileroVadConfig:
    model_path: str = ""
    sample_rate: int = 16000


class SileroVad(VadBackend):
    """
    Streaming Silero VAD wrapper.

    Load strategy:
    - If `model_path` is provided: load a local TorchScript model via `torch.jit.load`.
    - Else: try loading via the `silero-vad` PyPI package (`import silero_vad`).

    Notes:
    - Silero VAD expects fixed chunk sizes: 512 samples @16kHz (or 256 @8kHz).
    - This wrapper buffers arbitrary-length input into valid chunks and returns
      the mean probability across processed chunks (or 0.0 if not enough audio).
    """

    def __init__(self, cfg: SileroVadConfig) -> None:
        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "torch is required for Silero VAD. Install with: pip install -e .[vad_silero]"
            ) from e

        self._torch = torch
        self._sr = int(cfg.sample_rate)
        if self._sr not in (8000, 16000):
            raise ValueError("Silero VAD wrapper currently supports 8kHz or 16kHz only")
        self._chunk = 256 if self._sr == 8000 else 512

        self._buf = np.zeros((0,), dtype=np.float32)
        self._last_prob = 0.0
        self._model = None

        if cfg.model_path:
            self._model = torch.jit.load(cfg.model_path)
            self._model.eval()
        else:
            self._model = self._load_from_package()
            self._model.eval()

        # Some Silero wrappers keep internal state; reset if available.
        if hasattr(self._model, "reset_states"):
            try:
                self._model.reset_states()
            except Exception:
                pass

    @staticmethod
    def _call_with_supported_kwargs(fn, **kwargs):  # noqa: ANN001
        try:
            sig = inspect.signature(fn)
        except Exception:
            return fn(**kwargs)
        supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**supported)

    def _load_from_package(self):
        """
        Load Silero VAD via the `silero-vad` PyPI package.

        We keep this flexible because different versions expose helpers in
        different places.
        """
        try:
            silero = importlib.import_module("silero_vad")
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Silero VAD backend selected but `silero-vad` is not installed. "
                "Install with: pip install -e .[vad_silero]"
            ) from e

        load_fn = getattr(silero, "load_silero_vad", None)
        if load_fn is None:
            # Some package versions expose utilities under a submodule.
            try:
                utils = importlib.import_module("silero_vad.utils_vad")
                load_fn = getattr(utils, "load_silero_vad", None)
            except Exception:
                load_fn = None

        if load_fn is None:
            raise RuntimeError(
                "Installed `silero-vad` package does not expose `load_silero_vad`. "
                "Either pin a different version of `silero-vad` or provide `vad.silero_model_path`."
            )

        # Prefer CPU to keep setup simple.
        model = self._call_with_supported_kwargs(load_fn, device="cpu", onnx=False)
        return model

    def speech_prob(self, mono_pcm_f32: np.ndarray) -> float:
        if mono_pcm_f32.size == 0:
            return float(self._last_prob)
        x = mono_pcm_f32.astype(np.float32, copy=False).reshape(-1)
        self._buf = np.concatenate([self._buf, x], axis=0)

        probs: list[float] = []
        while self._buf.size >= self._chunk:
            chunk = self._buf[: self._chunk]
            self._buf = self._buf[self._chunk :]

            t = self._torch.from_numpy(chunk).float()
            # Different Silero wrappers accept different call signatures.
            with self._torch.no_grad():
                try:
                    out = self._model(t, self._sr)  # type: ignore[misc]
                except TypeError:
                    out = self._model(t)  # type: ignore[misc]

            # Normalize output to a scalar probability.
            try:
                p = float(out.squeeze().item())
            except Exception:
                try:
                    p = float(out)
                except Exception:
                    p = 0.0
            probs.append(max(0.0, min(1.0, p)))

        if not probs:
            # We received audio but didn't accumulate enough samples to run the model
            # (Silero requires fixed chunk sizes). Return the last computed probability
            # so downstream gating sees a stable value rather than alternating 0 / p / 0 / p.
            return float(self._last_prob)
        self._last_prob = float(np.mean(np.asarray(probs, dtype=np.float32)))
        return float(self._last_prob)
