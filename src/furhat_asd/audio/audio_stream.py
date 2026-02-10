from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    ts_ms: int
    pcm_f32: np.ndarray  # (frames, channels)


class AudioInput:
    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def read(self) -> AudioChunk:
        raise NotImplementedError


class SoundDeviceAudioInput(AudioInput):
    def __init__(self, device: str, sample_rate: int | None, channels: int, block_ms: int) -> None:
        self._device = device
        self._sample_rate_req = sample_rate
        self._sample_rate: int | None = None
        self._channels = channels
        self._block_ms = int(block_ms)
        self._block_frames: int | None = None
        self._queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=200)
        self._stream = None
        self._loop = asyncio.get_event_loop()
        self._log = logging.getLogger("furhat_asd.audio")

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            raise RuntimeError("Audio stream not started yet")
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def start(self) -> None:
        try:
            import sounddevice as sd
        except Exception as e:  # pragma: no cover
            raise RuntimeError("sounddevice is required for audio capture") from e

        device_arg = self._resolve_device(sd)

        # Validate requested channel count against the selected device.
        try:
            info = sd.query_devices(device_arg, "input")
            max_ch = int(info.get("max_input_channels", 0) or 0)
        except Exception:
            max_ch = 0
        if max_ch > 0 and self._channels > max_ch:
            raise RuntimeError(
                f"Audio device {device_arg!r} supports max_input_channels={max_ch}, "
                f"but config requests channels={self._channels}. "
                f"Fix config.audio.device/config.audio.channels or run --list-audio-devices."
            )

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                return
            pcm = np.asarray(indata, dtype=np.float32).copy()
            ts_ms = int(time.time() * 1000)
            chunk = AudioChunk(ts_ms=ts_ms, pcm_f32=pcm)
            try:
                self._loop.call_soon_threadsafe(_safe_put, chunk)
            except Exception:
                return

        def _safe_put(ch: AudioChunk) -> None:
            """
            Queue audio chunks without crashing the event loop if the consumer falls behind.
            If the queue is full, drop the oldest chunk and keep the newest (ring-buffer behavior).
            """
            try:
                self._queue.put_nowait(ch)
            except asyncio.QueueFull:
                try:
                    _ = self._queue.get_nowait()
                except Exception:
                    return
                try:
                    self._queue.put_nowait(ch)
                except asyncio.QueueFull:
                    return

        def _try_start(sr: int) -> None:
            self._sample_rate = int(sr)
            self._block_frames = int(self._sample_rate * (self._block_ms / 1000.0))
            self._stream = sd.InputStream(
                device=device_arg,
                channels=self._channels,
                samplerate=self._sample_rate,
                blocksize=self._block_frames,
                dtype="float32",
                callback=callback,
            )
            self._stream.start()

        # Resolve sample rate. PortAudio device capabilities vary on Windows; if the
        # configured rate fails, try a small set of common fallbacks.
        candidates: list[int] = []
        if self._sample_rate_req is not None and int(self._sample_rate_req) > 0:
            candidates.append(int(self._sample_rate_req))
        else:
            try:
                info = sd.query_devices(device_arg, "input")
                dsr = int(float(info.get("default_samplerate", 0)) or 0)
                if dsr > 0:
                    candidates.append(dsr)
            except Exception:
                pass
        for sr in (16000, 48000, 32000, 8000):
            candidates.append(sr)
        # De-dup while keeping order.
        seen: set[int] = set()
        dedup: list[int] = []
        for sr in candidates:
            if sr <= 0 or sr in seen:
                continue
            seen.add(sr)
            dedup.append(sr)
        candidates = dedup

        last_err: Exception | None = None
        for sr in candidates:
            try:
                _try_start(sr)
                return
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(
            f"Failed to open audio input stream (device={self._device!r}, channels={self._channels}). "
            f"Tried sample rates: {candidates}. Last error: {last_err}"
        ) from last_err

    def _resolve_device(self, sd) -> int | str | None:  # noqa: ANN001
        """
        Resolve `audio.device` into a PortAudio device argument.

        Supported values:
        - "" / null-like -> default input device
        - "14" -> device index 14
        - any other string -> treated as a case-insensitive substring hint; picks the first
          matching input device that supports the requested channel count.

        This avoids brittle device indices on Windows when devices are unplugged/replugged.
        """
        if not self._device:
            return None

        # Allow passing device index as a string, e.g. "14".
        try:
            return int(self._device)
        except ValueError:
            pass

        hint = str(self._device).strip().lower()
        if not hint:
            return None

        try:
            devices = sd.query_devices()
        except Exception:
            return self._device

        candidates: list[tuple[int, int, str]] = []
        for idx, d in enumerate(devices):
            try:
                name = str(d.get("name", ""))
                max_in = int(d.get("max_input_channels", 0) or 0)
            except Exception:
                continue
            if max_in <= 0:
                continue
            if hint in name.lower():
                candidates.append((idx, max_in, name))

        if not candidates:
            self._log.warning("No audio input device matched hint=%r; falling back to PortAudio name lookup.", self._device)
            return self._device

        # Prefer devices that can satisfy the requested channel count.
        good = [c for c in candidates if c[1] >= self._channels]
        chosen = (good[0] if good else sorted(candidates, key=lambda x: x[1], reverse=True)[0])
        self._log.info("Resolved audio.device hint=%r -> index=%d name=%r max_in=%d", self._device, chosen[0], chosen[2], chosen[1])
        return int(chosen[0])

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._sample_rate = None
        self._block_frames = None

    async def read(self) -> AudioChunk:
        return await self._queue.get()
