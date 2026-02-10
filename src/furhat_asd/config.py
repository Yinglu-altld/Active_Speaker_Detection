from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


def _as_int_or_none(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in {"auto", "none", ""}:
        return None
    if isinstance(v, (int, float)) and int(v) <= 0:
        return None
    try:
        return int(v)
    except Exception:
        return None


@dataclass(frozen=True)
class FurhatConfig:
    ip: str
    api_key: str
    ws_port: int = 9000


@dataclass(frozen=True)
class ControlConfig:
    loop_hz: int = 15
    speech_on_conf: float = 0.8
    speech_on_ms: int = 300
    speech_off_conf: float = 0.4
    speech_off_ms: int = 500
    min_burst_ms: int = 150
    doa_sigma_deg: float = 20.0
    doa_offset_deg: float = 0.0
    doa_sign: int = 1
    # Rolling smoothing window for DOA (ms) during speech_on.
    doa_window_ms: int = 1500
    # Treat DOA as "usable" only if both:
    # - average DOA confidence >= this threshold, and
    # - angular spread in the smoothing window <= doa_usable_max_spread_deg
    doa_usable_min_conf: float = 0.15
    doa_usable_max_spread_deg: float = 70.0
    switch_margin_ratio: float = 1.25
    switch_hold_ms: int = 300
    attend_addressing_threshold: float = 0.6


@dataclass(frozen=True)
class AudioConfig:
    device: str
    mode: Literal["local", "udp"] = "local"
    # If null/"auto", we try the device default and common fallbacks.
    sample_rate: int | None = 16000
    channels: int = 4
    channel_indices: list[int] | None = None
    # Optional: use a single device channel for VAD (e.g. ReSpeaker channel 0).
    # When set, this index refers to the *device* channel index (0..channels-1),
    # not the post-selected channel indices.
    vad_channel_index: int | None = None
    block_ms: int = 20


@dataclass(frozen=True)
class DoaConfig:
    enabled: bool = True
    method: Literal["srp_phat"] = "srp_phat"
    mic_positions_m: list[list[float]] | None = None
    # Use a longer window for more stable DOA estimates (at the cost of latency).
    frame_ms: int = 200
    search_step_deg: int = 5
    gcc_interp: int = 1


@dataclass(frozen=True)
class VadConfig:
    backend: Literal["energy", "webrtc", "silero"] = "energy"
    energy_threshold: float = 0.01
    webrtc_aggressiveness: int = 2
    silero_model_path: str = ""


@dataclass(frozen=True)
class VisionConfig:
    enabled: bool = False


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(frozen=True)
class OutputConfig:
    segments_jsonl: str = "out/segments.jsonl"


@dataclass(frozen=True)
class UdpAudioConfig:
    listen_host: str = "0.0.0.0"
    listen_port: int = 17800
    target_host: str = ""
    target_port: int = 17800
    send_hz: int = 20


@dataclass(frozen=True)
class AppConfig:
    furhat: FurhatConfig
    control: ControlConfig
    audio: AudioConfig
    doa: DoaConfig
    vad: VadConfig
    vision: VisionConfig
    output: OutputConfig
    udp_audio: UdpAudioConfig
    logging: LoggingConfig


def _get(d: dict[str, Any], key: str, default: Any) -> Any:
    return d[key] if key in d else default


def load_config(path: str | Path) -> AppConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    furhat_raw = raw.get("furhat", {})
    control_raw = raw.get("control", {})
    audio_raw = raw.get("audio", {})
    doa_raw = raw.get("doa", {})
    vad_raw = raw.get("vad", {})
    vision_raw = raw.get("vision", {})
    output_raw = raw.get("output", {})
    udp_audio_raw = raw.get("udp_audio", {})
    logging_raw = raw.get("logging", {})

    furhat = FurhatConfig(
        ip=str(furhat_raw["ip"]),
        api_key=str(_get(furhat_raw, "api_key", "")),
        ws_port=int(_get(furhat_raw, "ws_port", 9000)),
    )
    control = ControlConfig(
        loop_hz=int(_get(control_raw, "loop_hz", 15)),
        speech_on_conf=float(_get(control_raw, "speech_on_conf", 0.8)),
        speech_on_ms=int(_get(control_raw, "speech_on_ms", 300)),
        speech_off_conf=float(_get(control_raw, "speech_off_conf", 0.4)),
        speech_off_ms=int(_get(control_raw, "speech_off_ms", 500)),
        min_burst_ms=int(_get(control_raw, "min_burst_ms", 150)),
        doa_sigma_deg=float(_get(control_raw, "doa_sigma_deg", 20.0)),
        doa_offset_deg=float(_get(control_raw, "doa_offset_deg", 0.0)),
        doa_sign=int(_get(control_raw, "doa_sign", 1)),
        doa_window_ms=int(_get(control_raw, "doa_window_ms", 1500)),
        doa_usable_min_conf=float(_get(control_raw, "doa_usable_min_conf", 0.15)),
        doa_usable_max_spread_deg=float(_get(control_raw, "doa_usable_max_spread_deg", 70.0)),
        switch_margin_ratio=float(_get(control_raw, "switch_margin_ratio", 1.25)),
        switch_hold_ms=int(_get(control_raw, "switch_hold_ms", 300)),
        attend_addressing_threshold=float(_get(control_raw, "attend_addressing_threshold", 0.6)),
    )
    audio = AudioConfig(
        device=str(_get(audio_raw, "device", "")),
        mode=str(_get(audio_raw, "mode", "local")),
        sample_rate=_as_int_or_none(_get(audio_raw, "sample_rate", 16000)),
        channels=int(_get(audio_raw, "channels", 4)),
        channel_indices=_get(audio_raw, "channel_indices", None),
        vad_channel_index=_as_int_or_none(_get(audio_raw, "vad_channel_index", None)),
        block_ms=int(_get(audio_raw, "block_ms", 20)),
    )
    doa = DoaConfig(
        enabled=bool(_get(doa_raw, "enabled", True)),
        method=str(_get(doa_raw, "method", "srp_phat")),
        mic_positions_m=_get(doa_raw, "mic_positions_m", None),
        frame_ms=int(_get(doa_raw, "frame_ms", 200)),
        search_step_deg=int(_get(doa_raw, "search_step_deg", 5)),
        gcc_interp=int(_get(doa_raw, "gcc_interp", 1)),
    )
    vad = VadConfig(
        backend=str(_get(vad_raw, "backend", "energy")),
        energy_threshold=float(_get(vad_raw, "energy_threshold", 0.01)),
        webrtc_aggressiveness=int(_get(vad_raw, "webrtc_aggressiveness", 2)),
        silero_model_path=str(_get(vad_raw, "silero_model_path", "")),
    )
    vision = VisionConfig(enabled=bool(_get(vision_raw, "enabled", False)))
    output = OutputConfig(segments_jsonl=str(_get(output_raw, "segments_jsonl", "out/segments.jsonl")))
    udp_audio = UdpAudioConfig(
        listen_host=str(_get(udp_audio_raw, "listen_host", "0.0.0.0")),
        listen_port=int(_get(udp_audio_raw, "listen_port", 17800)),
        target_host=str(_get(udp_audio_raw, "target_host", "")),
        target_port=int(_get(udp_audio_raw, "target_port", 17800)),
        send_hz=int(_get(udp_audio_raw, "send_hz", 20)),
    )
    logging = LoggingConfig(level=str(_get(logging_raw, "level", "INFO")))

    return AppConfig(
        furhat=furhat,
        control=control,
        audio=audio,
        doa=doa,
        vad=vad,
        vision=vision,
        output=output,
        udp_audio=udp_audio,
        logging=logging,
    )
