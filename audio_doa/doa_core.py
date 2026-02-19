from dataclasses import dataclass
from collections import deque
import math
from typing import Optional, List, Dict, Deque

import numpy as np
import torch
from silero_vad import load_silero_vad

try:
    from .srp_phat import SRPPhatDOA
except ImportError:
    from srp_phat import SRPPhatDOA


MIC_XY = np.array(
    [[0.028, 0.0], [0.0, 0.028], [-0.028, 0.0], [0.0, -0.028]],
    dtype=np.float64,
)


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _wrap_deg(v: float) -> float:
    return ((float(v) + 180.0) % 360.0) - 180.0


def _circular_deg_diff(a_deg: float, b_deg: float) -> float:
    return ((float(b_deg) - float(a_deg) + 180.0) % 360.0) - 180.0


def _circular_blend(a_deg: float, b_deg: float, alpha: float) -> float:
    return float(a_deg) + float(alpha) * _circular_deg_diff(float(a_deg), float(b_deg))


@dataclass(frozen=True)
class DOAConfig:
    fs: int
    frame_ms: int
    vad_mic: int
    vad_threshold: float
    vad_smooth_alpha: float
    vad_update_threshold: Optional[float]
    energy_threshold: float
    energy_update_threshold: float
    noise_alpha: float
    snr_speech_ratio: float
    snr_speech_add: float
    snr_update_ratio: float
    snr_update_add: float
    speech_hold_ms: int
    doa_quality_threshold: float
    temporal_consistency_deg: float
    temporal_jump_penalty_deg: float
    temporal_conf_boost: float
    temporal_conf_penalty: float
    az_offset_deg: float
    flip_az: bool


@dataclass(frozen=True)
class DOAObservation:
    t: float
    vad_prob: float
    speech_prob: float
    snr_db: float
    audio_conf: float
    energy: float
    noise_energy: float
    speech_gate_energy: float
    update_gate_energy: float
    speech_detected: bool
    speech_active: bool
    speech_ended: bool
    allow_update: bool
    doa_deg: Optional[float]
    doa_conf: Optional[float]
    doa_conf_srp: Optional[float]
    doa_sigma_deg: Optional[float]
    doa_entropy: Optional[float]
    doa_conf_components: Optional[Dict[str, float]]
    doa_peaks: Optional[List[Dict]]
    doa_updated: bool

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "azimuth_deg": self.doa_deg,
            "conf_doa": self.doa_conf,
            "conf_doa_srp": self.doa_conf_srp,
            "sigma_deg": self.doa_sigma_deg,
            "entropy": self.doa_entropy,
            "conf_components": self.doa_conf_components,
            "peaks": self.doa_peaks,
            "vad_prob": self.vad_prob,
            "speech_prob": self.speech_prob,
            "snr_db": self.snr_db,
            "audio_conf": self.audio_conf,
            "energy": self.energy,
            "noise_energy": self.noise_energy,
            "speech_gate_energy": self.speech_gate_energy,
            "update_gate_energy": self.update_gate_energy,
            "speech_detected": self.speech_detected,
            "speech_active": self.speech_active,
            "speech_ended": self.speech_ended,
            "allow_update": self.allow_update,
            "doa_updated": self.doa_updated,
        }


class SileroGate:
    def __init__(self, thr: float = 0.55, sr: int = 16000):
        self.model, self.thr, self.sr = load_silero_vad(), float(thr), int(sr)
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()

    def speech(self, x_int16: np.ndarray) -> tuple[bool, float]:
        x = torch.from_numpy(x_int16.astype(np.float32) / 32768.0)
        best = 0.0
        for i in range(0, len(x), 512):
            c = x[i:i + 512]
            if len(c) < 512:
                c = torch.nn.functional.pad(c, (0, 512 - len(c)))
            p = float(self.model(c, self.sr).item())
            if p > best:
                best = p
        return best >= self.thr, best


class DOAEstimator:
    def __init__(self, cfg: DOAConfig, srp: SRPPhatDOA, vad: SileroGate):
        self.cfg = cfg
        self.srp = srp
        self.vad = vad
        self.vad_sm: Optional[float] = None
        self.noise_e: Optional[float] = None
        self.prev_doa_deg: Optional[float] = None
        self.vad_floor: Optional[float] = None
        self.srp_conf_hist: Deque[float] = deque(maxlen=160)
        self.hold_frames = max(0, int(round(cfg.speech_hold_ms / max(cfg.frame_ms, 1))))
        self.hold = 0
        self.last_good_doa_deg: Optional[float] = None
        self.last_good_conf: Optional[float] = None
        self.last_good_conf_srp: Optional[float] = None
        self.last_good_sigma: Optional[float] = None
        self.last_good_entropy: Optional[float] = None
        self.last_good_components: Optional[Dict[str, float]] = None
        self.last_good_peaks: Optional[List[Dict]] = None
        self.last_good_t: Optional[float] = None

    def reset(self) -> None:
        self.vad_sm = None
        self.noise_e = None
        self.prev_doa_deg = None
        self.vad_floor = None
        self.srp_conf_hist.clear()
        self.hold = 0
        self.last_good_doa_deg = None
        self.last_good_conf = None
        self.last_good_conf_srp = None
        self.last_good_sigma = None
        self.last_good_entropy = None
        self.last_good_components = None
        self.last_good_peaks = None
        self.last_good_t = None
        if hasattr(self.vad.model, "reset_states"):
            self.vad.model.reset_states()

    def process(self, mics_i16: np.ndarray, t: float) -> DOAObservation:
        mono = np.ascontiguousarray(mics_i16[:, self.cfg.vad_mic])
        _, vad_p = self.vad.speech(mono)

        if self.vad_sm is None or float(self.cfg.vad_smooth_alpha) <= 0.0:
            self.vad_sm = float(vad_p)
        else:
            aa = float(self.cfg.vad_smooth_alpha)
            self.vad_sm = aa * float(self.vad_sm) + (1.0 - aa) * float(vad_p)

        energy = float(np.mean(np.abs(mono)))
        if self.noise_e is None:
            self.noise_e = energy
        if self.vad_floor is None:
            self.vad_floor = float(self.vad_sm)

        snr_db = 20.0 * np.log10((energy + 1.0) / (float(self.noise_e) + 1.0))
        vad_thr = float(self.cfg.vad_threshold)
        vad_conf = (
            1.0
            if float(self.vad_sm) >= vad_thr and vad_thr >= 1.0
            else max(0.0, min(1.0, (float(self.vad_sm) - vad_thr) / max(1.0 - vad_thr, 1e-6)))
        )
        snr_conf = max(0.0, min(1.0, (snr_db - 3.0) / 12.0))
        audio_conf = 0.7 * vad_conf + 0.3 * snr_conf

        speech_gate_e = max(
            float(self.cfg.energy_threshold),
            float(self.noise_e) * float(self.cfg.snr_speech_ratio) + float(self.cfg.snr_speech_add),
        )
        speech_detected = (float(self.vad_sm) >= float(self.cfg.vad_threshold)) or (energy >= speech_gate_e)
        if not speech_detected:
            self.vad_floor = 0.985 * float(self.vad_floor) + 0.015 * float(self.vad_sm)

        prev_hold = self.hold
        self.hold = self.hold_frames if speech_detected else max(self.hold - 1, 0)
        speech_active = self.hold > 0
        speech_ended = prev_hold > 0 and self.hold == 0
        if speech_ended and hasattr(self.vad.model, "reset_states"):
            self.vad.model.reset_states()

        update_gate_e = max(
            float(self.cfg.energy_update_threshold),
            float(self.noise_e) * float(self.cfg.snr_update_ratio) + float(self.cfg.snr_update_add),
        )
        vad_update_thr = (
            float(self.cfg.vad_threshold)
            if self.cfg.vad_update_threshold is None
            else float(self.cfg.vad_update_threshold)
        )
        dynamic_vad_update_thr = max(
            0.08,
            min(vad_update_thr, float(self.vad_floor) + 0.08),
        )
        allow_update = (float(self.vad_sm) >= dynamic_vad_update_thr) or (energy >= update_gate_e)

        doa_deg = None
        doa_conf = None
        doa_conf_srp = None
        doa_sigma = None
        doa_entropy = None
        doa_conf_components = None
        doa_peaks = None
        doa_updated = False
        if speech_detected and allow_update:
            out = self.srp.estimate(mics_i16.astype(np.float32))
            if out is not None:
                doa_deg = float(out.doa_deg)
                if bool(self.cfg.flip_az):
                    doa_deg = -doa_deg
                doa_deg = _wrap_deg(doa_deg + float(self.cfg.az_offset_deg))
                srp_conf_raw = _clip01(float(out.conf))
                self.srp_conf_hist.append(float(srp_conf_raw))
                if len(self.srp_conf_hist) >= 12:
                    hist = np.asarray(self.srp_conf_hist, dtype=np.float64)
                    q20 = float(np.quantile(hist, 0.20))
                    q85 = float(np.quantile(hist, 0.85))
                    srp_conf_norm = _clip01((srp_conf_raw - q20) / max(1e-6, q85 - q20))
                else:
                    srp_conf_norm = srp_conf_raw
                doa_conf_srp = _clip01(0.45 * srp_conf_raw + 0.55 * srp_conf_norm)
                audio_blend = 0.55 + 0.45 * float(audio_conf)
                doa_conf_raw = float(doa_conf_srp * audio_blend)
                temporal_diff_deg = None
                temporal_score = 0.5
                temporal_boost = 1.0
                temporal_penalty = 1.0
                if self.prev_doa_deg is not None:
                    temporal_diff_deg = abs(_circular_deg_diff(self.prev_doa_deg, doa_deg))
                    temporal_score = math.exp(
                        -temporal_diff_deg / max(float(self.cfg.temporal_consistency_deg), 1e-6)
                    )
                    temporal_boost = 1.0 + float(self.cfg.temporal_conf_boost) * temporal_score
                    overflow = max(
                        0.0,
                        temporal_diff_deg - float(self.cfg.temporal_consistency_deg),
                    )
                    overflow_ratio = min(
                        1.0,
                        overflow / max(float(self.cfg.temporal_jump_penalty_deg), 1e-6),
                    )
                    temporal_penalty = 1.0 - float(self.cfg.temporal_conf_penalty) * overflow_ratio
                doa_conf = _clip01(doa_conf_raw * temporal_boost * temporal_penalty)

                doa_sigma = out.sigma_deg
                doa_entropy = float(out.entropy)
                doa_conf_components = dict(out.conf_components)
                doa_conf_components["srp_conf_raw"] = float(srp_conf_raw)
                doa_conf_components["srp_conf_norm"] = float(srp_conf_norm)
                doa_conf_components["conf_raw"] = float(doa_conf_raw)
                doa_conf_components["temporal_score"] = float(temporal_score)
                doa_conf_components["temporal_boost"] = float(temporal_boost)
                doa_conf_components["temporal_penalty"] = float(temporal_penalty)
                if temporal_diff_deg is not None:
                    doa_conf_components["temporal_diff_deg"] = float(temporal_diff_deg)
                doa_conf_components["audio_conf"] = float(audio_conf)
                doa_conf_components["vad_conf"] = float(vad_conf)
                doa_conf_components["snr_conf"] = float(snr_conf)
                doa_conf_components["dynamic_vad_update_thr"] = float(dynamic_vad_update_thr)
                doa_peaks = out.peaks
                if len(self.srp_conf_hist) >= 16:
                    hist = np.asarray(self.srp_conf_hist, dtype=np.float64)
                    adaptive_q = float(np.quantile(hist, 0.35))
                    quality_thr = max(0.04, min(float(self.cfg.doa_quality_threshold), adaptive_q))
                else:
                    quality_thr = float(self.cfg.doa_quality_threshold)
                doa_conf_components["quality_threshold"] = float(quality_thr)
                if doa_conf >= float(quality_thr):
                    doa_updated = True
                    self.prev_doa_deg = (
                        doa_deg
                        if self.prev_doa_deg is None
                        else _circular_blend(self.prev_doa_deg, doa_deg, 0.35)
                    )
                    self.last_good_doa_deg = float(doa_deg)
                    self.last_good_conf = float(doa_conf)
                    self.last_good_conf_srp = float(doa_conf_srp)
                    self.last_good_sigma = doa_sigma
                    self.last_good_entropy = doa_entropy
                    self.last_good_components = dict(doa_conf_components)
                    self.last_good_peaks = list(doa_peaks) if doa_peaks is not None else None
                    self.last_good_t = float(t)

        # Keep a short-lived DOA estimate during active speech to avoid az=None bursts.
        if (doa_deg is None or not doa_updated) and speech_active and self.last_good_doa_deg is not None and self.last_good_t is not None:
            age_s = max(0.0, float(t) - float(self.last_good_t))
            hold_s = max(0.12, float(self.cfg.speech_hold_ms) / 1000.0)
            if age_s <= hold_s:
                decay = math.exp(-age_s / max(1e-6, hold_s))
                doa_deg = float(self.last_good_doa_deg)
                doa_conf_srp = float(self.last_good_conf_srp or 0.0) * float(decay)
                held_conf = float(self.last_good_conf or 0.0) * float(decay)
                doa_conf = max(float(doa_conf or 0.0), held_conf)
                doa_sigma = self.last_good_sigma
                doa_entropy = self.last_good_entropy
                if self.last_good_components is not None:
                    doa_conf_components = dict(self.last_good_components)
                    doa_conf_components["held"] = 1.0
                    doa_conf_components["hold_age_s"] = float(age_s)
                if self.last_good_peaks is not None:
                    doa_peaks = list(self.last_good_peaks)

        if not speech_active and float(self.vad_sm) < 0.15:
            na = float(self.cfg.noise_alpha)
            self.noise_e = na * float(self.noise_e) + (1.0 - na) * energy

        return DOAObservation(
            t=t,
            vad_prob=float(vad_p),
            speech_prob=float(self.vad_sm),
            snr_db=float(snr_db),
            audio_conf=float(audio_conf),
            energy=energy,
            noise_energy=float(self.noise_e),
            speech_gate_energy=float(speech_gate_e),
            update_gate_energy=float(update_gate_e),
            speech_detected=speech_detected,
            speech_active=speech_active,
            speech_ended=speech_ended,
            allow_update=allow_update,
            doa_deg=doa_deg,
            doa_conf=doa_conf,
            doa_conf_srp=doa_conf_srp,
            doa_sigma_deg=doa_sigma,
            doa_entropy=doa_entropy,
            doa_conf_components=doa_conf_components,
            doa_peaks=doa_peaks,
            doa_updated=doa_updated,
        )


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Raw DOA estimator (JSON output)")
    p.add_argument("--fs", type=int, default=16000)
    p.add_argument("--channels", type=int, default=6)
    p.add_argument("--device", type=int, default=0, help="sounddevice input device index")
    p.add_argument("--mic-channels", default="1,2,3,4", help="0-based indices inside capture stream")
    p.add_argument("--vad-mic", type=int, default=0, help="index within --mic-channels used for VAD/energy (0..N-1)")
    p.add_argument("--frame-ms", type=int, default=40)
    p.add_argument("--srp-az-step-deg", type=float, default=2.0)
    p.add_argument("--srp-interp", type=int, default=4)
    p.add_argument("--srp-f-low-hz", type=float, default=300.0)
    p.add_argument("--srp-f-high-hz", type=float, default=3400.0)
    p.add_argument("--doa-quality-threshold", type=float, default=0.10)
    p.add_argument("--vad-threshold", type=float, default=0.22)
    p.add_argument("--vad-smooth-alpha", type=float, default=0.80, help="EMA alpha for VAD prob (0 disables smoothing)")
    p.add_argument("--vad-update-threshold", type=float, default=0.22, help="minimum VAD prob required to update DOA")
    p.add_argument("--energy-threshold", type=float, default=80.0, help="fallback speech gate on mean abs mono int16")
    p.add_argument("--energy-update-threshold", type=float, default=140.0, help="fallback update gate when VAD is uncertain")
    p.add_argument("--noise-alpha", type=float, default=0.97, help="EMA factor for noise floor energy estimate (higher = slower)")
    p.add_argument("--snr-speech-ratio", type=float, default=1.4, help="speech gate: energy >= noise*ratio + add")
    p.add_argument("--snr-speech-add", type=float, default=20.0, help="speech gate: energy >= noise*ratio + add")
    p.add_argument("--snr-update-ratio", type=float, default=1.7, help="update gate (when VAD low): energy >= noise*ratio + add")
    p.add_argument("--snr-update-add", type=float, default=35.0, help="update gate (when VAD low): energy >= noise*ratio + add")
    p.add_argument("--speech-hold-ms", type=int, default=220, help="continue tracking this long after VAD drops")
    p.add_argument(
        "--temporal-consistency-deg",
        type=float,
        default=18.0,
        help="smaller values enforce stronger confidence boost only for stable DOA trajectories",
    )
    p.add_argument(
        "--temporal-jump-penalty-deg",
        type=float,
        default=55.0,
        help="angular jump span where abrupt DOA changes are progressively penalized",
    )
    p.add_argument(
        "--temporal-conf-boost",
        type=float,
        default=0.25,
        help="max confidence boost for temporally consistent DOA (0 disables)",
    )
    p.add_argument(
        "--temporal-conf-penalty",
        type=float,
        default=0.40,
        help="max confidence penalty for abrupt temporal jumps (0 disables)",
    )
    p.add_argument(
        "--az-offset-deg",
        type=float,
        default=0.0,
        help="constant DOA azimuth offset (degrees) to align array coordinates with camera/Furhat bearing frame",
    )
    p.add_argument(
        "--flip-az",
        action="store_true",
        default=False,
        help="mirror DOA azimuth sign before applying --az-offset-deg",
    )
    p.add_argument("--no-flip-az", action="store_false", dest="flip_az")
    p.add_argument("--emit-idle", action="store_true", default=True, help="emit JSON even when not speaking")
    p.add_argument("--no-emit-idle", action="store_false", dest="emit_idle")
    p.add_argument("--max-frames", type=int, default=None, help="optional max frames to emit")
    return p.parse_args()


def main() -> None:
    import json
    import queue
    import time

    import sounddevice as sd

    a = _parse_args()
    idx = [int(s.strip()) for s in a.mic_channels.split(",")]
    vad_mic = int(a.vad_mic)
    if vad_mic < 0 or vad_mic >= len(idx):
        vad_mic = 0

    doa_cfg = DOAConfig(
        fs=a.fs,
        frame_ms=a.frame_ms,
        vad_mic=vad_mic,
        vad_threshold=a.vad_threshold,
        vad_smooth_alpha=a.vad_smooth_alpha,
        vad_update_threshold=a.vad_update_threshold,
        energy_threshold=a.energy_threshold,
        energy_update_threshold=a.energy_update_threshold,
        noise_alpha=a.noise_alpha,
        snr_speech_ratio=a.snr_speech_ratio,
        snr_speech_add=a.snr_speech_add,
        snr_update_ratio=a.snr_update_ratio,
        snr_update_add=a.snr_update_add,
        speech_hold_ms=a.speech_hold_ms,
        doa_quality_threshold=a.doa_quality_threshold,
        temporal_consistency_deg=a.temporal_consistency_deg,
        temporal_jump_penalty_deg=a.temporal_jump_penalty_deg,
        temporal_conf_boost=a.temporal_conf_boost,
        temporal_conf_penalty=a.temporal_conf_penalty,
        az_offset_deg=a.az_offset_deg,
        flip_az=bool(a.flip_az),
    )
    srp = SRPPhatDOA(
        MIC_XY,
        fs=a.fs,
        az_step_deg=a.srp_az_step_deg,
        interp=a.srp_interp,
        f_low_hz=a.srp_f_low_hz,
        f_high_hz=a.srp_f_high_hz,
    )
    doa_est = DOAEstimator(doa_cfg, srp, SileroGate(a.vad_threshold, a.fs))
    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=16)

    def cb(indata, frames, time_info, status):
        if not q.full():
            q.put_nowait(indata.copy())

    emitted = 0
    with sd.InputStream(
        device=a.device,
        samplerate=a.fs,
        channels=a.channels,
        dtype="int16",
        blocksize=int(a.fs * a.frame_ms / 1000),
        callback=cb,
    ):
        while True:
            frame = q.get()
            mics_i16 = frame[:, idx]
            obs = doa_est.process(mics_i16, time.time())
            if not a.emit_idle and (not obs.speech_active) and (not obs.doa_updated):
                continue
            print(json.dumps(obs.to_dict(), separators=(",", ":")), flush=True)
            emitted += 1
            if a.max_frames is not None and emitted >= a.max_frames:
                break


if __name__ == "__main__":
    main()
