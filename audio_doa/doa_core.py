from dataclasses import dataclass
from typing import Optional, List, Dict

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
        self.hold_frames = max(0, int(round(cfg.speech_hold_ms / max(cfg.frame_ms, 1))))
        self.hold = 0

    def reset(self) -> None:
        self.vad_sm = None
        self.noise_e = None
        self.hold = 0
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
        allow_update = (float(self.vad_sm) >= vad_update_thr) or (energy >= update_gate_e)

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
                doa_conf_srp = float(out.conf)
                doa_conf = float(doa_conf_srp * audio_conf)
                doa_sigma = out.sigma_deg
                doa_entropy = float(out.entropy)
                doa_conf_components = dict(out.conf_components)
                doa_conf_components["audio_conf"] = float(audio_conf)
                doa_conf_components["vad_conf"] = float(vad_conf)
                doa_conf_components["snr_conf"] = float(snr_conf)
                doa_peaks = out.peaks
                if doa_conf >= float(self.cfg.doa_quality_threshold):
                    doa_updated = True

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
    p.add_argument("--frame-ms", type=int, default=80)
    p.add_argument("--srp-az-step-deg", type=float, default=2.0)
    p.add_argument("--srp-interp", type=int, default=4)
    p.add_argument("--srp-f-low-hz", type=float, default=300.0)
    p.add_argument("--srp-f-high-hz", type=float, default=3400.0)
    p.add_argument("--doa-quality-threshold", type=float, default=0.20)
    p.add_argument("--vad-threshold", type=float, default=0.22)
    p.add_argument("--vad-smooth-alpha", type=float, default=0.80, help="EMA alpha for VAD prob (0 disables smoothing)")
    p.add_argument("--vad-update-threshold", type=float, default=0.30, help="minimum VAD prob required to update DOA")
    p.add_argument("--energy-threshold", type=float, default=150.0, help="fallback speech gate on mean abs mono int16")
    p.add_argument("--energy-update-threshold", type=float, default=250.0, help="fallback update gate when VAD is uncertain")
    p.add_argument("--noise-alpha", type=float, default=0.97, help="EMA factor for noise floor energy estimate (higher = slower)")
    p.add_argument("--snr-speech-ratio", type=float, default=1.8, help="speech gate: energy >= noise*ratio + add")
    p.add_argument("--snr-speech-add", type=float, default=35.0, help="speech gate: energy >= noise*ratio + add")
    p.add_argument("--snr-update-ratio", type=float, default=2.2, help="update gate (when VAD low): energy >= noise*ratio + add")
    p.add_argument("--snr-update-add", type=float, default=60.0, help="update gate (when VAD low): energy >= noise*ratio + add")
    p.add_argument("--speech-hold-ms", type=int, default=300, help="continue tracking this long after VAD drops")
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
