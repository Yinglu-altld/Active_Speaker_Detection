import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Optional

try:
    from .respeaker_tuning import RespeakerIds, RespeakerTuning
except ImportError:
    from respeaker_tuning import RespeakerIds, RespeakerTuning


# Board is placed so the hardware overview DOA=90 points to users/front (+Z).
# Convert to robot bearing convention: 0=front(+Z), +=right(+X), -=left(-X).
DOA_AZ_OFFSET_DEG = -90.0
VAD_THRESHOLD_DB = 4.0
APPLY_VAD_THRESHOLD_ON_START = True


def _wrap_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


@dataclass(frozen=True)
class DOAObservation:
    t: float
    raw_azimuth_deg: Optional[float]
    raw_azimuth_plot_deg: Optional[float]
    azimuth_deg: Optional[float]
    azimuth_plot_deg: Optional[float]
    vad_prob: float
    speech_detected: bool
    speech_active: bool
    speech_ended: bool
    audio_conf: float
    conf_doa: float
    conf_doa_srp: float
    sigma_deg: Optional[float]
    doa_updated: bool

    def to_dict(self) -> dict:
        return {
            "t": float(self.t),
            "raw_azimuth_deg": self.raw_azimuth_deg,
            "raw_azimuth_plot_deg": self.raw_azimuth_plot_deg,
            "azimuth_deg": self.azimuth_deg,
            "azimuth_plot_deg": self.azimuth_plot_deg,
            "conf_doa": float(self.conf_doa),
            "conf_doa_srp": float(self.conf_doa_srp),
            "sigma_deg": self.sigma_deg,
            "vad_prob": float(self.vad_prob),
            "speech_prob": float(self.vad_prob),
            "audio_conf": float(self.audio_conf),
            "speech_detected": bool(self.speech_detected),
            "speech_active": bool(self.speech_active),
            "speech_ended": bool(self.speech_ended),
            "entropy": None,
            "conf_components": None,
            "peaks": None,
            "doa_updated": bool(self.doa_updated),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ReSpeaker built-in DOA/VAD (JSON output)")
    p.add_argument("--usb-vendor-id", type=lambda s: int(s, 0), default=RespeakerIds.vendor_id)
    p.add_argument("--usb-product-id", type=lambda s: int(s, 0), default=RespeakerIds.product_id)
    p.add_argument("--poll-hz", type=float, default=20.0)
    p.add_argument("--emit-idle", action="store_true", default=True, help="emit JSON even when not speaking")
    p.add_argument("--no-emit-idle", action="store_false", dest="emit_idle")
    p.add_argument("--max-frames", type=int, default=None, help="optional max frames to emit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ids = RespeakerIds(vendor_id=int(args.usb_vendor_id), product_id=int(args.usb_product_id))
    tuning = RespeakerTuning.find(ids=ids)
    if APPLY_VAD_THRESHOLD_ON_START:
        try:
            tuning.set_vad_threshold_db(float(VAD_THRESHOLD_DB))
        except Exception as exc:
            print(
                f"[doa-core] warning: failed to set GAMMAVAD_SR={VAD_THRESHOLD_DB}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    poll_hz = max(1.0, float(args.poll_hz))
    poll_period = 1.0 / poll_hz

    last_raw_az: Optional[float] = None
    prev_speech_active = False
    emitted = 0
    next_tick = time.time()
    try:
        while True:
            now = time.time()
            try:
                voice = bool(tuning.is_voice())
                raw_az = float(tuning.direction_deg)
            except Exception as exc:
                print(f"[doa-core] warning: USB read failed, reconnecting: {exc}", file=sys.stderr, flush=True)
                try:
                    tuning.close()
                except Exception:
                    pass
                try:
                    tuning = RespeakerTuning.find(ids=ids)
                except Exception as reconnect_exc:
                    print(
                        f"[doa-core] warning: reconnect failed: {reconnect_exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(0.1)
                    continue
                time.sleep(0.05)
                continue
            if voice:
                last_raw_az = raw_az

            speech_detected = voice
            speech_active = speech_detected
            speech_ended = prev_speech_active and (not speech_active)

            doa_updated = speech_detected and last_raw_az is not None
            azimuth_deg = None
            if speech_active and last_raw_az is not None:
                azimuth_deg = float(_wrap_deg(float(last_raw_az) + float(DOA_AZ_OFFSET_DEG)))
            azimuth_plot_deg = float(_wrap_deg(float(raw_az) + float(DOA_AZ_OFFSET_DEG)))

            vad_prob = 1.0 if speech_detected else 0.0
            audio_conf = 1.0 if speech_active else 0.0
            conf_doa = 1.0 if speech_active and last_raw_az is not None else 0.0
            conf_doa_srp = conf_doa

            obs = DOAObservation(
                t=now,
                raw_azimuth_deg=None if last_raw_az is None else float(last_raw_az),
                raw_azimuth_plot_deg=float(raw_az),
                azimuth_deg=azimuth_deg,
                azimuth_plot_deg=azimuth_plot_deg,
                vad_prob=vad_prob,
                speech_detected=speech_detected,
                speech_active=speech_active,
                speech_ended=speech_ended,
                audio_conf=audio_conf,
                conf_doa=conf_doa,
                conf_doa_srp=conf_doa_srp,
                sigma_deg=None,
                doa_updated=doa_updated,
            )
            if not args.emit_idle and (not obs.speech_active) and (not obs.doa_updated):
                pass
            else:
                print(json.dumps(obs.to_dict(), separators=(",", ":")), flush=True)
                emitted += 1
                if args.max_frames is not None and emitted >= int(args.max_frames):
                    break
            prev_speech_active = speech_active

            next_tick += poll_period
            sleep_sec = next_tick - time.time()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
            else:
                next_tick = time.time()
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
