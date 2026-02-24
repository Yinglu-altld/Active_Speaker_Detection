import argparse
import queue
import time
from itertools import permutations

import numpy as np
import sounddevice as sd

try:
    from .srp_phat import SRPPhatDOA
except ImportError:
    from srp_phat import SRPPhatDOA


_MIC_R = 0.028 / np.sqrt(2.0)
MIC_XY = np.array(
    [
        [_MIC_R, _MIC_R],    # mic1 (front-right)
        [_MIC_R, -_MIC_R],   # mic2 (front-left)
        [-_MIC_R, -_MIC_R],  # mic3 (back-left)
        [-_MIC_R, _MIC_R],   # mic4 (back-right)
    ],
    dtype=np.float64,
)

def _wrap_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _circ_mean_deg(values: list[float]) -> float | None:
    if not values:
        return None
    rad = np.deg2rad(np.asarray(values, dtype=np.float64))
    vec = np.exp(1j * rad)
    mean_vec = np.mean(vec)
    if np.abs(mean_vec) < 1e-12:
        return None
    return float(np.rad2deg(np.angle(mean_vec)))


def _safe_fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "None"
    return f"{float(value):.{digits}f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate 4-mic channel permutation for SRP DOA using LEFT/CENTER/RIGHT speech samples."
    )
    p.add_argument("--device", type=int, default=1, help="sounddevice input device index")
    p.add_argument("--fs", type=int, default=16000)
    p.add_argument("--channels", type=int, default=6, help="total capture channels")
    p.add_argument("--frame-ms", type=int, default=60)
    p.add_argument(
        "--candidate-mics",
        default="1,2,3,4",
        help="4 capture-channel indices to permute, e.g. '1,2,3,4'",
    )
    p.add_argument("--segment-sec", type=float, default=4.0, help="recording duration per position")
    p.add_argument("--min-energy", type=float, default=80.0, help="minimum mean abs int16 to keep frame")
    p.add_argument("--min-conf", type=float, default=0.05, help="minimum SRP confidence to keep frame")
    p.add_argument("--min-separation-deg", type=float, default=25.0, help="minimum L-R separation after centering")
    p.add_argument("--srp-az-step-deg", type=float, default=2.0)
    p.add_argument("--srp-interp", type=int, default=4)
    p.add_argument("--srp-f-low-hz", type=float, default=100.0)
    p.add_argument("--srp-f-high-hz", type=float, default=3200.0)
    p.add_argument("--top-k", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_mics = [int(x.strip()) for x in args.candidate_mics.split(",") if x.strip()]
    if len(base_mics) != 4:
        raise ValueError("--candidate-mics must contain exactly 4 indices.")
    if len(set(base_mics)) != 4:
        raise ValueError("--candidate-mics must contain 4 unique indices.")
    if min(base_mics) < 0 or max(base_mics) >= int(args.channels):
        raise ValueError("candidate mic index out of range for --channels.")

    perms = list(permutations(base_mics, 4))
    srp_by_perm = {
        perm: SRPPhatDOA(
            MIC_XY,
            fs=int(args.fs),
            az_step_deg=float(args.srp_az_step_deg),
            interp=int(args.srp_interp),
            f_low_hz=float(args.srp_f_low_hz),
            f_high_hz=float(args.srp_f_high_hz),
        )
        for perm in perms
    }
    labels = ["LEFT", "CENTER", "RIGHT"]
    seg_az: dict[tuple[int, int, int, int], dict[str, list[float]]] = {
        perm: {label: [] for label in labels} for perm in perms
    }
    seg_conf: dict[tuple[int, int, int, int], dict[str, list[float]]] = {
        perm: {label: [] for label in labels} for perm in perms
    }

    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)

    def cb(indata, frames, time_info, status):
        if not q.full():
            q.put_nowait(indata.copy())

    blocksize = int(float(args.fs) * float(args.frame_ms) / 1000.0)
    print(f"Testing {len(perms)} permutations from candidate mics {base_mics}.")
    print("You will record 3 short segments: LEFT, CENTER, RIGHT.")
    print("For each segment: speak continuously in that position.")
    print("")

    with sd.InputStream(
        device=int(args.device),
        samplerate=int(args.fs),
        channels=int(args.channels),
        dtype="int16",
        blocksize=blocksize,
        callback=cb,
    ):
        for label in labels:
            input(f"Press Enter to start {label} segment ({args.segment_sec:.1f}s)... ")
            start = time.time()
            used = 0
            total = 0
            while time.time() - start < float(args.segment_sec):
                try:
                    frame = q.get(timeout=0.3)
                except queue.Empty:
                    continue
                total += 1
                frame = frame.astype(np.float32)
                for perm in perms:
                    mics = frame[:, perm]
                    energy = float(np.mean(np.abs(mics[:, 0])))
                    if energy < float(args.min_energy):
                        continue
                    out = srp_by_perm[perm].estimate(mics)
                    if out is None:
                        continue
                    if float(out.conf) < float(args.min_conf):
                        continue
                    seg_az[perm][label].append(float(out.doa_deg))
                    seg_conf[perm][label].append(float(out.conf))
                    used += 1
            print(f"{label}: collected frames={total}, accepted-evals={used}")

    ranked = []
    for perm in perms:
        l_raw = _circ_mean_deg(seg_az[perm]["LEFT"])
        c_raw = _circ_mean_deg(seg_az[perm]["CENTER"])
        r_raw = _circ_mean_deg(seg_az[perm]["RIGHT"])
        l_n = len(seg_az[perm]["LEFT"])
        c_n = len(seg_az[perm]["CENTER"])
        r_n = len(seg_az[perm]["RIGHT"])
        if l_raw is None or c_raw is None or r_raw is None:
            ranked.append(
                {
                    "perm": perm,
                    "loss": 1e9,
                    "reason": "insufficient-data",
                    "l_raw": l_raw,
                    "c_raw": c_raw,
                    "r_raw": r_raw,
                    "l_rel": None,
                    "r_rel": None,
                    "offset_deg": None,
                    "n": (l_n, c_n, r_n),
                    "conf": 0.0,
                }
            )
            continue

        # Remove unknown global offset using CENTER as reference.
        l_rel = _wrap_deg(l_raw - c_raw)
        r_rel = _wrap_deg(r_raw - c_raw)
        offset_deg = _wrap_deg(-c_raw)

        # We want left<0 and right>0 after centering.
        sign_pen = 0.0
        if l_rel >= 0.0:
            sign_pen += 50.0 + abs(l_rel)
        if r_rel <= 0.0:
            sign_pen += 50.0 + abs(r_rel)

        sep = r_rel - l_rel
        sep_pen = max(0.0, float(args.min_separation_deg) - sep) * 2.0
        asym_pen = abs(abs(l_rel) - abs(r_rel)) * 0.2
        mean_conf = float(
            np.mean(
                [
                    np.mean(seg_conf[perm]["LEFT"]) if seg_conf[perm]["LEFT"] else 0.0,
                    np.mean(seg_conf[perm]["CENTER"]) if seg_conf[perm]["CENTER"] else 0.0,
                    np.mean(seg_conf[perm]["RIGHT"]) if seg_conf[perm]["RIGHT"] else 0.0,
                ]
            )
        )
        conf_pen = max(0.0, 0.25 - mean_conf) * 40.0
        loss = sign_pen + sep_pen + asym_pen + conf_pen

        ranked.append(
            {
                "perm": perm,
                "loss": float(loss),
                "reason": "ok",
                "l_raw": l_raw,
                "c_raw": c_raw,
                "r_raw": r_raw,
                "l_rel": l_rel,
                "r_rel": r_rel,
                "offset_deg": offset_deg,
                "n": (l_n, c_n, r_n),
                "conf": mean_conf,
            }
        )

    ranked.sort(key=lambda item: float(item["loss"]))
    top_k = max(1, min(int(args.top_k), len(ranked)))
    print("")
    print("Top candidates (lower loss is better):")
    print("perm\tloss\tLraw/Craw/Rraw\tLrel/Rrel\toffset\tN(L,C,R)\tconf")
    for row in ranked[:top_k]:
        perm_str = ",".join(str(x) for x in row["perm"])
        n = row["n"]
        print(
            f"{perm_str}\t{row['loss']:.2f}\t"
            f"{_safe_fmt(row['l_raw'])}/{_safe_fmt(row['c_raw'])}/{_safe_fmt(row['r_raw'])}\t"
            f"{_safe_fmt(row['l_rel'])}/{_safe_fmt(row['r_rel'])}\t"
            f"{_safe_fmt(row['offset_deg'])}\t"
            f"{n[0]},{n[1]},{n[2]}\t{row['conf']:.3f}"
        )

    best = ranked[0]
    best_perm = ",".join(str(x) for x in best["perm"])
    print("")
    print(f"Recommended --mic-channels: {best_perm}")
    if best["offset_deg"] is not None:
        print(f"Estimated SRP->front offset (deg): {best['offset_deg']:.1f}")
        print("Use this as a starting calibration value for board offset in mapping layers.")


if __name__ == "__main__":
    main()
