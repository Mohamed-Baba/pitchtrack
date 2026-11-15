"""Distances from the pipeline versus the annotated positions -> markdown report.

    python scripts/eval_statistics.py --positions outputs/stats/valid_oracle/positions.parquet

Run it on positions computed from the GROUND-TRUTH tracks (run_statistics.py --oracle): every
track then has a reference trajectory, its bbox_pitch positions (an annotated
reference, noisy too). For several settings (smoothing window, summation step), the table
gives the median distance per minute of the reference and of the pipeline, their median relative
difference (the bias due to the calibration and to the foot point), and the noise level of
each trajectory (robust std of position minus smoothed position). Only tracks with positions
for at least 80 % of their frames and at least 5 s long are used.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pitchtrack.soccernet import list_sequences, load_sequence
from pitchtrack.statistics import smooth, track_kinematics

SETTINGS = [(0.0, 0.0), (0.0, 0.2), (0.4, 0.2), (1.0, 0.5), (2.0, 1.0)]   # (smooth_s, step_s)


def noise_std(t, xy):
    r = xy - smooth(t, xy, 0.4)
    mad = np.median(np.abs(r - np.median(r, axis=0)), axis=0)
    return float(np.mean(1.4826 * mad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--data-root", default=None, help="default: data.root of configs/baseline.yaml")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--max-speed", type=float, default=12.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pos = pd.read_parquet(args.positions)
    root = args.data_root
    if root is None:
        from pitchtrack.pipeline import load_config
        root = load_config("configs/baseline.yaml")["data"]["root"]
    rows = []
    for path in list_sequences(root, args.split):
        name = path.name
        if name not in set(pos["seq"]):
            continue
        gt = load_sequence(path, args.split).gt
        p = pos[pos["seq"] == name].merge(gt[["frame", "track_id", "pitch_x", "pitch_y"]], on=["frame", "track_id"])
        for tid, g in p.groupby("track_id"):
            g = g.sort_values("t")
            if np.isfinite(g["X"]).mean() < 0.8 or g["t"].iat[-1] - g["t"].iat[0] < 5.0:
                continue
            t = g["t"].to_numpy()
            ours, ref = g[["X", "Y"]].to_numpy(), g[["pitch_x", "pitch_y"]].to_numpy()
            both = np.isfinite(ours).all(1) & np.isfinite(ref).all(1)
            row = {"seq": name, "track_id": tid, "noise_ref": noise_std(t[both], ref[both]),
                   "noise_ours": noise_std(t[both], ours[both])}
            for s, step in SETTINGS:
                for label, xy in (("ref", ref), ("ours", ours)):
                    k = track_kinematics(t, np.where(both[:, None], xy, np.nan), 0.5, s, step, args.max_speed, 2.0)
                    row[f"{label}_{s}_{step}"] = 60 * k.distance_m / k.duration_s if k.duration_s > 0 else np.nan
            rows.append(row)
    r = pd.DataFrame(rows)
    lines = [f"{len(r)} ground-truth tracks of {r['seq'].nunique()} clips. Noise (robust std of position minus "
             f"position smoothed over 0.4 s): reference {r['noise_ref'].median():.3f} m, pipeline "
             f"{r['noise_ours'].median():.3f} m (medians over tracks).", "",
             "| smoothing (s) | step (s) | reference (m/min) | pipeline (m/min) | pipeline vs reference |",
             "|---|---|---|---|---|"]
    for s, step in SETTINGS:
        ref, ours = r[f"ref_{s}_{step}"], r[f"ours_{s}_{step}"]
        rel = ((ours - ref) / ref).median()
        lines.append(f"| {s:g} | {step:g} | {ref.median():.1f} | {ours.median():.1f} | {rel:+.1%} |")
    out = Path(args.out or f"reports/statistics_{Path(args.positions).parent.name}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Statistics against the annotated positions -- {args.positions}\n\n" + "\n".join(lines) + "\n",
                   encoding="utf-8")
    print("\n".join(lines) + f"\n-> {out}")


if __name__ == "__main__":
    main()
