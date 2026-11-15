"""Measure the localisation noise of the detector, to set R of the Kalman filter.

    python scripts/estimate_noise.py --dets outputs/det/valid_s5.parquet --tau 0.5

Detections with score >= tau are matched to the ground truth (greedy matching, IoU >= 0.5). For each
match, the residual (det - gt) in (cx, cy, w, h) is divided by the ground-truth height:
if the noise std is proportional to h, these normalised residuals have the same law
for every box size. Robust std = 1.4826 x median absolute deviation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitchtrack.boxes import xyxy_to_cxcywh
from pitchtrack.metrics import match_greedy
from pitchtrack.pipeline import frames_to_process
from pitchtrack.soccernet import list_sequences, load_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dets_path = Path(args.dets)
    table = pd.read_parquet(dets_path)
    table = table[table["score"] >= args.tau]
    info = json.loads(dets_path.with_suffix(".json").read_text(encoding="utf-8"))
    cols = ["x1", "y1", "x2", "y2"]
    res, heights = [], []
    for path in list_sequences(info["config"]["data"]["root"], info["split"]):
        if path.name not in info["clips"]:
            continue
        seq = load_sequence(path, info["split"])
        dg = dict(tuple(table[table["seq"] == seq.info.name].groupby("frame")))
        gg = dict(tuple(seq.gt.groupby("frame")))
        for f in frames_to_process(seq, info["stride"]):
            d, g = dg.get(int(f)), gg.get(int(f))
            if d is None or g is None:
                continue
            det_gt, _ = match_greedy(g[cols].to_numpy(), d[cols].to_numpy(), d["score"].to_numpy(), 0.5)
            j = np.flatnonzero(det_gt >= 0)
            if len(j) == 0:
                continue
            zd = xyxy_to_cxcywh(d[cols].to_numpy()[j])
            zg = xyxy_to_cxcywh(g[cols].to_numpy()[det_gt[j]])
            res.append((zd - zg) / zg[:, 3:4])
            heights.append(zg[:, 3])
    res, heights = np.concatenate(res), np.concatenate(heights)
    robust = 1.4826 * np.median(np.abs(res - np.median(res, axis=0)), axis=0)
    bins = [0, 40, 60, 90, 140, 1e4]
    which = np.digitize(heights, bins[1:-1])
    rows = []
    for b in range(len(bins) - 1):
        sel = which == b
        if sel.sum() < 20:
            continue
        r = res[sel]
        rows.append({"height_bin": f"[{bins[b]:g}, {bins[b + 1]:g})", "n": int(sel.sum()),
                     **{f"std_{k}": float(v) for k, v in zip(("cx", "cy", "w", "h"), r.std(axis=0))}})
    kurt = ((res - res.mean(0)) ** 4).mean(0) / res.var(0) ** 2 - 3
    out = Path(args.out or f"reports/noise_{dets_path.stem}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Localisation noise of the detector -- {dets_path.name}, score >= {args.tau}\n\n")
        f.write(f"{len(res)} matched boxes. Normalised residuals (det - gt) / h_gt.\n\n")
        f.write("| | cx | cy | w | h |\n|---|---|---|---|---|\n")
        f.write("| mean | " + " | ".join(f"{v:.4f}" for v in res.mean(0)) + " |\n")
        f.write("| std | " + " | ".join(f"{v:.4f}" for v in res.std(0)) + " |\n")
        f.write("| robust std | " + " | ".join(f"{v:.4f}" for v in robust) + " |\n")
        f.write("| excess kurtosis | " + " | ".join(f"{v:.2f}" for v in kurt) + " |\n\n")
        f.write("## By box height (is the std really proportional to h?)\n\n")
        f.write(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write(f"Suggested Kalman values: r_pos = {robust[:2].mean():.3f}, r_size = {robust[2:].mean():.3f}\n")
    print(f"r_pos ~ {robust[:2].mean():.3f}, r_size ~ {robust[2:].mean():.3f} (robust) -> {out}")


if __name__ == "__main__":
    main()
