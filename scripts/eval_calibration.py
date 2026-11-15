"""How many metres does the calibration get wrong? -> markdown report + figure.

    python scripts/eval_calibration.py --calib outputs/calib/valid.parquet

For every calibrated frame, the bottom-middle of each ground-truth box is projected with the
estimated homography and compared with its bbox_pitch (metres). This is the error that
the statistics inherit from the calibration ALONE (the boxes are the true ones). Reported: the
share of frames calibrated per source, the error per source, per depth (pitch y of the player:
far side, middle, near side) and over the image (a map of the median error per image cell,
where lens distortion shows up as a ring), the residuals of the line points, and the temporal
stability (frame-to-frame jump of the error of the same player). 95 % intervals over clips.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pitchtrack.boxes import foot_points  # noqa: E402
from pitchtrack.pitch_calibration import image_to_pitch  # noqa: E402
from pitchtrack.soccernet import list_sequences, load_sequence  # noqa: E402


def clip_errors(seq, cal: pd.DataFrame) -> pd.DataFrame:
    gt = seq.gt.dropna(subset=["pitch_x", "pitch_y"])
    cal = cal.set_index("frame")
    rows = []
    for f, g in gt.groupby("frame"):
        if f not in cal.index or not bool(cal.at[f, "ok"]):
            continue
        G = cal.loc[f, [f"g{i}" for i in range(9)]].to_numpy(dtype=np.float64).reshape(3, 3)
        uv = foot_points(g[["x1", "y1", "x2", "y2"]].to_numpy())
        xy, ok = image_to_pitch(G, uv, margin=20.0)
        err = np.hypot(xy[:, 0] - g["pitch_x"].to_numpy(), xy[:, 1] - g["pitch_y"].to_numpy())
        for k in range(len(g)):
            rows.append((seq.info.name, f, int(g["track_id"].iat[k]), cal.at[f, "source"], uv[k, 0], uv[k, 1],
                         float(g["pitch_y"].iat[k]), bool(ok[k]), err[k] if ok[k] else np.nan))
    return pd.DataFrame(rows, columns=["seq", "frame", "track_id", "source", "u", "v", "pitch_y", "valid", "err_m"])


def boot_median(per_clip: list[np.ndarray], n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    per_clip = [a for a in per_clip if len(a)]
    if not per_clip:
        return np.nan, np.nan, np.nan
    meds = [np.median(np.concatenate([per_clip[i] for i in rng.integers(0, len(per_clip), len(per_clip))]))
            for _ in range(n_boot)]
    return float(np.median(np.concatenate(per_clip))), *np.percentile(meds, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    calib_path = Path(args.calib)
    calib = pd.read_parquet(calib_path)
    run = json.loads(calib_path.with_suffix(".json").read_text(encoding="utf-8"))
    out = Path(args.out or f"reports/calibration_{calib_path.stem}.md")
    W = H = None
    errs = []
    for path in list_sequences(run["data_root"], run["split"]):
        if path.name not in run["clips"]:
            continue
        seq = load_sequence(path, run["split"])
        W, H = seq.info.width, seq.info.height
        errs.append(clip_errors(seq, calib[calib["seq"] == seq.info.name]))
    e = pd.concat(errs, ignore_index=True)
    good = e[e["valid"]]

    lines = []
    share = calib["source"].value_counts(normalize=True).reindex(["lines", "propagated", "global", "none"]).fillna(0.0)
    lines.append("| frames | from their lines | propagated | global | not calibrated |\n|---|---|---|---|---|")
    lines.append(f"| {len(calib)} | {share['lines']:.1%} | {share['propagated']:.1%} | {share['global']:.1%} | "
                 f"{share['none']:.1%} |\n")
    fails = calib.loc[~calib["ok"], "reason"].str.split("; ").explode().value_counts().head(6)
    if len(fails):
        lines.append("Reasons of failure (a frame may have several):\n")
        lines += [f"- {r}: {c} frames" for r, c in fails.items()]
        lines.append("")
    lines.append("| subset | players | median error (m) [95 % CI] | p90 (m) | invalid projections |\n|---|---|---|---|---|")
    subsets = [("all", good, e)]
    subsets += [(f"source = {s}", good[good["source"] == s], e[e["source"] == s]) for s in ("lines", "propagated", "global")]
    for name, lo, hi in (("far side (y < -11)", -99, -11), ("middle (-11 <= y < 11)", -11, 11), ("near side (y >= 11)", 11, 99)):
        m = (e["pitch_y"] >= lo) & (e["pitch_y"] < hi)
        subsets.append((name, good[m[e["valid"]].to_numpy()], e[m]))
    for name, g, allrows in subsets:
        if len(g) == 0:
            continue
        med, lo_ci, hi_ci = boot_median([g[g["seq"] == s]["err_m"].to_numpy() for s in g["seq"].unique()])
        lines.append(f"| {name} | {len(g)} | {med:.2f} [{lo_ci:.2f}, {hi_ci:.2f}] | {g['err_m'].quantile(0.9):.2f} | "
                     f"{1 - allrows['valid'].mean():.1%} |")
    res = calib.loc[calib["source"] == "lines", ["median_px", "p90_px", "circle_m"]].median()
    lines.append(f"\nLine points of the frames calibrated from their lines: median residual {res['median_px']:.2f} px, "
                 f"p90 {res['p90_px']:.2f} px; circle points: {res['circle_m']:.2f} m from their circle (median).\n")
    jump = good.sort_values("frame").groupby(["seq", "track_id"])["err_m"].diff().abs()
    lines.append(f"Temporal stability: median frame-to-frame change of the error of the same player "
                 f"{jump.median():.3f} m (p90 {jump.quantile(0.9):.3f} m).\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    nx, ny = 16, 9
    grid = np.full((ny, nx), np.nan)
    if W:
        ix = np.clip((good["u"] / W * nx).astype(int), 0, nx - 1)
        iy = np.clip((good["v"] / H * ny).astype(int), 0, ny - 1)
        for (j, i), v in good.groupby([iy, ix])["err_m"].median().items():
            grid[j, i] = v
    im = ax[0].imshow(grid, cmap="magma", extent=(0, W or 1, H or 1, 0))
    fig.colorbar(im, ax=ax[0], label="median error (m)")
    ax[0].set(title="error over the image (foot point)")
    bins = np.arange(-35, 36, 5)
    mids = 0.5 * (bins[1:] + bins[:-1])
    med = [good[(good["pitch_y"] >= a) & (good["pitch_y"] < b)]["err_m"].median() for a, b in zip(bins[:-1], bins[1:])]
    ax[1].plot(mids, med, "o-")
    ax[1].set(xlabel="pitch y of the player (m, far side < 0)", ylabel="median error (m)", title="error vs depth")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=130)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Calibration report -- {calib_path.name}\n\nSplit {run['split']}, {len(run['clips'])} clips, "
                f"propagation {'on' if run['propagate'] else 'off'}{', one homography per clip' if run.get('global') else ''}. Error = distance between the projection of the "
                f"bottom-middle of the GT box and its GT position bbox_pitch.\n\n")
        f.write("\n".join(lines) + f"\n![error maps]({out.with_suffix('.png').name})\n")
    print("\n".join(lines[:2]) + f"\nmedian error {good['err_m'].median():.2f} m over {len(good)} players -> {out}")


if __name__ == "__main__":
    main()
