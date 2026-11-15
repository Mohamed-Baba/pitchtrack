"""Evaluate cached detections against SoccerNet-GSR -> markdown report + figures.

    python scripts/eval_detection.py --dets outputs/det/valid_s5.parquet [--pitch-radius 20] [--tau 0.3]

The report goes to reports/detection_<dets name>[_r<radius>][_t<tau>].md unless --out is given.

Reports AP50 and AP@[.5:.95], the operating curve (precision, recall, DetA versus the
score threshold), recall by box height, and 95 % intervals obtained by resampling CLIPS.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pitchtrack.metrics import (cluster_bootstrap_ratio, coco_ap, operating_curve, pool_matches,  # noqa: E402
                                precision_recall, recall_by_height)
from pitchtrack.pipeline import frames_to_process, gt_det_frames  # noqa: E402
from pitchtrack.soccernet import list_sequences, load_sequence  # noqa: E402


def load_gt(root, split, clips, stride):
    """{clip name: (ground truth of the evaluated frames, evaluated frames)}."""
    out = {}
    for path in list_sequences(root, split):
        if path.name in clips:
            seq = load_sequence(path, split)
            frames = frames_to_process(seq, stride)
            out[path.name] = (seq.gt[seq.gt["frame"].isin(frames)], frames)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--out", default=None, help="default: reports/detection_<dets name>....md")
    ap.add_argument("--pitch-radius", type=float, default=None, help="drop detections farther (px) from the pitch")
    ap.add_argument("--tau", type=float, default=None, help="operating threshold (default: argmax DetA)")
    ap.add_argument("--iou", type=float, default=None)
    args = ap.parse_args()

    dets = pd.read_parquet(args.dets)
    info = json.loads(Path(args.dets).with_suffix(".json").read_text(encoding="utf-8"))
    cfg = info["config"]
    ev = cfg.get("eval", {})
    iou_thr = args.iou or ev.get("iou_threshold", 0.5)
    bins = ev.get("height_bins", [0, 40, 60, 90, 140, 10000])

    # every processed frame counts, even a frame without any detection
    gt = load_gt(cfg["data"]["root"], info["split"], set(info["clips"]), info["stride"])
    frames = list(gt_det_frames(gt, dets, pitch_radius=args.pitch_radius))
    aps = coco_ap(frames)
    m = pool_matches(frames, iou_thr)
    taus = np.round(np.arange(0.05, 0.96, 0.05), 2)
    curve = operating_curve(m.scores, m.is_tp, m.n_gt, taus)
    tau = args.tau if args.tau is not None else float(curve.loc[curve["det_a"].idxmax(), "threshold"])
    by_h = recall_by_height(m.gt_heights, m.gt_best_score, bins, tau)

    # clip-level intervals: resample clips, not frames (consecutive frames are correlated)
    n_clips = len(gt)
    tp_c = np.array([((m.det_clip == c) & m.is_tp & (m.scores >= tau)).sum() for c in range(n_clips)])
    fp_c = np.array([((m.det_clip == c) & ~m.is_tp & (m.scores >= tau)).sum() for c in range(n_clips)])
    gt_c = np.array([(m.gt_clip == c).sum() for c in range(n_clips)])
    rec = cluster_bootstrap_ratio(tp_c, gt_c)
    prec = cluster_bootstrap_ratio(tp_c, tp_c + fp_c)
    deta = cluster_bootstrap_ratio(tp_c, gt_c + fp_c)

    if args.out is None:
        suffix = ("" if args.pitch_radius is None else f"_r{args.pitch_radius:g}") + \
                 ("" if args.tau is None else f"_t{args.tau:g}")
        args.out = f"reports/detection_{Path(args.dets).stem}{suffix}.md"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    p, r = precision_recall(m.scores, m.is_tp, m.n_gt)
    axes[0].plot(r, p)
    axes[0].set(xlabel="recall", ylabel="precision", title=f"PR curve, IoU {iou_thr}", xlim=(0, 1), ylim=(0, 1.02))
    for col in ("precision", "recall", "det_a"):
        axes[1].plot(curve["threshold"], curve[col], label=col)
    axes[1].axvline(tau, color="gray", ls="--")
    axes[1].set(xlabel="score threshold", title="operating curve")
    axes[1].legend()
    axes[2].bar(by_h["height_bin"], by_h["recall"])
    axes[2].set(ylabel="recall", title=f"recall by box height (tau = {tau:g})", ylim=(0, 1))
    axes[2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=130)

    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Detection report -- {Path(args.dets).name}\n\n")
        f.write(f"Split {info['split']}, stride {info['stride']}, {info['n_frames']} frames, {n_clips} clips, "
                f"{m.n_gt} ground-truth boxes. Detector: {cfg['detector']['weights']} at imgsz "
                f"{cfg['detector']['imgsz']}, tiling {cfg.get('tiling', {}).get('enabled', False)}, "
                f"pitch radius {args.pitch_radius}. Latency {info['latency_ms_per_frame']:.1f} ms/frame.\n\n")
        f.write("| AP50 | AP75 | AP@[.5:.95] |\n|---|---|---|\n"
                f"| {aps['AP50']:.3f} | {aps['AP75']:.3f} | {aps['AP']:.3f} |\n\n")
        f.write(f"Operating point tau = {tau:g} (IoU {iou_thr}); 95 % CI by resampling clips:\n\n"
                "| precision | recall | DetA |\n|---|---|---|\n"
                f"| {prec[0]:.3f} [{prec[1]:.3f}, {prec[2]:.3f}] | {rec[0]:.3f} [{rec[1]:.3f}, {rec[2]:.3f}] "
                f"| {deta[0]:.3f} [{deta[1]:.3f}, {deta[2]:.3f}] |\n\n")
        f.write("## Recall by box height\n\n" + by_h.to_markdown(index=False, floatfmt=".3f") + "\n\n")
        f.write("## Operating curve\n\n" + curve.to_markdown(index=False, floatfmt=".3f") + "\n\n")
        f.write(f"![figures]({out.with_suffix('.png').name})\n")
    print(f"AP50 {aps['AP50']:.3f} | AP {aps['AP']:.3f} | tau {tau:g}: P {prec[0]:.3f} R {rec[0]:.3f} "
          f"DetA {deta[0]:.3f} -> {out}")


if __name__ == "__main__":
    main()
