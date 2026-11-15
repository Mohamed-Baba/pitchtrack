"""Image -> pitch homography for every frame, from the pitch lines -> parquet.

    python scripts/calibrate.py --split valid --out outputs/calib/valid.parquet
    python scripts/calibrate.py --split valid --no-propagate --out outputs/calib/valid_lines.parquet
    python scripts/calibrate.py --split valid --global --out outputs/calib/valid_global.parquet
    python scripts/calibrate.py --split valid --line-model models/pitch_lines.pt --out outputs/calib/valid_model.parquet

For each frame with lines: RANSAC on the lines and the quality checks of calibrate_frame. The
lines are the annotations of SoccerNet-GSR, or, with --line-model, the lines predicted by the
network (the same evaluation then measures the network against the annotations).
Frames that fail (too few lines, a midfield view with only the touch line and the halfway
line...) can receive the homography of a neighbouring frame moved with the image motion
(pitch_calibration.ImageMotion, forward then backward, at most propagate_max_s seconds away);
if the frame has annotated points, the moved homography must still fit them. The table keeps the
source of every estimate ("lines", "propagated", "none") and the reason of each failure.
Players are kept out of the image motion with the cached detections (--dets) if given.
--global is a baseline: the first homography of each clip estimated from its
lines is used for every frame of the clip, as if the camera did not move.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pitchtrack.pipeline import load_config
from pitchtrack.pitch_calibration import (Calibration, ImageMotion, calibrate_frame, line_residuals,
                                          parse_lines, propagate)
from pitchtrack.soccernet import list_sequences, load_sequence

COLUMNS = ["seq", "frame", "t", "ok", "source", "reason", "n_points", "n_inliers", "n_xlines", "n_ylines",
           "median_px", "p90_px", "circle_m", "hull_frac"] + [f"g{i}" for i in range(9)]


def check_propagated(G, lines, size, cfg) -> tuple[bool, float]:
    """A moved homography is kept if it still fits the frame's own points (when it has some)."""
    lp = parse_lines(lines, *size) if lines else None
    if lp is None or len(lp.pts) == 0:
        return True, np.nan
    med = float(np.median(line_residuals(G, lp.pts, lp.lines)))
    return med <= cfg["propagated_max_median_px"], med


def predict_lines(seq, annotator, batch: int = 8) -> dict:
    """Pitch lines of every frame of a clip predicted by the line network (frame -> annotation)."""
    out = {}
    for start in range(0, seq.info.n_frames, batch):
        frames = list(range(start, min(start + batch, seq.info.n_frames)))
        out.update(zip(frames, annotator([seq.read_frame(f) for f in frames])))
    return out


def calibrate_sequence(seq, cfg, propagate_on=True, boxes_of=None, global_h=False, lines=None):
    """lines: frame -> annotation; the annotated lines of the clip (seq.lines) by default."""
    lines = seq.lines if lines is None else lines
    n, size = seq.info.n_frames, (seq.info.width, seq.info.height)
    t = np.arange(n) / seq.info.fps
    cals = [calibrate_frame(lines[f], size, cfg, seed=f) if f in lines
            else Calibration(None, False, "none", "no annotated lines") for f in range(n)]
    if global_h:
        first = next((c for c in cals if c.ok), None)
        cals = [Calibration(first.G, True, "global") if first is not None else c for c in cals]
        propagate_on = False
    if propagate_on and not all(c.ok for c in cals):
        motion, Ms, oks = ImageMotion(scale=cfg["motion_scale"]), [], []
        for f in range(n):                      # Ms[f]: p_f ~ Ms[f] p_(f-1), oks[f] False on a cut
            M, ok = motion(seq.read_frame(f), None if boxes_of is None else boxes_of(f))
            Ms.append(M)
            oks.append(ok and f > 0)
        for direction in (+1, -1):
            G, anchor_t = None, None
            for f in (range(n) if direction > 0 else range(n - 1, -1, -1)):
                if cals[f].ok and cals[f].source == "lines":
                    G, anchor_t = cals[f].G, t[f]
                    continue
                if G is None:
                    continue
                link = f if direction > 0 else f + 1          # motion between f and the frame before it
                if not oks[link] or abs(t[f] - anchor_t) > cfg["propagate_max_s"]:
                    G = None
                    continue
                G = propagate(G, Ms[link] if direction > 0 else np.linalg.inv(Ms[link]))
                if cals[f].ok:                                # already filled by the other direction
                    continue
                good, med = check_propagated(G, lines.get(f), size, cfg)
                if good:
                    cals[f] = Calibration(G, True, "propagated", "", median_px=med)
                else:
                    cals[f].reason = (cals[f].reason + "; " if cals[f].reason else "") + "moved homography does not fit"
                    G = None
    rows = []
    for f, c in enumerate(cals):
        g = c.G.reshape(-1) if c.G is not None and c.ok else np.full(9, np.nan)
        rows.append([seq.info.name, f, t[f], bool(c.ok), c.source if c.ok else "none", c.reason, c.n_points,
                     c.n_inliers, c.n_xlines, c.n_ylines, c.median_px, c.p90_px, c.circle_m, c.hull_frac, *g])
    return pd.DataFrame(rows, columns=COLUMNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/analysis.yaml")
    ap.add_argument("--data-config", default="configs/baseline.yaml")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--dets", default=None, help="detections (run_detector.py) to mask the players")
    ap.add_argument("--no-propagate", action="store_true")
    ap.add_argument("--global", dest="global_h", action="store_true", help="one homography per clip (baseline)")
    ap.add_argument("--line-model", default=None, help="lines predicted by this network instead of the annotations")
    ap.add_argument("--device", default=None, help="device of the line network")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)["calibration"]
    root = load_config(args.data_config)["data"]["root"]
    dets = pd.read_parquet(args.dets) if args.dets else None
    annotator = None
    if args.line_model:
        from pitchtrack.line_model import LineAnnotator
        annotator = LineAnnotator(args.line_model, device=args.device)
    t0, tables = time.perf_counter(), []
    for path in list_sequences(root, args.split)[: args.limit]:
        seq = load_sequence(path, args.split)
        boxes_of = None
        if dets is not None:
            d = dets[dets["seq"] == seq.info.name]
            by_frame = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in d.groupby("frame")}
            boxes_of = lambda f, by_frame=by_frame: by_frame.get(f)
        lines = predict_lines(seq, annotator) if annotator is not None else None
        table = calibrate_sequence(seq, cfg, not args.no_propagate, boxes_of, args.global_h, lines)
        tables.append(table)
        counts = table["source"].value_counts().to_dict()
        print(f"{seq.info.name}: {counts}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.concat(tables, ignore_index=True)
    table.to_parquet(out, index=False)
    info = {"split": args.split, "data_root": str(root), "clips": sorted(table["seq"].unique().tolist()),
            "propagate": not args.no_propagate and not args.global_h, "global": args.global_h,
            "lines": args.line_model or "annotations",
            "calibration": cfg, "seconds": time.perf_counter() - t0}
    out.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"{len(table)} frames, {table['ok'].mean():.1%} calibrated "
          f"({(table['source'] == 'lines').mean():.1%} from their lines) -> {out}")


if __name__ == "__main__":
    main()
