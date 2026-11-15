"""Draw ground truth (green), detections (red) and the pitch mask (yellow) on a clip.

    python scripts/make_overlay.py --clip SNGS-XXX --out outputs/gt.mp4             # ground truth only
    python scripts/make_overlay.py --clip SNGS-XXX --dets outputs/det/try.parquet --out outputs/try.mp4
    python scripts/make_overlay.py --clip SNGS-XXX --mask --frame 0 --out outputs/mask0.png
    python scripts/make_overlay.py --clip SNGS-XXX --tracks outputs/trk/valid_base.parquet --out outputs/trk.mp4
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pitchtrack.pipeline import frames_to_process
from pitchtrack.pitch_mask import pitch_mask
from pitchtrack.soccernet import load_sequence
from pitchtrack.video_io import VideoWriter
from pitchtrack.visualization import GREEN, RED, draw_boxes, draw_mask_outline, put_header


def id_color(i: int):
    rng = np.random.default_rng(int(i) * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def render(seq, f, dets, tau, with_mask, tracks=None):
    img = seq.read_frame(f)
    gt = seq.gt_frame(f)
    if tracks is not None:   # tracks mode: one colour per track id, ground truth in thin green
        img = draw_boxes(img, gt[["x1", "y1", "x2", "y2"]].to_numpy(), GREEN, None, 1)
        tr = tracks[(tracks["seq"] == seq.info.name) & (tracks["frame"] == f)]
        for _, r in tr.iterrows():
            img = draw_boxes(img, [[r.x1, r.y1, r.x2, r.y2]], id_color(r.track_id), [int(r.track_id)], 2)
        return put_header(img, f"{seq.info.name} frame {f}  t={f / seq.info.fps:.2f}s  "
                               f"{len(tr)} tracks (colour = id), GT {len(gt)} (thin green)")
    img = draw_boxes(img, gt[["x1", "y1", "x2", "y2"]].to_numpy(), GREEN, gt["track_id"].tolist(), 2)
    n_det = 0
    if dets is not None:
        d = dets[(dets["seq"] == seq.info.name) & (dets["frame"] == f) & (dets["score"] >= tau)]
        n_det = len(d)
        img = draw_boxes(img, d[["x1", "y1", "x2", "y2"]].to_numpy(), RED,
                         [f"{s:.2f}" for s in d["score"]], 1)
    if with_mask:
        img = draw_mask_outline(img, pitch_mask(img))
    return put_header(img, f"{seq.info.name} frame {f}  t={f / seq.info.fps:.2f}s  "
                           f"GT {len(gt)} (green)  det {n_det} (red, score >= {tau:g})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/SoccerNetGS")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--dets", default=None)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--mask", action="store_true")
    ap.add_argument("--tracks", default=None, help="tracks table of run_tracker.py (draws ids)")
    ap.add_argument("--frame", type=int, default=None, help="render one frame to an image")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seq = load_sequence(Path(args.root) / args.split / args.clip, args.split)
    dets = pd.read_parquet(args.dets) if args.dets else None
    tracks = pd.read_parquet(args.tracks) if args.tracks else None
    if args.frame is not None:
        cv2.imwrite(args.out, render(seq, args.frame, dets, args.tau, args.mask, tracks))
        return
    stride = 1
    source = args.dets or args.tracks
    if source is not None:  # only the frames that went through the detector or the tracker
        stride = json.loads(Path(source).with_suffix(".json").read_text(encoding="utf-8"))["stride"]
    frames, fps = frames_to_process(seq, stride).tolist(), seq.info.fps / stride
    with VideoWriter(args.out, fps, (seq.info.width, seq.info.height)) as w:
        for f in frames:
            w.write(render(seq, int(f), dets, args.tau, args.mask, tracks))
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
