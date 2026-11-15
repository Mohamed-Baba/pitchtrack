"""Annotated video of a SoccerNet-GSR clip: boxes, ids and teams, the pitch lines projected
with the calibration of each frame, and a top view of the positions in the corner.

    python scripts/make_final_video.py --clip SNGS-XXX --stats outputs/stats/valid --calib outputs/calib/valid.parquet \
        --out outputs/final_SNGS-XXX.mp4
    ... --frame 120 --out outputs/final.png      # one frame

--stats is an output folder of run_statistics.py (positions.parquet: boxes, ids, teams, X, Y).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pitchtrack.soccernet import load_sequence
from pitchtrack.video_io import VideoWriter
from pitchtrack.visualization import TEAM_COLORS, draw_boxes, draw_pitch_lines, minimap, put_header


def render(seq, f, pos, cal):
    img = seq.read_frame(f)
    c = cal[cal["frame"] == f]
    calibrated = len(c) and bool(c["ok"].iat[0])
    if calibrated:
        G = c[[f"g{i}" for i in range(9)]].to_numpy(dtype=np.float64).reshape(3, 3)
        img = draw_pitch_lines(img, np.linalg.inv(G))
    p = pos[pos["frame"] == f]
    for _, r in p.iterrows():
        img = draw_boxes(img, [[r.x1, r.y1, r.x2, r.y2]], TEAM_COLORS.get(int(r.team), TEAM_COLORS[-1]),
                         [int(r.track_id)], 2)
    mm = minimap(p[["X", "Y"]].to_numpy(), p["team"].to_numpy(), p["track_id"].to_numpy(), scale=3.0)
    h, w = mm.shape[:2]
    H, W = img.shape[:2]
    img[H - h - 8:H - 8, W - w - 8:W - 8] = mm
    src = c["source"].iat[0] if len(c) else "none"
    return put_header(img, f"{seq.info.name} frame {f}  t={f / seq.info.fps:.2f}s  calibration: {src}  "
                           f"{len(p)} tracks (red / blue = teams, grey = unknown)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = json.loads(Path(args.calib).with_suffix(".json").read_text(encoding="utf-8"))
    seq = load_sequence(Path(run["data_root"]) / run["split"] / args.clip, run["split"])
    pos = pd.read_parquet(Path(args.stats) / "positions.parquet")
    pos = pos[pos["seq"] == args.clip]
    cal = pd.read_parquet(args.calib)
    cal = cal[cal["seq"] == args.clip]
    if args.frame is not None:
        cv2.imwrite(args.out, render(seq, args.frame, pos, cal))
        return
    with VideoWriter(args.out, seq.info.fps, (seq.info.width, seq.info.height)) as w:
        for f in range(seq.info.n_frames):
            w.write(render(seq, f, pos, cal))
    print(f"wrote {args.out} ({seq.info.n_frames} frames)")


if __name__ == "__main__":
    main()
