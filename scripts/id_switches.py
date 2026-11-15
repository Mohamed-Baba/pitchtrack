"""List the identity switches of a tracker, to look at them one by one.

    python scripts/id_switches.py --tracks outputs/trk/valid_base.parquet
    python scripts/make_overlay.py --clip SNGS-XXX --tracks outputs/trk/valid_base.parquet --frame 312 --out outputs/sw.png

At each frame, ground truth and tracks are matched optimally (IoU >= 0.5). A switch is a
ground-truth player whose matched track id differs from the last id it was matched to.
Classify each switch by looking at the frames: crossing, occlusion, image border, camera cut.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from pitchtrack.boxes import box_iou
from pitchtrack.pipeline import frames_to_process
from pitchtrack.soccernet import list_sequences, load_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path = Path(args.tracks)
    tracks = pd.read_parquet(path)
    run = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    cols = ["x1", "y1", "x2", "y2"]
    events = []
    for p in list_sequences(run["data_root"], run["split"]):
        if p.name not in run["clips"]:
            continue
        seq = load_sequence(p, run["split"])
        tg = dict(tuple(tracks[tracks["seq"] == seq.info.name].groupby("frame")))
        gg = dict(tuple(seq.gt.groupby("frame")))
        last = {}
        for f in frames_to_process(seq, run["stride"]):
            g, t = gg.get(int(f)), tg.get(int(f))
            if g is None or t is None:
                continue
            iou = box_iou(g[cols].to_numpy(), t[cols].to_numpy())
            r, c = linear_sum_assignment(-iou)
            for i, j in zip(r, c):
                if iou[i, j] < 0.5:
                    continue
                gid, tid = int(g["track_id"].iloc[i]), int(t["track_id"].iloc[j])
                if gid in last and last[gid] != tid:
                    x1, y1, x2, y2 = g[cols].to_numpy()[i]
                    events.append({"clip": seq.info.name, "frame": int(f), "gt_id": gid, "old_id": last[gid],
                                   "new_id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                   "role": g["role"].iloc[i]})
                last[gid] = tid
    out = Path(args.out or f"reports/id_switches_{path.stem}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(events)
    df.to_csv(out, index=False)
    print(f"{len(df)} identity switches -> {out}")
    if len(df):
        print(df.groupby("clip").size().sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
