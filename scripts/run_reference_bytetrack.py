"""The reference ByteTrack implementation (Ultralytics) on the SAME cached detections.

    python scripts/run_reference_bytetrack.py --dets outputs/det/valid_s1.parquet --out outputs/trk/valid_ref.parquet

A fair comparison changes only the tracker: same detections, same frames, same metric. Thresholds are taken from configs/tracker.yaml; track_buffer is max_lost_s
converted to frames. Needs `pip install lap` (Ultralytics asks for it).
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from pitchtrack.pipeline import detections_per_frame, frames_to_process
from pitchtrack.soccernet import list_sequences, load_sequence


class _Boxes:
    """The minimal Ultralytics `Boxes` interface that BYTETracker.update reads."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, float).reshape(-1, 4)
        self.conf, self.cls = np.asarray(conf, float), np.asarray(cls, float)
        self.xywh = np.concatenate([(self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2, self.xyxy[:, 2:] - self.xyxy[:, :2]], 1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _Boxes(self.xyxy[idx], self.conf[idx], self.cls[idx])


def main():
    from ultralytics.trackers.byte_tracker import BYTETracker

    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--config", default="configs/tracker.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    t = cfg["tracker"]
    dets_path = Path(args.dets)
    table = pd.read_parquet(dets_path)
    info = json.loads(dets_path.with_suffix(".json").read_text(encoding="utf-8"))
    root, split, stride = info["config"]["data"]["root"], info["split"], info["stride"]
    clips = info["clips"][: args.limit] if args.limit else info["clips"]
    rows = []
    for path in list_sequences(root, split):
        if path.name not in clips:
            continue
        seq = load_sequence(path, split)
        bt = BYTETracker(SimpleNamespace(
            track_high_thresh=t["high_thresh"], track_low_thresh=t["low_thresh"],
            new_track_thresh=t["new_track_thresh"], match_thresh=1.0 - t["iou_gate_high"],
            track_buffer=int(round(t["max_lost_s"] * seq.info.fps)), fuse_score=True))
        frames = frames_to_process(seq, stride)
        for f, tt, d in detections_per_frame(table, seq.info.name, frames, seq.info.fps, t["low_thresh"],
                                             cfg.get("pitch_radius")):
            res = bt.update(_Boxes(d.boxes, d.scores, d.classes))
            rows += [(seq.info.name, f, tt, int(r[4]), *r[:4], float(r[5])) for r in res]
        print(f"{seq.info.name}: done")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["seq", "frame", "t", "track_id", "x1", "y1", "x2", "y2", "score"]).to_parquet(out)
    run = {"split": split, "clips": clips, "stride": stride, "oracle": False, "cmc": False,
           "tracker": {"reference": "ultralytics BYTETracker", **t}, "data_root": root,
           "detections": str(dets_path), "tracking_ms_per_frame": float("nan")}
    out.with_suffix(".json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    print(f"{len(rows)} track boxes -> {out}")


if __name__ == "__main__":
    main()
