"""Run the tracker on cached detections -> tracks table (.parquet + .json).

    python scripts/run_detector.py --split valid --stride 1 --out outputs/det/valid_s1.parquet   # once
    python scripts/run_tracker.py --dets outputs/det/valid_s1.parquet --out outputs/trk/valid_base.parquet
    options: --oracle (ground-truth boxes as detections), --cmc, --motion none, --single-stage,
             --limit 2, --config configs/tracker.yaml

Track on EVERY frame (stride 1): with a stride, fast players jump further than their own
width between two frames.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pitchtrack.camera_motion import CameraMotionEstimator
from pitchtrack.kalman import KalmanBoxFilter
from pitchtrack.pipeline import detections_per_frame, frames_to_process, gt_per_frame
from pitchtrack.soccernet import list_sequences, load_sequence
from pitchtrack.tracker import Tracker, TrackerConfig, track_frames

COLUMNS = ["seq", "frame", "t", "track_id", "x1", "y1", "x2", "y2", "score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True, help="detections cached by run_detector.py (stride 1)")
    ap.add_argument("--config", default="configs/tracker.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--oracle", action="store_true", help="use the ground-truth boxes as detections")
    ap.add_argument("--cmc", action="store_true", help="enable camera motion compensation")
    ap.add_argument("--motion", choices=["kalman", "none"], default=None)
    ap.add_argument("--single-stage", action="store_true", help="disable the ByteTrack second stage")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    tcfg = TrackerConfig.from_dict(cfg.get("tracker"))
    if args.motion:
        tcfg.motion = args.motion
    if args.single_stage:
        tcfg.two_stage = False
    cmc_cfg = dict(cfg.get("cmc") or {})
    use_cmc = args.cmc or cmc_cfg.pop("enabled", False)
    cmc_cfg.pop("enabled", None)

    dets_path = Path(args.dets)
    table = pd.read_parquet(dets_path)
    info = json.loads(dets_path.with_suffix(".json").read_text(encoding="utf-8"))
    root, split, stride = info["config"]["data"]["root"], info["split"], info["stride"]
    if stride != 1 and not args.oracle:
        print(f"WARNING: detections with stride {stride}: the tracker should see every frame")

    rows, n_frames, elapsed = [], 0, 0.0
    clips = info["clips"][: args.limit] if args.limit else info["clips"]
    for path in list_sequences(root, split):
        if path.name not in clips:
            continue
        seq = load_sequence(path, split)
        frames = frames_to_process(seq, 1 if args.oracle else stride)
        per_frame = gt_per_frame(seq, frames) if args.oracle else detections_per_frame(
            table, seq.info.name, frames, seq.info.fps, tcfg.low_thresh, cfg.get("pitch_radius"))
        motion = None
        if use_cmc:
            est = CameraMotionEstimator(**cmc_cfg)
            motion = {f: est(seq.read_frame(f), d.boxes) for f, _, d in per_frame}
        tracker = Tracker(tcfg, KalmanBoxFilter(**(cfg.get("kalman") or {})))
        tic = time.perf_counter()
        for f, t, ids, boxes, scores in track_frames(tracker, per_frame, motion):
            rows += [(seq.info.name, f, t, int(i), *b, s) for i, b, s in zip(ids, boxes, scores)]
        elapsed += time.perf_counter() - tic
        n_frames += len(per_frame)
        print(f"{seq.info.name}: {len(per_frame)} frames, {tracker.next_id - 1} track ids")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_parquet(out, index=False)
    run = {"split": split, "clips": clips, "stride": 1 if args.oracle else stride, "oracle": args.oracle,
           "cmc": bool(use_cmc), "tracker": vars(tcfg), "kalman": cfg.get("kalman"), "data_root": root,
           "detections": str(dets_path), "tracking_ms_per_frame": 1000 * elapsed / max(1, n_frames)}
    out.with_suffix(".json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    print(f"{len(rows)} track boxes, {run['tracking_ms_per_frame']:.2f} ms/frame (tracking only) -> {out}")


if __name__ == "__main__":
    main()
