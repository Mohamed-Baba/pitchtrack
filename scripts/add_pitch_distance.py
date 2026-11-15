"""Add (or recompute) the distance of every cached detection to the pitch mask.

    python scripts/add_pitch_distance.py --dets outputs/det/valid_s5.parquet

Reads each processed frame once, computes pitch_mask with the `pitch_mask` section of
the config, and rewrites the pitch_dist column (pixels, 0 on the pitch). The detector is not
run again: any tolerance radius can then be evaluated offline.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitchtrack.pitch_mask import foot_distance_to_pitch, pitch_mask
from pitchtrack.soccernet import list_sequences, load_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--config", default=None, help="take the pitch_mask section from this config instead")
    args = ap.parse_args()
    dets_path = Path(args.dets)
    dets = pd.read_parquet(dets_path)
    info = json.loads(dets_path.with_suffix(".json").read_text(encoding="utf-8"))
    cfg = info["config"]
    if args.config:
        import yaml
        cfg["pitch_mask"] = yaml.safe_load(open(args.config, encoding="utf-8"))["pitch_mask"]
    pm = cfg.get("pitch_mask", {})
    dist = np.full(len(dets), np.nan)
    for path in list_sequences(cfg["data"]["root"], info["split"]):
        if path.name not in info["clips"]:
            continue
        seq = load_sequence(path, info["split"])
        sel = dets["seq"] == seq.info.name
        for frame, rows in dets[sel].groupby("frame"):
            mask = pitch_mask(seq.read_frame(int(frame)), pm.get("hsv_low", (35, 40, 40)),
                              pm.get("hsv_high", (85, 255, 255)), pm.get("close_px", 25), pm.get("scale", 0.5))
            dist[rows.index] = foot_distance_to_pitch(rows[["x1", "y1", "x2", "y2"]].to_numpy(), mask)
        print(f"{seq.info.name}: done")
    dets["pitch_dist"] = dist
    dets.to_parquet(dets_path, index=False)
    info["config"]["pitch_mask"] = {**pm, "enabled": True}
    dets_path.with_suffix(".json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print(f"pitch_dist written for {np.isfinite(dist).sum()} / {len(dets)} detections -> {dets_path}")


if __name__ == "__main__":
    main()
