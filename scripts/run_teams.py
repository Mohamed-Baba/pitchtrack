"""Team of every track from shirt colours -> parquet (one row per sampled detection).

    python scripts/run_teams.py --tracks outputs/trk/v_byte_cmc.parquet --out outputs/teams/valid.parquet
    python scripts/run_teams.py --oracle --split valid --out outputs/teams/valid_oracle.parquet

Per clip: one frame in `stride`, torso colour of every box at least min_height tall,
k-means on the colours of the clip (the two largest clusters are the teams), then the team
of each detection alone (team_frame) and of each track (team_track). --oracle uses the
ground-truth boxes and ids, and keeps their role and team for eval_teams.py.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pitchtrack.pipeline import load_config
from pitchtrack.soccernet import list_sequences, load_sequence
from pitchtrack.team_classifier import UNKNOWN, assign_teams, fit_team_colours, torso_colour


def teams_of_clip(seq, boxes: pd.DataFrame, cfg) -> pd.DataFrame:
    """boxes: rows (frame, track_id, x1, y1, x2, y2, ...) of one clip -> the same rows sampled,
    with L, a, b, valid, team_frame, team_track (0, 1 or UNKNOWN = -1)."""
    keep = boxes["frame"] % cfg["stride"] == 0
    rows = boxes[keep].sort_values("frame", kind="mergesort").reset_index(drop=True)
    feats, valid = np.full((len(rows), 3), np.nan), np.zeros(len(rows), bool)
    xyxy = rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
    for f, idx in rows.groupby("frame").indices.items():
        tall = idx[(xyxy[idx, 3] - xyxy[idx, 1]) >= cfg["min_height"]]
        if len(tall):
            feats[tall], valid[tall] = torso_colour(seq.read_frame(int(f)), xyxy[tall], cfg["min_pixels"],
                                                    cfg["hsv_low"], cfg["hsv_high"])
    rows[["L", "a", "b"]] = feats
    rows["valid"] = valid
    rows["team_frame"] = rows["team_track"] = UNKNOWN
    if valid.sum() >= max(cfg["k"], 10):
        centres, teams, _ = fit_team_colours(feats[valid], cfg["k"], cfg["l_weight"])
        common = dict(centres=centres, team_clusters=teams, margin=cfg["margin"], max_dist=cfg["max_dist"],
                      min_frames=cfg["min_frames"], l_weight=cfg["l_weight"])
        tid = rows["track_id"].to_numpy()
        rows["team_frame"] = assign_teams(feats, valid, tid, level="frame", **common)
        rows["team_track"] = assign_teams(feats, valid, tid, level="track", **common)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/analysis.yaml")
    ap.add_argument("--tracks", default=None, help="tracks table of run_tracker.py")
    ap.add_argument("--oracle", action="store_true", help="ground-truth boxes and ids instead of tracks")
    ap.add_argument("--data-config", default="configs/baseline.yaml")
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.oracle and not args.tracks:
        ap.error("give --tracks or --oracle")
    cfg = load_config(args.config)["teams"]
    if args.tracks:
        tracks = pd.read_parquet(args.tracks)
        run = json.loads(Path(args.tracks).with_suffix(".json").read_text(encoding="utf-8"))
        root, split, clips = run["data_root"], run["split"], set(run["clips"])
    else:
        root = load_config(args.data_config)["data"]["root"]
        split, clips = args.split or "valid", None
    t0, out_rows = time.perf_counter(), []
    for path in list_sequences(root, split)[: args.limit]:
        if clips is not None and path.name not in clips:
            continue
        seq = load_sequence(path, split)
        if args.oracle:
            boxes = seq.gt[["frame", "track_id", "x1", "y1", "x2", "y2", "role", "team"]]
        else:
            boxes = tracks[tracks["seq"] == seq.info.name][["frame", "track_id", "x1", "y1", "x2", "y2"]]
        rows = teams_of_clip(seq, boxes.reset_index(drop=True), cfg)
        rows.insert(0, "seq", seq.info.name)
        out_rows.append(rows)
        per_track = rows.groupby("track_id")["team_track"].first()
        print(f"{seq.info.name}: {len(per_track)} tracks, teams {np.bincount(per_track + 1, minlength=3)[1:]}, "
              f"unknown {(per_track == UNKNOWN).mean():.0%}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.concat(out_rows, ignore_index=True)
    table.to_parquet(out, index=False)
    info = {"split": split, "data_root": str(root), "oracle": args.oracle, "tracks": args.tracks,
            "clips": sorted(table["seq"].unique().tolist()), "teams": cfg, "seconds": time.perf_counter() - t0}
    out.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"{len(table)} detections -> {out}")


if __name__ == "__main__":
    main()
