"""Evaluate the teams of run_teams.py -> markdown report.

    python scripts/eval_teams.py --teams outputs/teams/valid_oracle.parquet
    python scripts/eval_teams.py --teams outputs/teams/valid.parquet      # tracks of run_tracker.py

Per clip, the clusters 0/1 are mapped to "left"/"right" by the best permutation (on the track
decisions). Only boxes and tracks whose ground truth has a team (players, goalkeepers) are
scored: coverage (share not "unknown"), accuracy among the covered ones, macro-F1 over
left/right where "unknown" counts as an error. Rows: the frame-level rule judged per detection;
the track-level rule judged per detection (each detection gets the decision of its track)
and per track; a majority vote of the frame decisions, per track. Then the decisions per
role (a referee should be "unknown") and the confusion matrix. With tracks of run_tracker.py, each track
gets the ground-truth identity it overlaps most often (IoU >= 0.5, greedy per frame).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitchtrack.boxes import box_iou
from pitchtrack.soccernet import list_sequences, load_sequence
from pitchtrack.team_classifier import UNKNOWN, match_clusters_to_labels


def gt_identity(rows: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """Predicted track id -> ground-truth track id (majority of greedy IoU >= 0.5 matches)."""
    votes: dict = {}
    gt_by_frame = {f: g for f, g in gt.groupby("frame")}
    for f, g in rows.groupby("frame"):
        if f not in gt_by_frame:
            continue
        G = gt_by_frame[f]
        iou = box_iou(g[["x1", "y1", "x2", "y2"]].to_numpy(), G[["x1", "y1", "x2", "y2"]].to_numpy())
        while iou.size and iou.max() >= 0.5:
            i, j = np.unravel_index(np.argmax(iou), iou.shape)
            key = (int(g["track_id"].iat[i]), int(G["track_id"].iat[j]))
            votes[key] = votes.get(key, 0) + 1
            iou[i, :], iou[:, j] = -1, -1
    best: dict = {}
    for (p, q), n in votes.items():
        if n > best.get(p, (None, 0))[1]:
            best[p] = (q, n)
    return {p: q for p, (q, _) in best.items()}


def labelled(rows: pd.DataFrame, seq) -> pd.DataFrame:
    """Rows of one clip + gt_team and role of their track (oracle: already there)."""
    if "team" in rows.columns:
        return rows.rename(columns={"team": "gt_team"})
    ident = gt_identity(rows, seq.gt)
    roles = seq.gt.groupby("track_id").agg(gt_team=("team", "first"), role=("role", "first"))
    lab = pd.DataFrame({"track_id": list(ident), "gt_id": list(ident.values())}).join(roles, on="gt_id")
    return rows.merge(lab[["track_id", "gt_team", "role"]], on="track_id", how="left")


def scores(pred: pd.Series, gt: pd.Series) -> dict:
    has = gt.isin(["left", "right"])
    pred, gt = pred[has], gt[has]
    covered = pred != "unknown"
    f1 = []
    for c in ("left", "right"):
        tp = ((pred == c) & (gt == c)).sum()
        fp = ((pred == c) & (gt != c)).sum()
        fn = ((pred != c) & (gt == c)).sum()
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    return {"n": int(has.sum()), "coverage": covered.mean() if has.any() else np.nan,
            "accuracy": (pred[covered] == gt[covered]).mean() if covered.any() else np.nan,
            "f1": float(np.mean(f1))}


def majority(s: pd.Series) -> int:
    s = s[s != UNKNOWN]
    return int(s.mode().iat[0]) if len(s) else UNKNOWN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    path = Path(args.teams)
    table = pd.read_parquet(path)
    run = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    out = Path(args.out or f"reports/teams_{path.stem}.md")
    dets, tracks = [], []
    for clip in list_sequences(run["data_root"], run["split"]):
        if clip.name not in run["clips"]:
            continue
        seq = load_sequence(clip, run["split"])
        d = labelled(table[table["seq"] == seq.info.name], seq)
        t = d.groupby("track_id").agg(team_track=("team_track", "first"), vote=("team_frame", majority),
                                      gt_team=("gt_team", "first"), role=("role", "first")).reset_index()
        mapping = match_clusters_to_labels(t["team_track"].to_numpy(), t["gt_team"].to_numpy())
        name = lambda v: mapping.get(int(v), "unknown")
        for col in ("team_track", "team_frame"):
            d[col + "_name"] = d[col].map(name)
        for col in ("team_track", "vote"):
            t[col + "_name"] = t[col].map(name)
        t.insert(0, "seq", seq.info.name)
        dets.append(d)
        tracks.append(t)
    d, t = pd.concat(dets, ignore_index=True), pd.concat(tracks, ignore_index=True)
    rng = np.random.default_rng(0)
    clips = t["seq"].unique()
    rows = (("frame-level", "detection", d, "team_frame_name"), ("track-level", "detection", d, "team_track_name"),
            ("track-level", "track", t, "team_track_name"), ("majority of frame decisions", "track", t, "vote_name"))
    lines = ["| rule | unit | n | coverage | accuracy (covered) | macro-F1, unknown = error [95 % CI] |",
             "|---|---|---|---|---|---|"]
    for rule, unit, data, col in rows:
        sc = scores(data[col], data["gt_team"])
        boot = []
        for _ in range(args.boot):
            sub = data[data["seq"].isin(rng.choice(clips, len(clips)))]
            boot.append(scores(sub[col], sub["gt_team"])["f1"])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        lines.append(f"| {rule} | {unit} | {sc['n']} | {sc['coverage']:.1%} | {sc['accuracy']:.3f} | "
                     f"{sc['f1']:.3f} [{lo:.3f}, {hi:.3f}] |")
    lines += ["", "Track-level decisions per role (a referee should be unknown):", "",
              "| role | tracks | left | right | unknown |", "|---|---|---|---|---|"]
    for role, g in t.groupby(t["role"].fillna("no match")):
        c = g["team_track_name"].value_counts()
        lines.append(f"| {role} | {len(g)} | {c.get('left', 0)} | {c.get('right', 0)} | {c.get('unknown', 0)} |")
    conf = pd.crosstab(t["gt_team"].fillna("none"), t["team_track_name"])
    lines += ["", "Confusion per track (rows: ground truth, columns: track-level prediction):", "", conf.to_markdown()]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Teams report -- {path.name}\n\nSplit {run['split']}, {len(clips)} clips, oracle boxes: "
                   f"{run['oracle']}. Unit: a track. Intervals: clips resampled.\n\n" + "\n".join(lines) + "\n",
                   encoding="utf-8")
    print("\n".join(lines[:6]) + f"\n-> {out}")


if __name__ == "__main__":
    main()
