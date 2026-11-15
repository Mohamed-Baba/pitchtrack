"""Positions on the pitch and statistics per track -> parquet, CSV, heatmaps.

    python scripts/run_statistics.py --tracks outputs/trk/v_byte_cmc.parquet --calib outputs/calib/valid.parquet \
        --teams outputs/teams/valid.parquet --out outputs/stats/valid
    python scripts/run_statistics.py --oracle --calib outputs/calib/valid.parquet --out outputs/stats/valid_oracle

Positions: bottom-middle of each box projected with the homography of its frame; NaN where
the frame is not calibrated or the point is not valid. Statistics: statistics.track_table with the settings of
configs/analysis.yaml (statistics). A track is "reliable" if at least one segment is reliable, no step
was implausible, and at least 80 % of its frames have a position; distances are given for
reliable segments only (gaps are marked, never interpolated). Outputs in --out:
positions.parquet, player_stats.csv, heatmaps.png (per team, all clips), summary.md.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pitchtrack.boxes import foot_points  # noqa: E402
from pitchtrack.pipeline import load_config  # noqa: E402
from pitchtrack.pitch_calibration import image_to_pitch, pitch_covariance  # noqa: E402
from pitchtrack.soccernet import list_sequences, load_sequence  # noqa: E402
from pitchtrack.statistics import heatmap, track_table  # noqa: E402
from pitchtrack.visualization import plot_pitch  # noqa: E402

G_COLS = [f"g{i}" for i in range(9)]


def positions(boxes: pd.DataFrame, cal: pd.DataFrame, fps: float, cfg) -> pd.DataFrame:
    """Rows of boxes (frame, track_id, x1..y2) -> + t, X, Y, pos_std_m (NaN where unavailable)."""
    out = boxes.copy()
    out["t"] = out["frame"] / fps
    out[["X", "Y"]] = np.nan
    out["pos_std_m"] = np.nan
    cal = cal.set_index("frame")
    for f, idx in out.groupby("frame").indices.items():
        if f not in cal.index or not bool(cal.at[f, "ok"]):
            continue
        G = cal.loc[f, G_COLS].to_numpy(dtype=np.float64).reshape(3, 3)
        uv = foot_points(out.iloc[idx][["x1", "y1", "x2", "y2"]].to_numpy())
        xy, ok = image_to_pitch(G, uv, cfg["margin_m"])
        out.iloc[idx, out.columns.get_indexer(["X", "Y"])] = xy
        cov = pitch_covariance(G, uv, np.eye(2) * cfg["pixel_sigma"] ** 2)
        std = np.sqrt(np.trace(cov, axis1=1, axis2=2) / 2.0)
        out.iloc[idx, out.columns.get_loc("pos_std_m")] = np.where(ok, std, np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/analysis.yaml")
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--oracle", action="store_true", help="ground-truth boxes and ids")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--teams", default=None, help="run_teams.py table (same tracks)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.oracle and not args.tracks:
        ap.error("give --tracks or --oracle")
    cfg = load_config(args.config)["statistics"]
    calib = pd.read_parquet(args.calib)
    run = json.loads(Path(args.calib).with_suffix(".json").read_text(encoding="utf-8"))
    tracks = pd.read_parquet(args.tracks) if args.tracks else None
    teams = pd.read_parquet(args.teams) if args.teams else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pos_all = []
    for path in list_sequences(run["data_root"], run["split"]):
        if path.name not in run["clips"]:
            continue
        seq = load_sequence(path, run["split"])
        name = seq.info.name
        boxes = (seq.gt[["frame", "track_id", "x1", "y1", "x2", "y2"]] if args.oracle
                 else tracks[tracks["seq"] == name][["frame", "track_id", "x1", "y1", "x2", "y2"]])
        p = positions(boxes.reset_index(drop=True), calib[calib["seq"] == name], seq.info.fps, cfg)
        team_of = {}
        if teams is not None:
            team_of = teams[teams["seq"] == name].groupby("track_id")["team_track"].first().to_dict()
        p["team"] = p["track_id"].map(team_of).fillna(-1).astype(int)
        p.insert(0, "seq", name)
        pos_all.append(p)
    pos = pd.concat(pos_all, ignore_index=True)
    pos.to_parquet(out / "positions.parquet", index=False)
    st = track_table(pos, cfg)
    st.to_csv(out / "player_stats.csv", index=False, float_format="%.3f")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, (team, label) in zip(axes, ((0, "team 0"), (1, "team 1"), (-1, "unknown"))):
        h = heatmap(pos.loc[pos["team"] == team, ["X", "Y"]].to_numpy())
        ax.imshow(h.T, extent=(-52.5, 52.5, 34, -34), cmap="magma", interpolation="nearest")
        plot_pitch(ax)
        ax.set_title(f"{label}: {int(np.isfinite(pos.loc[pos['team'] == team, 'X']).sum())} positions")
    fig.tight_layout()
    fig.savefig(out / "heatmaps.png", dpi=120)
    rel = st[st["reliable"]]
    lines = [f"# Statistics -- {out.name}", "",
             f"{len(st)} tracks, {len(rel)} reliable ({len(rel) / max(1, len(st)):.0%}); positions available for "
             f"{np.isfinite(pos['X']).mean():.1%} of the boxes. Settings: {cfg}.", "",
             "| | reliable tracks | median distance per minute (m) | median max speed (m/s) | median position std (m) |",
             "|---|---|---|---|---|"]
    for team, g in rel.groupby("team"):
        per_min = 60 * g["distance_m"] / g["reliable_s"].clip(lower=1e-9)
        lines.append(f"| team {team} | {len(g)} | {per_min.median():.1f} | {g['max_speed'].median():.2f} | "
                     f"{g['median_pos_std_m'].median():.2f} |")
    lines += ["", "![heatmaps](heatmaps.png)", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[2:-3]) + f"\n-> {out}")


if __name__ == "__main__":
    main()
