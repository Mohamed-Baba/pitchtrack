"""Dataset audit: what is really in the annotations, before any model is run."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .soccernet import ROLES, Sequence


# --------------------------------------------------------------------------- audit of one clip
def audit_sequence(seq: Sequence, small_px: float = 40.0, border_px: float = 1.0) -> dict:
    """Summary of one clip (one row of the audit table).

    Keys: name, split, game_id, n_frames, n_labeled, n_boxes, n_tracks, boxes_per_frame
    (mean over labeled frames), n_<role> for each role in ROLES, height_p05, height_p50,
    height_p95 (box heights in pixels), frac_small (height < small_px), frac_border
    (box within border_px of an image side: truncated players), frac_degenerate
    (width or height <= 0), frac_no_pitch (no pitch position), track_len_median
    (labeled frames per track).
    """
    gt, info = seq.gt, seq.info
    # Number of bounding boxes
    n_boxes = len(gt)

    # Width and height of every bounding box: shape (N,)
    w = (gt["x2"] - gt["x1"]).to_numpy()
    h = (gt["y2"] - gt["y1"]).to_numpy()

    # Detect boxes touching the image borders: shape (N,)
    border = (
        (gt["x1"] <= border_px)
        | (gt["y1"] <= border_px)
        | (gt["x2"] >= info.width - border_px)
        | (gt["y2"] >= info.height - border_px)
    )

    # Helper: fraction of boxes satisfying a boolean condition
    def frac(mask) -> float:
        return float(np.mean(mask)) if n_boxes else float("nan")

    # Helper: height quantile
    def quantile(q) -> float:
        return float(np.quantile(h, q)) if n_boxes else float("nan")

    # Basic sequence information
    row = {
        "name": info.name,
        "split": info.split,
        "game_id": info.game_id,
        "n_frames": info.n_frames,
        "n_labeled": int(len(seq.labeled_frames)),
        "n_boxes": n_boxes,
        "n_tracks": int(gt["track_id"].nunique()),
        "boxes_per_frame": n_boxes / max(len(seq.labeled_frames), 1),
    }

    # Count the number of boxes for each role
    for role in ROLES:
        row[f"n_{role}"] = int((gt["role"] == role).sum())

    # Statistical indicators
    row.update({
        "height_p05": quantile(0.05),
        "height_p50": quantile(0.50),
        "height_p95": quantile(0.95),

        "frac_small": frac(h < small_px),
        "frac_border": frac(border.to_numpy()),
        "frac_degenerate": frac((w <= 0) | (h <= 0)),
        "frac_no_pitch": frac(gt["pitch_x"].isna().to_numpy()),

        "track_len_median": (
            float(gt.groupby("track_id").size().median())
            if n_boxes else float("nan")
        ),
    })

    return row


def check_game_disjoint(audit: pd.DataFrame) -> pd.DataFrame:
    """Games that appear in more than one split (an empty table means a clean split).

    `audit` needs the columns game_id, split and name (one row per clip).
    """
    known = audit.dropna(subset=["game_id"])
    per_game = known.groupby("game_id")["split"].nunique()
    shared = per_game[per_game > 1].index
    return (known[known["game_id"].isin(shared)]
            .groupby(["game_id", "split"])["name"].count().rename("n_clips").reset_index())


def summarize_splits(audit: pd.DataFrame) -> pd.DataFrame:
    """Split-level totals of the audit table (clips, games, boxes, mean heights...)."""
    g = audit.groupby("split")
    return pd.DataFrame({
        "n_clips": g["name"].count(),
        "n_games": g["game_id"].nunique(),
        "n_boxes": g["n_boxes"].sum(),
        "boxes_per_frame": g["boxes_per_frame"].mean(),
        "height_p50": g["height_p50"].median(),
        "frac_small": (audit["frac_small"] * audit["n_boxes"]).groupby(audit["split"]).sum()
                      / g["n_boxes"].sum(),
    })
