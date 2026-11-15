"""Glue code: configuration, the detection stage over a dataset split, ground-truth frames.

The stages run on cached tables (detections, tracks, calibration...), so that each one can be
evaluated and tuned on its own; scripts/analyze_video.py chains them on a video, frame by frame.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .pitch_mask import foot_distance_to_pitch, pitch_mask
from .soccernet import iter_sequences

DET_COLUMNS = ["seq", "frame", "t", "x1", "y1", "x2", "y2", "score", "cls", "pitch_dist"]


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def frames_to_process(seq, stride: int) -> np.ndarray:
    """Labeled frames whose index is a multiple of `stride` (consecutive frames are correlated)."""
    return seq.labeled_frames[seq.labeled_frames % stride == 0]


def detect_split(cfg: dict, detector, split: str, stride: int, limit: int | None = None,
                 progress=None) -> tuple[pd.DataFrame, dict]:
    """Run `detector` on the selected frames of a split. Returns (table, run_info).

    The table has one row per detection (columns DET_COLUMNS) in original-frame pixels;
    pitch_dist is the distance of the foot point to the pitch mask (NaN if disabled).
    """
    use_mask = cfg.get("pitch_mask", {}).get("enabled", False)
    pm = cfg.get("pitch_mask", {})
    batch = int(cfg["detector"].get("batch", 8))
    rows, latencies, n_frames, clips = [], [], 0, []
    for k, seq in enumerate(iter_sequences(cfg["data"]["root"], split)):
        if limit is not None and k >= limit:
            break
        clips.append(seq.info.name)
        frames = frames_to_process(seq, stride)
        for start in range(0, len(frames), batch):
            idx = frames[start:start + batch]
            images = [seq.read_frame(int(f)) for f in idx]
            tic = time.perf_counter()
            dets = detector(images)
            latencies.append((time.perf_counter() - tic) / len(images))
            for f, img, d in zip(idx, images, dets):
                dist = np.full(len(d), np.nan)
                if use_mask and len(d):
                    mask = pitch_mask(img, pm.get("hsv_low", (35, 40, 40)), pm.get("hsv_high", (85, 255, 255)),
                                      pm.get("close_px", 25), pm.get("scale", 0.5))
                    dist = foot_distance_to_pitch(d.boxes, mask)
                for b, s, c, dd in zip(d.boxes, d.scores, d.classes, dist):
                    rows.append((seq.info.name, int(f), f / seq.info.fps, *b, s, int(c), dd))
            n_frames += len(idx)
        if progress is not None:
            progress(seq.info.name, n_frames)
    table = pd.DataFrame(rows, columns=DET_COLUMNS)
    lat = np.asarray(latencies[1:] if len(latencies) > 1 else latencies)  # drop warm-up batch
    info = {"split": split, "stride": stride, "clips": clips, "n_frames": n_frames, "n_detections": len(table),
            "latency_ms_per_frame": float(1000 * lat.mean()) if len(lat) else float("nan"),
            "config": cfg}
    return table, info


def save_detections(table: pd.DataFrame, info: dict, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)


def gt_det_frames(gt_by_seq: dict, dets: pd.DataFrame, min_score: float = 0.0,
                  pitch_radius: float | None = None):
    """Yield (clip_index, gt_boxes, det_boxes, det_scores) for every evaluated frame.

    gt_by_seq: {seq_name: (gt_table restricted to the evaluated frames, evaluated frames)}.
    Detections can be filtered by score and by distance to the pitch.
    """
    if pitch_radius is not None:
        dets = dets[dets["pitch_dist"].fillna(0.0) <= pitch_radius]
    dets = dets[dets["score"] >= min_score]
    det_groups = {key: g for key, g in dets.groupby(["seq", "frame"])}
    cols = ["x1", "y1", "x2", "y2"]
    for clip, (name, (gt, frames)) in enumerate(sorted(gt_by_seq.items())):
        gt_groups = {f: g for f, g in gt.groupby("frame")}
        for f in frames:
            g = gt_groups.get(int(f))
            d = det_groups.get((name, int(f)))
            yield (clip,
                   g[cols].to_numpy() if g is not None else np.zeros((0, 4)),
                   d[cols].to_numpy() if d is not None else np.zeros((0, 4)),
                   d["score"].to_numpy() if d is not None else np.zeros(0))


# --------------------------------------------------------------------------- tracking helpers
def detections_per_frame(table: pd.DataFrame, seq_name: str, frames, fps: float,
                         min_score: float = 0.0, pitch_radius: float | None = None):
    """[(frame, t, Detections)] of one clip from a detection table, every frame present."""
    from .detector import Detections

    d = table[(table["seq"] == seq_name) & (table["score"] >= min_score)]
    if pitch_radius is not None and "pitch_dist" in d:
        d = d[d["pitch_dist"].fillna(0.0) <= pitch_radius]
    groups = dict(tuple(d.groupby("frame")))
    cols = ["x1", "y1", "x2", "y2"]
    out = []
    for f in frames:
        g = groups.get(int(f))
        dets = Detections.empty() if g is None else Detections(g[cols].to_numpy(), g["score"].to_numpy(),
                                                                g["cls"].to_numpy())
        out.append((int(f), int(f) / fps, dets))
    return out


def gt_per_frame(seq, frames):
    """Oracle detections: the ground-truth boxes with score 1 (association-only setting)."""
    from .detector import Detections

    groups = dict(tuple(seq.gt.groupby("frame")))
    cols = ["x1", "y1", "x2", "y2"]
    out = []
    for f in frames:
        g = groups.get(int(f))
        dets = Detections.empty() if g is None else Detections(g[cols].to_numpy(), np.ones(len(g)),
                                                                np.zeros(len(g), dtype=np.int64))
        out.append((int(f), int(f) / seq.info.fps, dets))
    return out
