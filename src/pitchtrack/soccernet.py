"""SoccerNet Game State Reconstruction (SoccerNet-GSR) clips.

Layout after unzipping a split (dataset version >= 1.3):

    <root>/<split>/SNGS-XXX/img1/000001.jpg ... 000750.jpg
    <root>/<split>/SNGS-XXX/Labels-GameState.json

Labels-GameState.json holds "info" (name, game id, frame rate, length...), "images"
(one entry per frame: image_id, file_name, is_labeled...) and "annotations". An
annotation with supercategory "object" is one person in one frame: track_id,
attributes {role, team, jersey}, bbox_image (pixels) and bbox_pitch (metres, origin at
the centre spot, x towards the right goal). Annotations with supercategory "pitch"
hold the pitch lines (normalised image coordinates), used for the calibration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

GT_COLUMNS = ["frame", "track_id", "x1", "y1", "x2", "y2", "role", "team", "jersey",
              "pitch_x", "pitch_y"]
ROLES = ("player", "goalkeeper", "referee", "other")


@dataclass
class SequenceInfo:
    name: str
    split: str
    game_id: int | None
    fps: float
    n_frames: int
    width: int
    height: int
    img_dir: Path
    version: str


@dataclass
class Sequence:
    """One 30-second clip: metadata, ground truth table and pitch-line annotations."""

    info: SequenceInfo
    gt: pd.DataFrame                    # columns GT_COLUMNS, sorted by (frame, track_id)
    labeled_frames: np.ndarray          # sorted 0-based indices of annotated frames
    file_names: dict = field(default_factory=dict)   # 0-based frame -> file name
    lines: dict = field(default_factory=dict)        # 0-based frame -> raw "lines" dict

    def frame_path(self, frame: int) -> Path:
        name = self.file_names.get(frame, f"{frame + 1:06d}.jpg")
        return self.info.img_dir / name

    def read_frame(self, frame: int) -> np.ndarray:
        img = cv2.imread(str(self.frame_path(frame)), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.frame_path(frame))
        return img

    def gt_frame(self, frame: int) -> pd.DataFrame:
        return self.gt[self.gt["frame"] == frame]


def frame_index(file_name: str) -> int:
    """'000001.jpg' -> 0: files are numbered from 1, frames are indexed from 0."""
    return int(Path(file_name).stem) - 1


# --------------------------------------------------------------------------- ground truth
def annotation_to_row(ann: dict, frame_of_image: dict) -> dict | None:
    """One GSR annotation -> one ground-truth row (a dict with the keys GT_COLUMNS).

    Returns None for annotations that are not persons (supercategory != "object").
    frame_of_image maps an image_id to its 0-based frame index (built from "images").
    bbox_image gives the CENTRE (x_center, y_center) and the size (w, h) in pixels:
    this is what the reference loader (TrackLab) reads. Missing team, jersey or pitch
    position become None / NaN.
    """
    # 1. Ignore annotations that do not describe a person
    if ann.get("supercategory") != "object":
        return None

    # 2. Recover the 0-based frame index from image_id
    frame = frame_of_image[ann["image_id"]]

    # 3. Convert bbox_image from (xc, yc, w, h) to (x1, y1, x2, y2)
    bbox = ann["bbox_image"]
    xc = bbox["x_center"]
    yc = bbox["y_center"]
    w = bbox["w"]
    h = bbox["h"]

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    # 4. Recover optional person attributes
    attributes = ann.get("attributes") or {}
    role = attributes.get("role")
    team = attributes.get("team")
    jersey = attributes.get("jersey")

    # 5. Recover pitch position (NaN if missing)
    bbox_pitch = ann.get("bbox_pitch") or {}

    pitch_x = bbox_pitch.get("x_bottom_middle")
    pitch_y = bbox_pitch.get("y_bottom_middle")

    if pitch_x is None:
        pitch_x = np.nan
    if pitch_y is None:
        pitch_y = np.nan

    # 6. Return one ground-truth row
    return {
        "frame": frame,
        "track_id": int(ann["track_id"]),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "role": role,
        "team": team,
        "jersey": jersey,
        "pitch_x": pitch_x,
        "pitch_y": pitch_y,
    }


def load_sequence(seq_dir, split: str | None = None) -> Sequence:
    """Read one clip folder (img1/ + Labels-GameState.json)."""
    seq_dir = Path(seq_dir)
    with open(seq_dir / "Labels-GameState.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    info, images = data.get("info", {}), data.get("images", [])
    frame_of_image = {img["image_id"]: frame_index(img["file_name"]) for img in images}
    file_names = {frame_index(img["file_name"]): img["file_name"] for img in images}
    labeled = sorted(frame_index(img["file_name"]) for img in images if img.get("is_labeled", True))

    rows, lines = [], {}
    for ann in data.get("annotations", []):
        if ann.get("supercategory") == "pitch" and "lines" in ann:
            lines[frame_of_image[ann["image_id"]]] = ann["lines"]
            continue
        row = annotation_to_row(ann, frame_of_image)
        if row is not None:
            rows.append(row)
    gt = pd.DataFrame(rows, columns=GT_COLUMNS)
    gt = gt.sort_values(["frame", "track_id"], kind="mergesort").reset_index(drop=True)

    game_id = info.get("game_id", info.get("gameID"))
    first = images[0] if images else {}
    meta = SequenceInfo(
        name=info.get("name", seq_dir.name),
        split=split or seq_dir.parent.name,
        game_id=None if game_id in (None, "") else int(game_id),
        fps=float(info.get("frame_rate", 25)),
        n_frames=int(info.get("seq_length", len(images))),
        width=int(first.get("width", 1920)),
        height=int(first.get("height", 1080)),
        img_dir=seq_dir / info.get("im_dir", "img1"),
        version=str(info.get("version", "unknown")),
    )
    return Sequence(meta, gt, np.asarray(labeled, dtype=np.int64), file_names, lines)


def list_sequences(root, split: str) -> list[Path]:
    """Clip folders of a split, sorted by name."""
    base = Path(root) / split
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found: download and unzip the split first (data/README.md)")
    return sorted(p for p in base.iterdir() if (p / "Labels-GameState.json").is_file())


def iter_sequences(root, split: str):
    for path in list_sequences(root, split):
        yield load_sequence(path, split)
