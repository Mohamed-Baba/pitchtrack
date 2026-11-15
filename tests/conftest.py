"""Synthetic data for the tests: no download, CPU only."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

W, H = 320, 180

def _bbox_image(x1, y1, w, h):
    return {"x": x1, "y": y1, "x_center": x1 + w / 2, "y_center": y1 + h / 2, "w": w, "h": h}


def make_gsr_clip(root: Path, split: str, name: str, game_id: int, n_frames: int = 4) -> Path:
    """A tiny SoccerNet-GSR clip with known content (see EXPECTED below)."""
    clip = root / split / name
    (clip / "img1").mkdir(parents=True)
    images = []
    for k in range(1, n_frames + 1):
        img = np.full((H, W, 3), (40, 140, 40), np.uint8)        # green-ish BGR
        cv2.imwrite(str(clip / "img1" / f"{k:06d}.jpg"), img)
        images.append({"image_id": f"9{game_id:02d}{k:06d}", "file_name": f"{k:06d}.jpg",
                       "height": H, "width": W, "is_labeled": k <= 3})
    iid = {k: images[k - 1]["image_id"] for k in range(1, n_frames + 1)}
    pitch = {"x_bottom_middle": 1.5, "y_bottom_middle": -2.0, "x_bottom_left": 1.0,
             "y_bottom_left": -2.0, "x_bottom_right": 2.0, "y_bottom_right": -2.0}
    ann = []
    for k in (1, 2, 3):   # track 1: player of the left team, moves 10 px per frame
        ann.append({"id": f"a1{k}", "image_id": iid[k], "track_id": 1, "supercategory": "object",
                    "category_id": 1, "attributes": {"role": "player", "jersey": "9", "team": "left"},
                    "bbox_image": _bbox_image(100 + 10 * (k - 1), 50, 20, 40), "bbox_pitch": pitch})
    for k in (1, 2):      # track 2: referee, no team, no jersey
        ann.append({"id": f"a2{k}", "image_id": iid[k], "track_id": 2, "supercategory": "object",
                    "category_id": 3, "attributes": {"role": "referee", "jersey": None, "team": None},
                    "bbox_image": _bbox_image(200, 60, 16, 32), "bbox_pitch": pitch})
    ann.append({"id": "a31", "image_id": iid[1], "track_id": 3, "supercategory": "object",   # goalkeeper on the border
                "category_id": 2, "attributes": {"role": "goalkeeper", "jersey": "1", "team": "right"},
                "bbox_image": _bbox_image(0, 100, 12, 30), "bbox_pitch": pitch})
    ann.append({"id": "a43", "image_id": iid[3], "track_id": 4, "supercategory": "object",   # other, no pitch position
                "category_id": 4, "attributes": {"role": "other", "jersey": None, "team": None},
                "bbox_image": _bbox_image(250, 20, 10, 60)})
    ann.append({"id": "p1", "image_id": iid[1], "supercategory": "pitch", "category_id": 5,
                "lines": {"Middle line": [{"x": 0.5, "y": 0.1}, {"x": 0.5, "y": 0.9}]}})
    ann.append({"id": "c1", "image_id": iid[1], "supercategory": "camera", "category_id": 6,
                "parameters": {}})
    info = {"version": "1.3", "name": name, "id": name[-3:], "game_id": str(game_id),
            "frame_rate": 25, "seq_length": n_frames, "im_dir": "img1", "im_ext": ".jpg"}
    with open(clip / "Labels-GameState.json", "w", encoding="utf-8") as f:
        json.dump({"info": info, "images": images, "annotations": ann, "categories": []}, f)
    return clip


@pytest.fixture
def gsr_root(tmp_path) -> Path:
    make_gsr_clip(tmp_path, "valid", "SNGS-901", game_id=7)
    make_gsr_clip(tmp_path, "valid", "SNGS-902", game_id=7)
    make_gsr_clip(tmp_path, "train", "SNGS-801", game_id=3)
    return tmp_path


@pytest.fixture
def pitch_image():
    """A 'broadcast' frame: grey stands on top, a green trapezoid pitch, a red player and a
    white line on the pitch, a person-sized grey blob off the pitch."""
    img = np.full((360, 640, 3), (90, 90, 90), np.uint8)
    poly = np.array([[120, 120], [520, 120], [640, 330], [0, 330]], np.int32)
    cv2.fillPoly(img, [poly], (50, 150, 60))
    cv2.line(img, (320, 120), (320, 330), (255, 255, 255), 3)
    cv2.rectangle(img, (300, 200), (315, 240), (30, 30, 200), -1)   # player on the pitch
    cv2.rectangle(img, (40, 40), (55, 80), (120, 120, 120), -1)      # person in the stands
    return img, poly


# --------------------------------------------------------------------------- a broadcast camera
def broadcast_camera(target=(-30.0, 0.0), cam=(-20.0, 74.0, -18.0), width=1920, height=1080,
                     hfov_deg=40.0):
    """Pinhole camera above the near touch line looking at `target` on the ground (SoccerNet
    frame: z < 0 above the ground). Returns (H pitch -> image 3x3, P 3x4, (width, height))."""
    f = (width / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
    K = np.array([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])
    C = np.asarray(cam, dtype=np.float64)
    z = np.array([target[0], target[1], 0.0]) - C
    z /= np.linalg.norm(z)
    down = np.array([0.0, 0.0, 1.0])
    y = down - (down @ z) * z
    y /= np.linalg.norm(y)
    R = np.stack([np.cross(y, z), y, z])
    P = K @ np.c_[R, -R @ C]
    return P[:, [0, 1, 3]], P, (width, height)


def pitch_line_annotation(Hm, size, step=0.25) -> dict:
    """The "lines" annotation of a view, in the GSR format: for every class, the longest visible run
    of its points, normalised with the (W - 1, H - 1) convention."""
    from pitchtrack import pitch_model as pm
    from pitchtrack.pitch_calibration import pitch_to_image
    W, H = size
    out = {}
    for name, xy in pm.sample_segments(step).items():
        uv = pitch_to_image(Hm, xy)
        ok = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
        runs = [r for r in np.split(np.arange(len(uv)), np.flatnonzero(np.diff(ok.astype(int))) + 1) if ok[r[0]]]
        if runs and len(max(runs, key=len)) >= 2:
            run = max(runs, key=len)
            out[name] = [{"x": float(u / (W - 1)), "y": float(v / (H - 1))} for u, v in uv[run]]
    return out


def render_broadcast_frame(Hm, size, P=None, players=(), seed=0) -> np.ndarray:
    """A rendered broadcast view: noisy stands, striped grass, white lines, and optional players
    [(X, Y, bgr)] drawn as 1.8 m tall boxes (P is the 3 x 4 camera, needed for their height)."""
    from pitchtrack import pitch_model as pm
    from pitchtrack.pitch_calibration import pitch_to_image
    W, H = size
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(90, 25, (H, W, 3)), 0, 255).astype(np.uint8)
    edges = np.linspace(-pm.HALF_L - 6, pm.HALF_L + 6, 21)
    for k in range(20):
        band = [(edges[k], -pm.HALF_W - 5), (edges[k + 1], -pm.HALF_W - 5), (edges[k + 1], pm.HALF_W + 5),
                (edges[k], pm.HALF_W + 5)]
        uv = pitch_to_image(Hm, band)
        if np.isfinite(uv).all():
            cv2.fillPoly(img, [np.round(uv).astype(np.int32)], (45, 135 + 18 * (k % 2), 50), cv2.LINE_AA)
    thick = max(1, int(round(3 * W / 1920)))
    for name, xy in pm.sample_segments(0.25).items():
        uv = pitch_to_image(Hm, xy)
        ok = np.isfinite(uv).all(axis=1)
        cv2.polylines(img, [np.round(uv[ok]).astype(np.int32)], name == "Circle central", (235, 235, 235), thick,
                      cv2.LINE_AA)
    for X, Y, colour in players:
        foot, head = P @ np.array([X, Y, 0.0, 1.0]), P @ np.array([X, Y, -1.8, 1.0])
        (u, v), (_, v_top) = foot[:2] / foot[2], head[:2] / head[2]
        h = v - v_top
        cv2.rectangle(img, (int(round(u - 0.2 * h)), int(round(v_top))), (int(round(u + 0.2 * h)),
                      int(round(v_top + 0.6 * h))), colour, -1)
        cv2.rectangle(img, (int(round(u - 0.2 * h)), int(round(v_top + 0.6 * h))), (int(round(u + 0.2 * h)),
                      int(round(v))), (25, 25, 25), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def make_line_clip(root: Path, split: str, name: str, views, size=(480, 272)) -> Path:
    """A GSR clip of rendered broadcast views [(target, cam)] with their pitch-line annotations."""
    clip = root / split / name
    (clip / "img1").mkdir(parents=True)
    images, ann = [], []
    for k, (target, cam) in enumerate(views, start=1):
        Hm, _, _ = broadcast_camera(target, cam, *size)
        cv2.imwrite(str(clip / "img1" / f"{k:06d}.jpg"), render_broadcast_frame(Hm, size, seed=k))
        images.append({"image_id": f"{k:06d}", "file_name": f"{k:06d}.jpg", "height": size[1], "width": size[0],
                       "is_labeled": True, "has_labeled_pitch": True})
        ann.append({"id": f"p{k}", "image_id": f"{k:06d}", "supercategory": "pitch", "category_id": 5,
                    "lines": pitch_line_annotation(Hm, size)})
    info = {"version": "1.3", "name": name, "frame_rate": 25, "seq_length": len(views), "im_dir": "img1"}
    with open(clip / "Labels-GameState.json", "w", encoding="utf-8") as f:
        json.dump({"info": info, "images": images, "annotations": ann, "categories": []}, f)
    return clip


def visible_line_points(Hm, size, names=None, step=1.0, noise_px=0.0, seed=0):
    """Points of the pitch segments seen by the camera, in pixels, with their classes."""
    from pitchtrack import pitch_model as pm
    from pitchtrack.pitch_calibration import pitch_to_image
    rng = np.random.default_rng(seed)
    W, H = size
    pts, cls = [], []
    for name, xy in pm.sample_segments(step).items():
        if (names is not None and name not in names) or not pm.is_ground_line(name):
            continue
        uv = pitch_to_image(Hm, xy)
        ok = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
        pts.append(uv[ok])
        cls += [name] * int(ok.sum())
    pts = np.concatenate(pts) if pts else np.zeros((0, 2))
    return pts + rng.normal(0.0, noise_px, pts.shape), cls
