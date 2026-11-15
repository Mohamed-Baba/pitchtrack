"""Pitch-line annotations as label maps, and back.

SoccerNet-GSR annotates, for most frames, points on the visible pitch lines: a dictionary
{class name: [{"x": .., "y": ..}, ...]} in normalised image coordinates (pixel u = x (W - 1),
v = y (H - 1)). `pitch_calibration.calibrate_frame` turns such a dictionary into a homography.

This module converts the dictionary into a per-pixel label map (the training target of the line
network of `line_model.py`) and converts a predicted label map back into the same dictionary,
so that the calibration code runs unchanged on the output of the network:

    frame -> network -> label map -> label_map_to_lines -> {"Middle line": [...], ...}
          -> calibrate_frame

Label 0 is the background; label k >= 1 is CLASSES[k - 1]. Only the classes of the pitch model
are learned (17 straight segments and 3 arcs); goal posts and unidentified lines are ignored, as
they are by the calibration.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import pitch_model as pm

CLASSES = tuple(pm.SEGMENTS) + tuple(pm.CIRCLES)
NUM_LABELS = len(CLASSES) + 1                      # + background
LABEL_OF = {name: k + 1 for k, name in enumerate(CLASSES)}


def mirror_name(name: str) -> str:
    """Class seen in the horizontally flipped image. Flipping the image maps the pitch point
    (x, y) to (-x, y): left and right swap, top and bottom do not."""
    return name.replace("left", "\0").replace("right", "left").replace("\0", "right")


FLIP_LUT = np.array([0] + [LABEL_OF[mirror_name(n)] for n in CLASSES], dtype=np.uint8)


# --------------------------------------------------------------------------- annotations
def normalised_points(lines: dict | None) -> dict:
    """{class: (n, 2) normalised points} for the learned classes; other classes are dropped."""
    out = {}
    for name, polyline in (lines or {}).items():
        name = name.strip()
        if name in LABEL_OF and polyline:
            out[name] = np.array([[float(p["x"]), float(p["y"])] for p in polyline], dtype=np.float64)
    return out


def to_annotation(points: dict) -> dict:
    """{class: (n, 2) normalised points} -> the dictionary format of GSR."""
    return {name: [{"x": float(x), "y": float(y)} for x, y in np.asarray(p).reshape(-1, 2)]
            for name, p in points.items()}


def flip_points(points: dict) -> dict:
    """Annotation of the horizontally flipped frame: x -> 1 - x, left <-> right."""
    return {mirror_name(n): np.c_[1.0 - p[:, 0], p[:, 1]] for n, p in points.items()}


def crop_points(points: dict, box) -> dict:
    """Annotation of the crop (x0, y0, x1, y1) (normalised) once resized to the full frame."""
    x0, y0, x1, y1 = (float(v) for v in box)
    return {n: np.c_[(p[:, 0] - x0) / (x1 - x0), (p[:, 1] - y0) / (y1 - y0)] for n, p in points.items()}


def render_label_map(points: dict, size_wh, thickness: int = 3) -> np.ndarray:
    """(h, w) uint8 label map with every polyline drawn with its label.

    Consecutive annotated points are joined, as in the SoccerNet calibration code. Arcs are drawn
    first, so that the straight lines, the ones the homography uses, win at the crossings. No
    anti-aliasing: a blended pixel would get a meaningless label.
    """
    w, h = (int(v) for v in size_wh)
    out = np.zeros((h, w), np.uint8)
    scale = np.array([w - 1, h - 1], dtype=np.float64) * 4.0          # shift=2: quarter pixels
    for name in sorted(points, key=lambda n: not pm.is_circle(n)):
        uv = np.round(np.asarray(points[name]) * scale)
        uv = np.clip(uv, -2**20, 2**20).astype(np.int32).reshape(-1, 1, 2)
        if len(uv) == 1:
            cv2.circle(out, tuple(int(c) for c in uv[0, 0]), max(1, thickness // 2) * 4, int(LABEL_OF[name]),
                       -1, cv2.LINE_8, shift=2)
        else:
            cv2.polylines(out, [uv], False, int(LABEL_OF[name]), thickness, cv2.LINE_8, shift=2)
    return out


# --------------------------------------------------------------------------- label map -> annotation
def _components(mask: np.ndarray, straight: bool, tol_px: float, min_area: int) -> np.ndarray:
    """The main connected component of a class, plus the components that continue it: collinear
    ones for a straight line (a line cut by a player), large enough ones for an arc. Small
    isolated blobs are dropped. Returns the kept pixels as a boolean mask of the same shape."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = 1 + int(np.argmax(areas))
    keep = [main]
    if straight:
        yy, xx = np.nonzero(lab == main)
        P = np.c_[xx, yy].astype(np.float64)
        c = P.mean(axis=0)
        normal = np.linalg.svd(P - c, full_matrices=False)[2][-1] if len(P) > 1 else np.array([0.0, 1.0])
        for j in range(1, n):
            if j != main and areas[j - 1] >= min_area:
                yy, xx = np.nonzero(lab == j)
                if np.median(np.abs((np.c_[xx, yy] - c) @ normal)) <= tol_px:
                    keep.append(j)
    else:
        keep += [j for j in range(1, n) if j != main and areas[j - 1] >= max(min_area, 0.2 * areas.max())]
    return np.isin(lab, keep)


def _run_centres(mask: np.ndarray, weight: np.ndarray, max_run: int) -> np.ndarray:
    """Centre line of a thick curve: the weighted centre of every vertical run of pixels
    (column scan) and of every horizontal run (row scan) that crosses the band, i.e. is at most
    max_run long. A run longer than that follows the band instead of crossing it: the other scan
    handles that part. Returns (n, 2) points (u, v)."""
    out = []
    for transpose in (True, False):                    # column scan, then row scan
        m = mask.T if transpose else mask
        wt = (weight.T if transpose else weight) * m
        pad = np.pad(m.astype(np.int8), ((0, 0), (1, 1)))
        d = np.diff(pad, axis=1)
        line_s, start = np.nonzero(d == 1)
        _, end = np.nonzero(d == -1)
        ok = (end - start) <= max_run
        if not ok.any():
            continue
        line_s, start, end = line_s[ok], start[ok], end[ok]
        idx = np.arange(m.shape[1], dtype=np.float64)
        cw = np.pad(np.cumsum(wt, axis=1), ((0, 0), (1, 0)))
        cwi = np.pad(np.cumsum(wt * idx, axis=1), ((0, 0), (1, 0)))
        sw = cw[line_s, end] - cw[line_s, start]
        centre = (cwi[line_s, end] - cwi[line_s, start]) / np.maximum(sw, 1e-12)
        pts = np.c_[line_s, centre] if transpose else np.c_[centre, line_s]
        out.append(pts[sw > 0])
    return np.concatenate(out) if out else np.zeros((0, 2))


def band_width(mask: np.ndarray) -> float:
    """Mean width of a thin band of pixels: 2 x area / perimeter (a band of length L and width t
    has an area L t and a perimeter close to 2 L)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    return 2.0 * float(mask.sum()) / max(perimeter, 1.0)


def _spread(pts: np.ndarray, straight: bool, max_points: int) -> np.ndarray:
    """At most max_points points spread evenly along the curve (along its principal direction
    for a line, by angle around the centroid for an arc)."""
    c = pts.mean(axis=0)
    if straight:
        d = np.linalg.svd(pts - c, full_matrices=False)[2][0] if len(pts) > 1 else np.array([1.0, 0.0])
        order = np.argsort((pts - c) @ d, kind="mergesort")
    else:
        order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]), kind="mergesort")
    pts = pts[order]
    if len(pts) > max_points:
        pts = pts[np.round(np.linspace(0, len(pts) - 1, max_points)).astype(int)]
    return pts


def label_map_to_lines(labels: np.ndarray, conf: np.ndarray | None = None, min_conf: float = 0.5,
                       min_pixels: int = 30, max_points: int = 40, max_run: int = 8,
                       line_tol_px: float = 3.0) -> dict:
    """Predicted label map (h, w) and confidence (h, w) -> annotation dictionary (GSR format).

    For each class: its pixels with confidence >= min_conf; nothing if fewer than min_pixels; the
    main component and its continuations (_components); the centre line of the band
    (_run_centres, weighted by the confidence); at most max_points points (_spread); coordinates
    normalised with the (W - 1, H - 1) convention of GSR. A band of width t is crossed by runs of
    length t / cos(angle): the two scans cover every angle only if the longest run kept is at
    least t sqrt(2), so max_run grows to 2 t for bands wider than max_run / 2.
    """
    labels = np.asarray(labels)
    h, w = labels.shape
    conf = np.ones((h, w), np.float32) if conf is None else np.asarray(conf, dtype=np.float32)
    lab = np.where(conf >= min_conf, labels, 0)
    counts = np.bincount(lab.ravel(), minlength=NUM_LABELS)
    out = {}
    for k, name in enumerate(CLASSES, start=1):
        if counts[k] < min_pixels:
            continue
        vs, us = np.nonzero(lab == k)                      # work in the bounding box of the class
        v0, v1, u0, u1 = vs.min(), vs.max() + 1, us.min(), us.max() + 1
        straight = pm.is_ground_line(name)
        mask = _components(lab[v0:v1, u0:u1] == k, straight, line_tol_px, min_area=max(5, min_pixels // 4))
        if int(mask.sum()) < min_pixels:
            continue
        run = max(max_run, int(np.ceil(2.0 * band_width(mask))))
        pts = _run_centres(mask, conf[v0:v1, u0:u1], run)
        if len(pts) < 2:
            continue
        pts = _spread(pts + [u0, v0], straight, max_points)
        out[name] = [{"x": float(u / (w - 1)), "y": float(v / (h - 1))} for u, v in pts]
    return out


def colour_label_map(labels: np.ndarray) -> np.ndarray:
    """BGR image of a label map (one colour per class, black background), for inspection."""
    rng = np.random.default_rng(7)
    lut = rng.integers(60, 256, (NUM_LABELS, 3)).astype(np.uint8)
    lut[0] = 0
    return lut[np.asarray(labels)]
