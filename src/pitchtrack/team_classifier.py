"""Team assignment from shirt colours.

Baseline: a torso region of each box, a robust colour statistic in Lab with the
grass pixels removed, k-means on the colours of one clip, then an assignment that aggregates the
frames of each track and is allowed to answer "unknown" (referees, goalkeepers, shadows, too few
pixels). Cluster numbers are arbitrary: they are mapped to the "left" / "right" labels of GSR by
the best permutation, per clip, only for evaluation.
"""
from __future__ import annotations

import cv2
import numpy as np

UNKNOWN = -1
TORSO = (0.15, 0.50, 0.25, 0.75)     # rows (top, bottom) and columns (left, right), fractions of the box


def torso_box(box, image_shape, region=TORSO) -> tuple[int, int, int, int]:
    """Integer torso rectangle (x1, y1, x2, y2) of a box, clipped to the image (may be empty)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    top, bottom, left, right = region
    H, W = image_shape[:2]
    c1, c2 = int(round(x1 + left * w)), int(round(x1 + right * w))
    r1, r2 = int(round(y1 + top * h)), int(round(y1 + bottom * h))
    return max(0, c1), max(0, r1), min(W, c2), min(H, r2)


# --------------------------------------------------------------------------- shirt colour
def torso_colour(image_bgr: np.ndarray, boxes, min_pixels: int = 20, hsv_low=(35, 40, 40),
                 hsv_high=(85, 255, 255), region=TORSO):
    """Median Lab colour of the non-grass pixels of the torso of each box.

    Returns (features (n, 3) float64: L in [0, 100], a and b in about [-127, 127], NaN rows for
    invalid boxes; valid (n,) bool). A box is invalid if fewer than min_pixels torso pixels are
    left once the grass (HSV inside [hsv_low, hsv_high], as in pitch_mask) is removed.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    feats = np.full((len(boxes), 3), np.nan)
    valid = np.zeros(len(boxes), dtype=bool)

    for i, box in enumerate(boxes):
        # Torso rectangle of the box, clipped to the image, then the crop
        c1, r1, c2, r2 = torso_box(box, image_bgr.shape, region=region)
        crop = image_bgr[r1:r2, c1:c2]

        if crop.size == 0:
            continue

        # Convert the crop to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Detect grass pixels
        grass_mask = cv2.inRange(
            hsv,
            np.array(hsv_low, dtype=np.uint8),
            np.array(hsv_high, dtype=np.uint8)
        )

        # Keep non-grass pixels
        non_grass_mask = cv2.bitwise_not(grass_mask)
        pixels = crop[non_grass_mask > 0]

        if len(pixels) < min_pixels:
            continue

        # Convert BGR pixels to Lab
        pixels_float = pixels.astype(np.float32) / 255.0
        pixels_lab = cv2.cvtColor(
            pixels_float.reshape(-1, 1, 3),
            cv2.COLOR_BGR2Lab
        ).reshape(-1, 3)

        # Median Lab feature
        feats[i] = np.median(pixels_lab, axis=0)
        valid[i] = True

    return feats, valid



def colour_space(feats, l_weight: float = 0.5) -> np.ndarray:
    """Features used for clustering: (l_weight L, a, b). A small weight on L limits the effect of
    shadows and sun, which change the lightness much more than the chroma."""
    f = np.asarray(feats, dtype=np.float64).reshape(-1, 3).copy()
    f[:, 0] *= l_weight
    return f


# --------------------------------------------------------------------------- k-means
def kmeans(x, k: int, n_init: int = 10, max_iter: int = 100, seed: int = 0):
    """k-means from scratch: k-means++ seeding, Lloyd iterations, best of n_init runs.

    Returns (centres (k, d), labels (n,), inertia = sum of squared distances to the centres).
    An empty cluster is re-seeded with the point farthest from its centre.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < k:
        raise ValueError(f"need at least k={k} points, got {n}")
    rng = np.random.default_rng(seed)
    best = None

    for _ in range(n_init):

        # --------------------------------------------------
        # 1. K-means++ initialization
        # --------------------------------------------------

        # First centre: uniform random observation
        first_idx = rng.integers(n)
        centres = [x[first_idx].copy()]

        # Add the remaining k - 1 centres
        for _ in range(1, k):

            # Squared distance to the nearest chosen centre
            d2 = np.min(
                ((x[:, None, :] - np.asarray(centres)[None, :, :]) ** 2).sum(axis=2),
                axis=1
            )

            # Probability proportional to squared distance
            total = d2.sum()

            if total == 0:
                # All observations are identical to chosen centres
                candidates = np.setdiff1d(
                    np.arange(n),
                    np.array([
                        np.argmin(((x - c) ** 2).sum(axis=1))
                        for c in centres
                    ])
                )
                idx = rng.choice(candidates) if len(candidates) else rng.integers(n)
            else:
                probabilities = d2 / total
                idx = rng.choice(n, p=probabilities)

            centres.append(x[idx].copy())

        centres = np.asarray(centres, dtype=np.float64)

        # --------------------------------------------------
        # 2. Lloyd iterations
        # --------------------------------------------------

        labels = None

        for _ in range(max_iter):

            # Assignment: assign each point to its nearest centre
            d2 = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            new_labels = np.argmin(d2, axis=1)

            # Stop if labels no longer change
            if labels is not None and np.array_equal(new_labels, labels):
                labels = new_labels
                break

            labels = new_labels

            # Update: recompute each centre as the mean of its cluster
            new_centres = centres.copy()

            for j in range(k):
                members = x[labels == j]

                if len(members) > 0:
                    new_centres[j] = members.mean(axis=0)

                else:
                    # Re-seed an empty cluster with the point farthest
                    # from its currently assigned centre
                    assigned_d2 = d2[np.arange(n), labels]
                    farthest_idx = np.argmax(assigned_d2)
                    new_centres[j] = x[farthest_idx]

            centres = new_centres

        # --------------------------------------------------
        # 3. Compute inertia
        # --------------------------------------------------

        final_d2 = ((x - centres[labels]) ** 2).sum(axis=1)
        inertia = final_d2.sum()

        # Keep the best initialization
        if best is None or inertia < best[2]:
            best = (centres.copy(), labels.copy(), float(inertia))

    return best



def fit_team_colours(feats, k: int = 3, l_weight: float = 0.5, seed: int = 0):
    """k-means on the colours of one clip; the two largest clusters are the teams (the others
    gather referees, goalkeepers and noise). Returns (centres (k, 3) in colour_space,
    team_clusters (2,) cluster ids sorted by size, labels)."""
    f = colour_space(feats, l_weight)
    centres, labels, _ = kmeans(f, k, seed=seed)
    sizes = np.bincount(labels, minlength=k)
    return centres, np.argsort(-sizes)[:2], labels


# --------------------------------------------------------------------------- team of each track
def assign_teams(feats, valid, track_ids, centres, team_clusters, margin: float = 0.15,
                 max_dist: float = 25.0, min_frames: int = 3, l_weight: float = 0.5,
                 level: str = "track"):
    """Team (0 or 1, the index in team_clusters) or UNKNOWN for each detection.

    For a feature f, with d0, d1 its distances to the two team centres (colour_space): team
    argmin(d0, d1) if min(d0, d1) <= max_dist and |d0 - d1| / (d0 + d1) >= margin, else UNKNOWN.
    level="frame": the rule is applied to each detection alone. level="track": it is applied once
    per track to the median of its valid features, and the answer is given to all its detections;
    a track with fewer than min_frames valid features is UNKNOWN. Returns (n,) int array.
    """
    f = colour_space(feats, l_weight)
    valid = np.asarray(valid, dtype=bool)
    track_ids = np.asarray(track_ids)
    c = np.asarray(centres, dtype=np.float64)[np.asarray(team_clusters)]

    def rule(F):
        d = np.sqrt(((F[:, None, :] - c[None]) ** 2).sum(-1))           # (m, 2)
        team = np.argmin(d, axis=1)
        rel = np.abs(d[:, 0] - d[:, 1]) / np.maximum(d.sum(axis=1), 1e-12)
        ok = (d.min(axis=1) <= max_dist) & (rel >= margin)
        return np.where(ok, team, UNKNOWN)

    out = np.full(len(f), UNKNOWN, dtype=np.int64)
    if level == "frame":
        # Classify each valid detection independently
        out[valid] = rule(f[valid])

    elif level == "track":
        # Process each track separately
        for tid in np.unique(track_ids):

            # Indices of all detections belonging to this track
            idx = np.flatnonzero(track_ids == tid)

            # Keep only valid detections of this track
            idx_valid = idx[valid[idx]]

            # Check the minimum number of valid observations
            if len(idx_valid) < min_frames:
                continue

            # Compute the median colour of the track
            track_feature = np.median(f[idx_valid], axis=0)

            # Classify this representative feature
            team = rule(track_feature[None, :])[0]

            # Assign the same classification to all detections
            out[idx] = team

    else:
        raise ValueError("level must be 'frame' or 'track'")

    return out


def match_clusters_to_labels(pred, labels) -> dict:
    """Best mapping {0, 1} -> {"left", "right"} for one clip (evaluation only): the permutation
    that agrees with the most labels. UNKNOWN predictions and missing labels are ignored."""
    pred, labels = np.asarray(pred), np.asarray(labels, dtype=object)
    keep = (pred != UNKNOWN) & np.array([l in ("left", "right") for l in labels])
    a = {0: "left", 1: "right"}
    b = {0: "right", 1: "left"}
    agree = lambda m: sum(m[p] == l for p, l in zip(pred[keep], labels[keep]))
    return a if agree(a) >= agree(b) else b
