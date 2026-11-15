"""Image <-> pitch homography from the pitch lines.

G maps the image to the pitch and H = G^-1 the pitch to the image:

    (X, Y, 1)^T ~ G (u, v, 1)^T          (u, v, 1)^T ~ H (X, Y, 1)^T

SoccerNet-GSR does not give point correspondences: for most frames it gives points ON the pitch
lines (one polyline per class, normalised image coordinates). A point p on the image of the
ground line l (pitch coordinates, l^T P = 0) satisfies l^T G p = 0: one linear equation in the
nine entries of G per annotated point. Circles give quadratic equations; they are kept out of
the estimation and serve as held-out points in the quality checks.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import pitch_model as pm


# --------------------------------------------------------------------------- parsing
@dataclass
class LinePoints:
    """Annotated points of one frame, in pixels."""

    pts: np.ndarray            # (n, 2) points on straight ground lines
    names: list                # class of each of them
    lines: np.ndarray          # (n, 3) pitch line of each of them, (a, b, c) with a^2 + b^2 = 1
    circle_pts: np.ndarray     # (m, 2) points on the circles
    circle_names: list


def parse_lines(lines: dict, width: int, height: int) -> LinePoints:
    """Raw GSR "lines" dict -> pixels. As in SoccerNet's code, u = x (W - 1), v = y (H - 1)."""
    pts, names, circ, cnames = [], [], [], []
    for name, polyline in (lines or {}).items():
        name = name.strip()
        xy = [(float(p["x"]) * (width - 1), float(p["y"]) * (height - 1)) for p in polyline]
        if pm.is_ground_line(name):
            pts += xy
            names += [name] * len(xy)
        elif pm.is_circle(name):
            circ += xy
            cnames += [name] * len(xy)
    arr = lambda a: np.asarray(a, dtype=np.float64).reshape(-1, 2)
    lines_arr = np.array([pm.LINES[n] for n in names], dtype=np.float64).reshape(-1, 3)
    return LinePoints(arr(pts), names, lines_arr, arr(circ), cnames)


def line_groups(lines) -> dict:
    """Infinite line -> indices of its points. Keys ("x", value) or ("y", value) as pm.LINE_KEY."""
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 3)
    groups: dict = {}
    for i, (a, _, c) in enumerate(lines):
        key = ("x" if abs(a) > 0.5 else "y", round(-c, 3))
        groups.setdefault(key, []).append(i)
    return {k: np.asarray(v) for k, v in groups.items()}


def well_posed(lines) -> bool:
    """True if the lines contain two distinct x-lines and two distinct y-lines: with a single line in one direction, a
    one-parameter family of homographies fits every point."""
    keys = line_groups(lines).keys()
    return sum(k[0] == "x" for k in keys) >= 2 and sum(k[0] == "y" for k in keys) >= 2


def similarity_normaliser(pts) -> np.ndarray:
    """3x3 similarity T such that the points T p have centroid 0 and mean norm sqrt(2) (Hartley)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    c = pts.mean(axis=0)
    d = np.mean(np.hypot(*(pts - c).T))
    s = np.sqrt(2.0) / d if d > 0 else 1.0
    return np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])


PITCH_NORMALISER = np.diag([1.0 / pm.HALF_L, 1.0 / pm.HALF_L, 1.0])   # metres -> about [-1, 1]


def homogeneous(pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return np.c_[pts, np.ones(len(pts))]


# --------------------------------------------------------------------------- homography from lines
def dlt_from_lines(pts, lines) -> np.ndarray | None:
    """G (image -> pitch) from points on known pitch lines, by the normalised DLT.

    pts (n, 2) pixels; lines (n, 3) the pitch line of each point. Returns None if the lines do
    not determine G (`well_posed` is False) or if n < 8. G has unit Frobenius norm and the sign
    that makes the third coordinate of G p positive for most of the given points: they lie on
    the ground in front of the camera, and image_to_pitch relies on this sign.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 8 or not well_posed(lines):
        return None
    if len(pts) != len(lines):
        raise ValueError("pts and lines must have the same number of elements")
    # 1. Normalize image points and pitch lines
    T = similarity_normaliser(pts)
    U = PITCH_NORMALISER

    p = homogeneous(pts)
    p_norm = (T @ p.T).T

    U_inv = np.linalg.inv(U)

    lines_norm = (U_inv.T @ lines.T).T

    # Normalize each line so that its first two coefficients
    # have unit Euclidean norm
    line_norms = np.linalg.norm(
        lines_norm[:, :2],
        axis=1,
        keepdims=True
    )

    if np.any(line_norms < 1e-12):
        return None

    lines_norm = lines_norm / line_norms

    # 2. Build the linear system A g = 0
    A = np.asarray([
        np.kron(line, point)
        for line, point in zip(lines_norm, p_norm)
    ])

    # 3. Solve using SVD
    _, _, Vh = np.linalg.svd(A, full_matrices=True)

    G_norm = Vh[-1].reshape(3, 3)

    # 4. Denormalize the homography
    G = U_inv @ G_norm @ T

    # 5. Normalize G to have unit Frobenius norm
    G_norm_value = np.linalg.norm(G)

    if G_norm_value < 1e-12:
        return None

    G = G / G_norm_value

    # 6. Choose the sign such that most projected points
    # have a positive third homogeneous coordinate
    projected = (G @ p.T).T

    if np.median(projected[:, 2]) < 0:
        G = -G

    return G


def line_residuals(G, pts, lines) -> np.ndarray:
    """Distance in pixels from each point to the image of its pitch line, l_img = G^T l."""
    li = np.asarray(lines, dtype=np.float64).reshape(-1, 3) @ G       # rows (G^T l)^T
    num = np.abs(np.sum(li * homogeneous(pts), axis=1))
    return num / np.maximum(np.hypot(li[:, 0], li[:, 1]), 1e-12)


# --------------------------------------------------------------------------- RANSAC on lines
def ransac_lines(pts,lines,thr_px: float = 4.0,p_success: float = 0.99,max_iter: int = 300,seed: int = 0):
    """Robust G from annotated line points.

    A minimal sample is two distinct x-lines and two distinct y-lines,
    fitted with ALL their points.

    A point is an inlier if it lies within thr_px of the image of its line;
    a line is an inlier if at least half of its points are.

    Models that mirror the pitch (det G <= 0) are rejected.

    The consensus counts inlier LINES first, then inlier points,
    then prefers the smaller median residual.

    Returns (G or None, inlier mask (n,), number of iterations).
    """

    # --------------------------------------------------
    # 1. Prepare the input data
    # --------------------------------------------------

    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 3)

    if len(pts) != len(lines):
        raise ValueError("pts and lines must have the same number of elements")

    none = (None, np.zeros(len(pts), dtype=bool), 0)

    if len(pts) < 8 or not well_posed(lines):
        return none

    if not (0.0 < p_success < 1.0):
        raise ValueError("p_success must be between 0 and 1")

    if thr_px <= 0:
        raise ValueError("thr_px must be positive")

    if max_iter <= 0:
        return none

    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # 2. Group points according to their pitch lines
    # --------------------------------------------------

    groups = line_groups(lines)

    xkeys = [k for k in groups if k[0] == "x"]
    ykeys = [k for k in groups if k[0] == "y"]

    if len(xkeys) < 2 or len(ykeys) < 2:
        return none

    # --------------------------------------------------
    # 3. Initialize the best model and iteration count
    # --------------------------------------------------

    best_G = None
    best_inliers = None
    best_score = None

    it = 0
    N = max_iter

    # --------------------------------------------------
    # 4. RANSAC loop
    # --------------------------------------------------

    while it < min(N, max_iter):

        it += 1

        # 4.1 Select two distinct x-lines and two distinct y-lines

        idx_x = rng.choice(len(xkeys), size=2, replace=False)
        idx_y = rng.choice(len(ykeys), size=2, replace=False)

        selected_keys = (
            [xkeys[i] for i in idx_x]
            + [ykeys[i] for i in idx_y]
        )

        # 4.2 Retrieve ALL points belonging to the four selected lines

        sample_idx = np.concatenate([
            np.asarray(groups[key], dtype=np.int64)
            for key in selected_keys
        ])

        sample_idx = np.unique(sample_idx)

        sample_pts = pts[sample_idx]
        sample_lines = lines[sample_idx]

        # 4.3 Estimate a candidate homography (DLT on the four lines)

        G = dlt_from_lines(sample_pts, sample_lines)

        if G is None:
            continue

        # 4.4 Reject homographies that mirror the pitch

        det_G = np.linalg.det(G)

        if not np.isfinite(det_G) or det_G <= 0:
            continue

        # 4.5 Compute residuals for ALL annotated points

        residuals = line_residuals(G, pts, lines)

        # A point is an inlier if its residual is below the threshold

        inliers = np.isfinite(residuals) & (residuals < thr_px)

        # --------------------------------------------------
        # 5. Count the inlier lines
        # --------------------------------------------------

        n_inlier_lines = 0

        for indices in groups.values():

            indices = np.asarray(indices, dtype=np.int64)

            n_inliers_on_line = np.count_nonzero(inliers[indices])

            # A line is an inlier if at least half its points are inliers

            if 2 * n_inliers_on_line >= len(indices):
                n_inlier_lines += 1

        # Count the individual inlier points

        n_inlier_points = np.count_nonzero(inliers)

        if n_inlier_points == 0:
            continue

        # Median residual among the inlier points

        median_residual = np.median(residuals[inliers])

        # --------------------------------------------------
        # 6. Score the candidate homography
        # --------------------------------------------------

        score = (
            n_inlier_lines,
            n_inlier_points,
            -median_residual
        )

        # Python compares tuples lexicographically:
        # 1. More inlier lines is better.
        # 2. If tied, more inlier points is better.
        # 3. If still tied, smaller median residual is better.

        if best_score is None or score > best_score:

            best_score = score
            best_G = G.copy()
            best_inliers = inliers.copy()

            # --------------------------------------------------
            # 7. Update the adaptive number of iterations
            # --------------------------------------------------

            w = n_inlier_lines / len(groups)

            if w >= 1.0:

                # All lines are inliers: no more iterations needed

                N = 0

            elif w > 0.0:

                numerator = np.log1p(-p_success)
                denominator = np.log1p(-(w ** 4))

                N_required = int(
                    np.ceil(numerator / denominator)
                )

                # Keep the smaller iteration limit

                N = min(N, max(1, N_required))

    # --------------------------------------------------
    # 8. Check whether a valid model was found
    # --------------------------------------------------

    if best_G is None:
        return none

    # --------------------------------------------------
    # 9. Refit the homography using all best inlier points
    # --------------------------------------------------

    G_refit = dlt_from_lines(
        pts[best_inliers],
        lines[best_inliers]
    )

    # If refitting succeeds and does not mirror the pitch,
    # use the refined model and recompute the inliers.

    if G_refit is not None:

        det_refit = np.linalg.det(G_refit)

        if np.isfinite(det_refit) and det_refit > 0:

            residuals_refit = line_residuals(
                G_refit,
                pts,
                lines
            )

            final_inliers = (
                np.isfinite(residuals_refit)
                & (residuals_refit < thr_px)
            )

            return G_refit, final_inliers, it

    # If refitting fails, retain the best valid model found.

    return best_G, best_inliers, it



# --------------------------------------------------------------------------- projection
def image_to_pitch(G, pts, margin: float = 5.0):
    """Project image points to the pitch (for a player: the bottom-middle of the box).

    Returns (xy (n, 2) metres with NaN where invalid, valid (n,) bool).
    A point is valid if its third coordinate is positive (below the horizon:
    G has the sign set by dlt_from_lines), the result is finite, and it lies inside
    the pitch enlarged by `margin` metres.
    """

    q = homogeneous(pts) @ np.asarray(G, dtype=np.float64).T

    # Third homogeneous coordinate
    w = q[:, 2]

    # Initialize all projected coordinates as invalid
    xy = np.full((len(pts), 2), np.nan, dtype=np.float64)

    # Only divide when w is finite and strictly positive
    positive_w = np.isfinite(w) & (w > 0)

    # Convert homogeneous coordinates to Cartesian pitch coordinates
    xy[positive_w] = q[positive_w, :2] / w[positive_w, None]

    # Check that the projected coordinates are finite
    finite = np.isfinite(xy).all(axis=1)

    # Candidate points: positive w and finite coordinates
    candidate = positive_w & finite

    # Check whether candidate points lie inside the enlarged pitch
    inside = np.zeros(len(pts), dtype=bool)
    inside[candidate] = pm.inside_pitch(xy[candidate], margin)

    # Final validity mask
    valid = candidate & inside

    # Invalid points must have NaN coordinates
    xy[~valid] = np.nan

    return xy, valid



def pitch_to_image(H, xy) -> np.ndarray:
    """Pitch points (metres) -> pixels with H = G^-1; NaN for points behind the camera."""
    q = homogeneous(xy) @ np.asarray(H, dtype=np.float64).T
    out = np.full((len(q), 2), np.nan)
    ok = q[:, 2] > 1e-12
    out[ok] = q[ok, :2] / q[ok, 2:3]
    return out


# --------------------------------------------------------------------------- uncertainty
def pitch_covariance(G, pts, cov_px) -> np.ndarray:
    """First-order covariance (n, 2, 2), in m^2, of the projections of noisy image points.

    cov_px: (2, 2) or (n, 2, 2) covariance of the points in pixels^2. With (X, Y) = g(u, v),
    Sigma_XY = J Sigma_uv J^T and J = d(X, Y) / d(u, v).
    """

    G = np.asarray(G, dtype=np.float64)

    # Project points to homogeneous pitch coordinates
    q = homogeneous(pts) @ G.T

    # Allow either one common covariance or one covariance per point
    cov_px = np.broadcast_to(np.asarray(cov_px, dtype=np.float64),(len(q), 2, 2))

    # Third homogeneous coordinate
    w = q[:, 2]

    # Initialize Jacobian with NaN
    J = np.full((len(q), 2, 2), np.nan, dtype=np.float64)

    # Avoid division by zero and invalid homogeneous coordinates
    valid = np.isfinite(q).all(axis=1) & (np.abs(w) > 1e-12)

    # Cartesian coordinates on the pitch
    X = q[valid, 0] / w[valid]
    Y = q[valid, 1] / w[valid]

    # First row: derivatives of X with respect to u and v
    J[valid, 0, :] = (
        G[0, :2] - X[:, None] * G[2, :2]
    ) / w[valid, None]

    # Second row: derivatives of Y with respect to u and v
    J[valid, 1, :] = (
        G[1, :2] - Y[:, None] * G[2, :2]
    ) / w[valid, None]

    # Covariance propagation: Sigma_XY = J Sigma_uv J^T
    cov_pitch = J @ cov_px @ np.swapaxes(J, 1, 2)

    return cov_pitch

# --------------------------------------------------------------------------- quality
@dataclass
class Calibration:
    """Result for one frame. `source`: "lines" (estimated on this frame), "propagated" (moved
    from a neighbouring frame with the image motion, then checked), or "none"."""

    G: np.ndarray | None
    ok: bool
    source: str
    reason: str = ""
    n_points: int = 0
    n_inliers: int = 0
    n_xlines: int = 0
    n_ylines: int = 0
    median_px: float = np.nan
    p90_px: float = np.nan
    circle_m: float = np.nan
    hull_frac: float = np.nan


DEFAULT_QUALITY = {"thr_px": 4.0, "min_inliers": 20, "max_median_px": 2.5, "max_circle_m": 0.6,
                   "min_hull_frac": 0.05, "max_iter": 300, "propagate_max_s": 2.0,
                   "propagated_max_median_px": 4.0}


def quality(G, lp: LinePoints, inliers, image_size, cfg: dict | None = None) -> Calibration:
    """Quality checks of one estimate. image_size = (width, height).

    (i) number of inliers and their spread (area of their convex hull / image area), both
    directions present among the inliers; (ii) error on held-out points: distance in metres of
    the projected circle points to their circle; (iii) plausibility: the inliers and the bottom
    of the image are below the horizon; (v) residuals of the line points (pixels).
    The temporal stability (iv) is measured across frames by scripts/eval_calibration.py.
    """
    cfg = {**DEFAULT_QUALITY, **(cfg or {})}
    out = Calibration(G=G, ok=False, source="lines", n_points=len(lp.pts))
    if G is None:
        out.reason = "no model (fewer than 2 x-lines and 2 y-lines, or RANSAC failed)"
        return out
    inliers = np.asarray(inliers, dtype=bool)
    res = line_residuals(G, lp.pts, lp.lines)
    keys = line_groups(lp.lines[inliers]).keys()
    out.n_inliers = int(inliers.sum())
    out.n_xlines = sum(k[0] == "x" for k in keys)
    out.n_ylines = sum(k[0] == "y" for k in keys)
    out.median_px = float(np.median(res[inliers])) if inliers.any() else np.inf
    out.p90_px = float(np.percentile(res[inliers], 90)) if inliers.any() else np.inf
    W, H = image_size
    if out.n_inliers >= 3:
        hull = cv2.convexHull(lp.pts[inliers].astype(np.float32))
        out.hull_frac = float(cv2.contourArea(hull)) / float(W * H)
    if len(lp.circle_pts):
        xy, ok = image_to_pitch(G, lp.circle_pts, margin=20.0)
        names = np.asarray(lp.circle_names)
        errs = [pm.circle_residual(n, xy[ok & (names == n)]) for n in set(lp.circle_names)]
        errs = np.concatenate(errs) if errs else np.zeros(0)
        out.circle_m = float(np.median(errs)) if len(errs) else np.nan
    bottom = np.array([[0.0, H - 1.0], [(W - 1) / 2.0, H - 1.0], [W - 1.0, H - 1.0]])
    horizon_ok = bool(np.all(homogeneous(lp.pts[inliers]) @ G[2] > 0)) and \
        bool(image_to_pitch(G, bottom, margin=40.0)[1].all()) and np.linalg.det(G) > 0
    checks = [
        (out.n_inliers >= cfg["min_inliers"], "too few inlier points"),
        (out.n_xlines >= 2 and out.n_ylines >= 2, "inliers on fewer than 2 x-lines and 2 y-lines"),
        (out.median_px <= cfg["max_median_px"], "median residual too large"),
        (np.isnan(out.circle_m) or out.circle_m <= cfg["max_circle_m"], "circle points off their circle"),
        (np.isnan(out.hull_frac) or out.hull_frac >= cfg["min_hull_frac"], "inliers concentrated in a small region"),
        (horizon_ok, "mirrored pitch, points above the horizon or implausible bottom of the image"),
    ]
    failed = [msg for good, msg in checks if not good]
    out.ok, out.reason = not failed, "; ".join(failed)
    return out


def calibrate_frame(lines: dict, image_size, cfg: dict | None = None, seed: int = 0) -> Calibration:
    """Parse, RANSAC, quality checks. image_size = (width, height)."""
    cfg = {**DEFAULT_QUALITY, **(cfg or {})}
    lp = parse_lines(lines, *image_size)
    G, inliers, _ = ransac_lines(lp.pts, lp.lines, thr_px=cfg["thr_px"], max_iter=cfg["max_iter"], seed=seed)
    return quality(G, lp, inliers, image_size, cfg)


# --------------------------------------------------------------------------- temporal propagation
def propagate(G, M) -> np.ndarray:
    """G of the previous frame -> G of the current one, given the image motion M (p_cur ~ M p_prev):
    P ~ G_prev p_prev ~ G_prev M^-1 p_cur. Unit norm, same sign convention."""
    G2 = np.asarray(G, dtype=np.float64) @ np.linalg.inv(M)
    return G2 / np.linalg.norm(G2)


class ImageMotion:
    """Homography between consecutive frames from background corners (LK + RANSAC), players
    excluded. A pure rotation and zoom of the camera induce an exact homography."""

    def __init__(self, scale: float = 0.5, max_corners: int = 600, ransac_px: float = 2.0,
                 min_inliers: int = 40):
        self.scale, self.max_corners = scale, max_corners
        self.ransac_px, self.min_inliers = ransac_px, min_inliers
        self.prev = None

    def reset(self):
        self.prev = None

    def __call__(self, frame_bgr: np.ndarray, boxes=None):
        """Returns (M, ok) with p_cur ~ M p_prev in full-resolution pixels; (I, False) on a cut."""
        s = self.scale
        gray = cv2.cvtColor(cv2.resize(frame_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        prev, self.prev = self.prev, gray
        if prev is None:
            return np.eye(3), False
        mask = np.full(prev.shape, 255, np.uint8)
        for x1, y1, x2, y2 in (np.asarray(boxes, dtype=np.float64).reshape(-1, 4) * s if boxes is not None else []):
            mask[max(0, int(y1)):max(0, int(np.ceil(y2))), max(0, int(x1)):max(0, int(np.ceil(x2)))] = 0
        p0 = cv2.goodFeaturesToTrack(prev, self.max_corners, 0.01, 8, mask=mask)
        if p0 is None or len(p0) < self.min_inliers:
            return np.eye(3), False
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None)
        good = st.reshape(-1) == 1
        if good.sum() < self.min_inliers:
            return np.eye(3), False
        M, inl = cv2.findHomography(p0[good], p1[good], cv2.RANSAC, self.ransac_px * s)
        if M is None or inl is None or int(inl.sum()) < self.min_inliers:
            return np.eye(3), False
        S = np.diag([s, s, 1.0])
        return np.linalg.inv(S) @ M @ S, True
