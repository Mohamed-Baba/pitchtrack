"""Camera motion compensation.

A broadcast camera pans and zooms to follow the ball: a player standing still moves in the
image. Between two consecutive frames, the background (the pitch, its lines, the boards)
moves by an approximately affine map p' = M p + b. We estimate it from sparse feature
points OUTSIDE the player boxes, and express the Kalman states in the new frame before
associating. Too few reliable points means a scene cut: the tracker is reset.
"""
from __future__ import annotations

import cv2
import numpy as np

from .boxes import as_boxes

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


# --------------------------------------------------------------------------- camera motion
def estimate_camera_motion(prev_gray: np.ndarray, gray: np.ndarray, exclude_boxes=None,
                           max_corners: int = 400, ransac_px: float = 3.0,
                           min_inliers: int = 20) -> tuple[np.ndarray, bool]:
    """Partial affine map (rotation, uniform scale, translation) of the background.

    1. corners of prev_gray outside exclude_boxes (cv2.goodFeaturesToTrack with a mask,
       quality 0.01, min distance 8 px); 2. their positions in gray (cv2.calcOpticalFlowPyrLK),
    keep status == 1; 3. cv2.estimateAffinePartial2D with RANSAC (threshold ransac_px).
    Returns (A (2, 3) float64, ok). On failure, or with fewer than min_inliers inliers:
    (IDENTITY, False).
    """

    h, w = prev_gray.shape[:2]

    # Step 1: Create a mask
    # 255 = allowed pixels, 0 = excluded pixels
    mask = np.full((h, w), 255, dtype=np.uint8)

    # Exclude player bounding boxes
    if exclude_boxes is not None:
        for box in as_boxes(exclude_boxes):
            x1, y1, x2, y2 = box

            # Clip the box to image boundaries
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))

            # Mask only valid boxes
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0
    # ------------------------------------------------------------
    # Step 2: Detect corners in the previous frame
    # ------------------------------------------------------------

    points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=8,
        mask=mask
    )

    if points is None or len(points) < min_inliers:
        return IDENTITY.copy(), False

    # ------------------------------------------------------------
    # Step 3: Track points from prev_gray to gray
    # ------------------------------------------------------------

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        points,
        None
    )

    if next_points is None or status is None:
        return IDENTITY.copy(), False

    # Keep only successfully tracked points
    status = status.ravel().astype(bool)

    old_pts = points.reshape(-1, 2)[status]
    new_pts = next_points.reshape(-1, 2)[status]

    if len(old_pts) < min_inliers:
        return IDENTITY.copy(), False

    # ------------------------------------------------------------
    # Step 4: Estimate partial affine transformation with RANSAC
    # ------------------------------------------------------------

    A, inliers = cv2.estimateAffinePartial2D(
        old_pts,
        new_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_px
    )

    if A is None or inliers is None:
        return IDENTITY.copy(), False

    # Check the number of reliable correspondences
    if int(inliers.sum()) < min_inliers:
        return IDENTITY.copy(), False

    return A.astype(np.float64), True



# --------------------------------------------------------------------------- tracks in the new frame
def warp_track_state(mean, cov, A) -> tuple[np.ndarray, np.ndarray]:
    """Express a Kalman state (cx, cy, w, h, vcx, vcy, vw, vh) and its covariance after the
    camera motion A = [M | b] (2 x 3).

    Centre: c' = M c + b. Sizes: multiplied by s = sqrt(|det M|) (uniform scale of a partial
    affine map). Velocities: the linear part only, v' = M v and (vw, vh)' = s (vw, vh).
    Covariance: P' = T P T^T with T = blockdiag(M, s I2, M, s I2).
    """
    A = np.asarray(A, dtype=np.float64)
    
    # Extract the linear transformation and translation
    M = A[:, :2]
    b = A[:, 2]

    # Compute the uniform scale factor
    s = np.sqrt(abs(np.linalg.det(M)))

    # Build the 8x8 transformation matrix
    T = np.zeros((8, 8), dtype=np.float64)

    T[0:2, 0:2] = M
    T[2:4, 2:4] = s * np.eye(2)
    T[4:6, 4:6] = M
    T[6:8, 6:8] = s * np.eye(2)

    # Build the translation vector
    d = np.zeros(8, dtype=np.float64)
    d[:2] = b

    # Transform the Kalman state
    mean = T @ mean + d

    # Transform the covariance
    cov = T @ cov @ T.T

    return mean, cov



def warp_box(box, A) -> np.ndarray:
    """xyxy box after the camera motion (centre mapped by A, size scaled by s)."""
    A = np.asarray(A, dtype=np.float64)
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    c = A[:, :2] @ np.array([(x1 + x2) / 2, (y1 + y2) / 2]) + A[:, 2]
    s = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
    w, h = s * (x2 - x1), s * (y2 - y1)
    return np.array([c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2])


class CameraMotionEstimator:
    """Frame-to-frame camera motion on downscaled grey images.

    Call it on consecutive frames in time order with the boxes to ignore; returns the affine
    map in FULL-resolution coordinates (M unchanged, b divided by the scale) and ok.
    """

    def __init__(self, scale: float = 0.5, **kwargs):
        self.scale, self.kwargs, self.prev = scale, kwargs, None

    def __call__(self, image_bgr: np.ndarray, exclude_boxes=None) -> tuple[np.ndarray, bool]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if self.scale != 1.0:
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        prev, self.prev = self.prev, gray
        if prev is None:
            return IDENTITY.copy(), True
        boxes = as_boxes(exclude_boxes if exclude_boxes is not None else []) * self.scale
        A, ok = estimate_camera_motion(prev, gray, boxes, **self.kwargs)
        A = A.copy()
        A[:, 2] /= self.scale
        return A, ok
