"""Box geometry shared by every stage of pitchtrack.

Conventions (used everywhere in the project)
--------------------------------------------
* Coordinates are continuous pixel coordinates in the ORIGINAL frame: origin at the
  top-left corner of the top-left pixel, x to the right, y downwards. Pixel (row i,
  column j) covers the square [j, j+1) x [i, i+1).
* A box is a row (x1, y1, x2, y2) with x1 <= x2 and y1 <= y2 ("xyxy"). Its width is
  x2 - x1 (no "+1"). MOTChallenge files store (left, top, width, height) ("xywh");
  the Kalman filter uses (cx, cy, w, h) ("cxcywh").
* An array of boxes has shape (N, 4) and dtype float64; N may be 0.
* The foot point of a player is the bottom-middle of the box, ((x1 + x2) / 2, y2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def as_boxes(boxes) -> np.ndarray:
    """Return `boxes` as a float64 array of shape (N, 4) (accepts lists and (4,) arrays)."""
    b = np.asarray(boxes, dtype=np.float64)
    if b.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    if b.ndim != 2 or b.shape[1] != 4:
        raise ValueError(f"expected boxes of shape (N, 4), got {b.shape}")
    return b


# --------------------------------------------------------------------------- format conversions
def xywh_to_xyxy(boxes) -> np.ndarray:
    """(left, top, width, height) -> (x1, y1, x2, y2). Shape (N, 4) -> (N, 4), new array."""
    b = as_boxes(boxes)
    # x2 = left + width, y2 = top + height, in a new array (the input is not modified)
    out = b.copy()
    out[:,2] = b[:,0] + b[:,2]
    out[:,3] = b[:,1] + b[:,3]
    return out
    

def xyxy_to_xywh(boxes) -> np.ndarray:
    """(x1, y1, x2, y2) -> (left, top, width, height). Shape (N, 4) -> (N, 4), new array."""
    b = as_boxes(boxes)
    out = b.copy()
    out[:,2] = b[:,2] - b[:,0]
    out[:,3] = b[:,3] - b[:,1]
    return out

def xyxy_to_cxcywh(boxes) -> np.ndarray:
    """(x1, y1, x2, y2) -> (centre x, centre y, width, height). Shape (N, 4) -> (N, 4)."""
    b = as_boxes(boxes)
    # centre = mean of the two corners; size = difference of the corners.
    out = b.copy()
    out[:,0] = (b[:,0] + b[:,2])/2
    out[:,1] = (b[:,1] + b[:,3])/2
    out[:,2] = b[:,2] - b[:,0]
    out[:,3] = b[:,3] - b[:,1]
    return out

def cxcywh_to_xyxy(boxes) -> np.ndarray:
    """(centre x, centre y, width, height) -> (x1, y1, x2, y2). Shape (N, 4) -> (N, 4)."""
    b = as_boxes(boxes)
    half = b[:, 2:] / 2.0
    return np.concatenate([b[:, :2] - half, b[:, :2] + half], axis=1)


def box_area(boxes) -> np.ndarray:
    """Areas (N,) of xyxy boxes; a degenerate box (x2 < x1 or y2 < y1) has area 0."""
    b = as_boxes(boxes)
    # width and height clipped at 0, then their product.
    areas = []
    width = np.maximum(b[:, 2] - b[:, 0], 0)
    height = np.maximum(b[:, 3] - b[:, 1], 0)
    areas = width * height
    return areas


def foot_points(boxes) -> np.ndarray:
    """Bottom-middle points ((x1 + x2) / 2, y2) of xyxy boxes, shape (N, 2)."""
    b = as_boxes(boxes)
    return np.stack([(b[:, 0] + b[:, 2]) / 2.0, b[:, 3]], axis=1)


# --------------------------------------------------------------------------- overlaps
def box_intersection(a, b) -> np.ndarray:
    """Pairwise intersection areas between a (N, 4) and b (M, 4): array (N, M)."""
    a, b = as_boxes(a), as_boxes(b)
    # The overlap of two intervals [p1, p2] and [q1, q2] is max(0, min(p2, q2) - max(p1, q1)).
    
    # Upper-left corner of each intersection: shape (N, M, 2)
    left_top = np.maximum(a[:, None, :2], b[None, :, :2])

    # Bottom-right corner of each intersection: shape (N, M, 2)
    right_bottom = np.minimum(a[:, None, 2:], b[None, :, 2:])

    # Intersection width and height, clipped at 0: shape (N, M, 2)
    inter_wh = np.maximum(right_bottom - left_top, 0)

    # Intersection areas: shape (N, M)
    return inter_wh[..., 0] * inter_wh[..., 1]


def box_iou(a, b) -> np.ndarray:
    """Pairwise intersection over union (N, M). Convention: IoU = 0 where the union is 0."""
    a, b = as_boxes(a), as_boxes(b)
    # IoU = inter / (area_a + area_b - inter). Never return NaN: use
    # np.divide(..., out=np.zeros_like(inter), where=union > 0).

    # Pairwise intersection areas: shape (N, M)
    inter = box_intersection(a, b)

    # Individual areas: shape (N,) and (M,)
    area_a = box_area(a)
    area_b = box_area(b)

    # Pairwise union areas: shape (N, M)
    union = area_a[:, None] + area_b[None, :] - inter

    # IoU: shape (N, M); return 0 where union == 0
    return np.divide(inter,union,out=np.zeros_like(inter),where=union > 0)


def box_ios(a, b) -> np.ndarray:
    """Pairwise intersection over the SMALLER of the two areas (N, M); 0 if that area is 0.

    IoS is close to 1 when one box is (almost) inside the other, even if their sizes
    differ a lot: it is the right criterion to merge a box cut by a tile border with
    the full box seen in the neighbouring tile.
    """
    a, b = as_boxes(a), as_boxes(b)

    # Pairwise intersection areas: shape (N, M)
    inter = box_intersection(a, b)
    
    # Individual areas: shape (N,) and (M,)
    area_a = box_area(a)
    area_b = box_area(b)
    
    # Pairwise smaller areas: shape (N, M)
    smaller = np.minimum(area_a[:,None], area_b[None,:])
    
    # IoU: shape (N, M); return 0 where smaller == 0
    return np.divide(inter,smaller,out=np.zeros_like(inter),where=smaller > 0)
    


# --------------------------------------------------------------------------- resizing, cropping, letterboxing
@dataclass(frozen=True)
class BoxTransform:
    """Axis-aligned affine map of the image plane, u' = sx * u + tx, v' = sy * v + ty.

    sx and sy must be > 0. Such a map sends xyxy boxes to xyxy boxes. Typical uses:
    resizing (tx = ty = 0), cropping a tile (sx = sy = 1), letterboxing (scale + pad).
    """

    sx: float
    sy: float
    tx: float = 0.0
    ty: float = 0.0

    def __post_init__(self):
        if self.sx <= 0 or self.sy <= 0:
            raise ValueError("scales must be positive")

    def matrix(self) -> np.ndarray:
        """3 x 3 matrix acting on homogeneous coordinates (u, v, 1)."""
        return np.array([[self.sx, 0.0, self.tx], [0.0, self.sy, self.ty], [0.0, 0.0, 1.0]])

    def apply(self, boxes) -> np.ndarray:
        """Map xyxy boxes (N, 4) to xyxy boxes (N, 4)."""
        b = as_boxes(boxes)
        # apply the map to both corners (x columns 0 and 2, y columns 1 and 3).
        
        # Create an independent copy
        out = b.copy()

        # Transform x coordinates: columns 0 and 2
        out[:, [0, 2]] = self.sx * b[:, [0, 2]] + self.tx

        # Transform y coordinates: columns 1 and 3
        out[:, [1, 3]] = self.sy * b[:, [1, 3]] + self.ty

        return out

    def apply_points(self, points) -> np.ndarray:
        """Map points (N, 2) given as (u, v)."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return p * np.array([self.sx, self.sy]) + np.array([self.tx, self.ty])

    def inverse(self) -> "BoxTransform":
        """The map T^{-1} such that T^{-1}.apply(T.apply(b)) == b."""
        # solve u' = sx * u + tx for u (and the same for v).
        return BoxTransform(1/self.sx , 1/self.sy , -self.tx/self.sx , -self.ty/self.sy)

    def compose(self, first: "BoxTransform") -> "BoxTransform":
        """The map `self o first`: apply `first`, then `self`."""
        return BoxTransform(
            self.sx * first.sx, self.sy * first.sy,
            self.sx * first.tx + self.tx, self.sy * first.ty + self.ty,
        )


def letterbox_transform(src_hw, dst_hw) -> BoxTransform:
    """Map from an H x W source image to a letterboxed dst_h x dst_w canvas.

    The image is scaled by r = min(dst_h / H, dst_w / W) (aspect ratio kept), then
    centred: pad_x = (dst_w - r W) / 2 and pad_y = (dst_h - r H) / 2.
    """
    (h, w), (dh, dw) = src_hw, dst_hw
    # Compute the uniform scaling factor
    r = min(dh / h, dw / w)

    # Compute the padding required to centre the resized image
    pad_x = (dw - r * w) / 2
    pad_y = (dh - r * h) / 2

    # Return the corresponding geometric transformation
    return BoxTransform(r, r, pad_x, pad_y)


def to_yolo_labels(boxes, width: int, height: int, cls: int = 0) -> list[str]:
    """xyxy pixel boxes -> YOLO label lines "cls cx cy w h" (normalised), clipped to the image."""
    b = as_boxes(boxes).copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, width)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, height)
    b = b[(b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])]
    c = np.concatenate([(b[:, :2] + b[:, 2:]) / 2.0, b[:, 2:] - b[:, :2]], axis=1)
    c = c / np.array([width, height, width, height], dtype=np.float64)
    return [f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for x, y, w, h in c]


# --------------------------------------------------------------------------- tiles
def _tile_starts(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    step = tile * (1.0 - overlap)
    n = int(np.ceil((length - tile) / step)) + 1
    return [int(round(s)) for s in np.linspace(0, length - tile, n)]


def tile_grid(frame_hw, tile: int, overlap: float) -> np.ndarray:
    """Overlapping square tiles covering a frame, as xyxy boxes (T, 4).

    Tiles are spread evenly; the last row and column touch the frame border, so the
    real overlap is at least `overlap` (a fraction of the tile size). A side shorter
    than `tile` gets a single, clipped tile.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    h, w = frame_hw
    xs, ys = _tile_starts(w, tile, overlap), _tile_starts(h, tile, overlap)
    return np.array(
        [[x, y, min(x + tile, w), min(y + tile, h)] for y in ys for x in xs], dtype=np.float64
    )
