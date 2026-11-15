"""Person detection: a thin wrapper around a pre-trained detector, NMS and tiled inference.

Contract: every detector returns `Detections` in xyxy pixels of the ORIGINAL frame
(see boxes.py), whatever resizing, letterboxing or tiling happened inside.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .boxes import BoxTransform, as_boxes, box_ios, box_iou, tile_grid

COCO_PERSON = 0


@dataclass
class Detections:
    """Detections of one frame: boxes (N, 4) xyxy, scores (N,), classes (N,) int."""

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray

    def __post_init__(self):
        self.boxes = as_boxes(self.boxes)
        self.scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        self.classes = np.asarray(self.classes, dtype=np.int64).reshape(-1)
        if not len(self.boxes) == len(self.scores) == len(self.classes):
            raise ValueError("boxes, scores and classes must have the same length")

    def __len__(self) -> int:
        return len(self.scores)

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64))

    @staticmethod
    def concat(items: Sequence["Detections"]) -> "Detections":
        items = [d for d in items if len(d)]
        if not items:
            return Detections.empty()
        return Detections(
            np.concatenate([d.boxes for d in items]),
            np.concatenate([d.scores for d in items]),
            np.concatenate([d.classes for d in items]),
        )

    def select(self, index) -> "Detections":
        return Detections(self.boxes[index], self.scores[index], self.classes[index])

    def transformed(self, transform: BoxTransform) -> "Detections":
        return Detections(transform.apply(self.boxes), self.scores.copy(), self.classes.copy())


# --------------------------------------------------------------------------- non-maximum suppression
def nms(boxes, scores, threshold: float, metric: str = "iou") -> np.ndarray:
    """Greedy non-maximum suppression.

    Visit the boxes by decreasing score (stable order for ties); keep the current box
    and suppress every remaining box whose similarity with it is > threshold.
    metric is "iou" (box_iou) or "ios" (box_ios). Returns the kept indices (int64),
    by decreasing score.
    """
    # Convertir les bounding boxes en tableau NumPy de shape (N, 4)
    boxes = as_boxes(boxes)

    # Convertir les scores en tableau NumPy 1D de shape (N,)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    # Vérifier que le critère de similarité est valide
    if metric not in ("iou", "ios"):
        raise ValueError("metric must be 'iou' or 'ios'")

    # Nombre de bounding boxes
    n = len(scores)

    # Cas particulier : aucune détection
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    # Calculer toutes les similarités entre les bounding boxes
    # Shape : (N, N)
    similarity = (
        box_iou(boxes, boxes)
        if metric == "iou"
        else box_ios(boxes, boxes)
    )

    # Trier les indices par score décroissant
    # Mergesort garantit un ordre stable en cas d'égalité
    order = np.argsort(-scores, kind="mergesort")

    # Tableau booléen indiquant les boîtes déjà supprimées
    # Shape : (N,)
    suppressed = np.zeros(n, dtype=bool)

    # Liste des indices des boîtes conservées
    keep = []

    # Parcourir les boîtes dans l'ordre décroissant des scores
    for i in order:

        # Si la boîte a déjà été supprimée, passer à la suivante
        if suppressed[i]:
            continue

        # Sinon, conserver son indice
        keep.append(i)

        # Supprimer les boîtes dont la similarité avec la boîte
        # actuellement conservée est strictement supérieure au seuil
        # similarity[i] : shape (N,)
        suppressed |= similarity[i] > threshold

    # Retourner les indices conservés dans l'ordre des scores décroissants
    return np.asarray(keep, dtype=np.int64)


def batched_nms(dets: Detections, threshold: float, metric: str = "iou") -> Detections:
    """Class-wise NMS: boxes of different classes never suppress each other."""
    keep = []
    for c in np.unique(dets.classes):
        idx = np.flatnonzero(dets.classes == c)
        keep.append(idx[nms(dets.boxes[idx], dets.scores[idx], threshold, metric)])
    if not keep:
        return Detections.empty()
    keep = np.concatenate(keep)
    keep = keep[np.argsort(-dets.scores[keep], kind="mergesort")]
    return dets.select(keep)


# --------------------------------------------------------------------------- tiled inference
def merge_tile_detections(
    per_tile: Sequence[Detections], tiles, frame_hw, threshold: float = 0.6, metric: str = "ios",
    border_px: float = 2.0,
) -> Detections:
    """Merge the detections of overlapping tiles into frame coordinates.

    per_tile[k] holds boxes in the coordinates of tile k, whose xyxy position in the
    frame is tiles[k] (a tile may be the whole frame). Three steps:
    1. in each tile, drop the boxes closer than border_px to an INNER side of the tile
       (a side that is not a side of the frame): they are truncated, and a complete view
       exists in a neighbouring tile or in the full frame;
    2. translate the remaining boxes to the frame;
    3. class-wise NMS (batched_nms) with the given metric.
    """
    
    # Convert tiles to an array of shape (K, 4)
    tiles = as_boxes(tiles)

    # We need exactly one Detections object per tile
    if len(per_tile) != len(tiles):
        raise ValueError("one Detections per tile is required")

    # Original frame dimensions
    fh, fw = frame_hw

    # Will contain detections expressed in full-frame coordinates
    moved = []

    # Process each tile and its corresponding detections
    for dets, (x0, y0, x1, y1) in zip(per_tile, tiles):

        # Local bounding boxes of the current tile: shape (N, 4)
        b = dets.boxes

        # Detect boxes truncated by an internal tile boundary
        truncated = (
            ((x0 > 0) & (b[:, 0] <= border_px))
            | ((y0 > 0) & (b[:, 1] <= border_px))
            | ((x1 < fw) & (b[:, 2] >= (x1 - x0) - border_px))
            | ((y1 < fh) & (b[:, 3] >= (y1 - y0) - border_px))
        )

        # Keep only non-truncated boxes and translate them
        # from local tile coordinates to full-frame coordinates
        moved.append(
            dets.select(~truncated).transformed(
                BoxTransform(1.0, 1.0, x0, y0)
            )
        )

    # Concatenate detections from all tiles
    # Then remove duplicate boxes using class-wise NMS
    return batched_nms(
        Detections.concat(moved),
        threshold,
        metric,
    )


# --------------------------------------------------------------------------- detectors
BatchDetector = Callable[[Sequence[np.ndarray]], list]
"""A detector maps a list of BGR uint8 images to a list of Detections (one per image)."""


class YoloDetector:
    """Ultralytics YOLO (COCO pre-trained) behind the pitchtrack contract.

    Ultralytics letterboxes internally and returns boxes in the coordinates of the image
    it was given: no rescaling is needed here.
    Licence: Ultralytics is AGPL-3.0; keep it behind this class so the rest of the code
    does not depend on it.
    """

    def __init__(self, weights="yolo26m.pt", imgsz=1280, conf=0.01, classes=(COCO_PERSON,),
                 device=None, half=True, max_det=300):
        from ultralytics import YOLO  # optional dependency, imported lazily

        self.model = YOLO(weights)
        self.imgsz, self.conf, self.classes = imgsz, conf, list(classes)
        self.device, self.max_det = device, max_det
        self.half = bool(half) and device is not None and str(device).startswith("cuda")
        self.precision = self.precision_kwargs(self.half)

    @staticmethod
    def precision_kwargs(half: bool) -> dict:
        """FP16 inference: `quantize=16` in recent Ultralytics releases, `half=True` in older ones."""
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            if "quantize" in DEFAULT_CFG_DICT:
                return {"quantize": 16} if half else {}
        except ImportError:
            pass
        return {"half": True} if half else {}

    def __call__(self, images: Sequence[np.ndarray]) -> list:
        results = self.model.predict(
            list(images), imgsz=self.imgsz, conf=self.conf, classes=self.classes,
            device=self.device, max_det=self.max_det, verbose=False, **self.precision,
        )
        out = []
        for r in results:
            b = r.boxes
            out.append(Detections(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()))
        return out


class TiledDetector:
    """Run a base detector on overlapping tiles of each frame and merge (SAHI-like).

    With 1080p frames, tiling only pays off if the tiles are UPSAMPLED by the base
    detector (tile smaller than its input size). Keep
    full_frame=True unless the overlap is larger than the tallest player: boxes cut by
    an inner tile side are dropped, and the full frame is then their only complete view.
    """

    def __init__(self, base: BatchDetector, tile: int = 640, overlap: float = 0.2,
                 threshold: float = 0.6, metric: str = "ios", full_frame: bool = True):
        self.base, self.tile, self.overlap = base, tile, overlap
        self.threshold, self.metric, self.full_frame = threshold, metric, full_frame

    def __call__(self, images: Sequence[np.ndarray]) -> list:
        out = []
        for img in images:
            hw = img.shape[:2]
            tiles = tile_grid(hw, self.tile, self.overlap)
            crops = [img[int(y0):int(y1), int(x0):int(x1)] for x0, y0, x1, y1 in tiles]
            if self.full_frame:  # the full frame keeps the large players intact
                crops.append(img)
                tiles = np.vstack([tiles, [0, 0, hw[1], hw[0]]])
            per_tile = list(self.base(crops))
            out.append(merge_tile_detections(per_tile, tiles, hw, self.threshold, self.metric))
        return out


def build_detector(cfg: dict) -> BatchDetector:
    """Detector from the `detector` and `tiling` sections of a config (configs/*.yaml)."""
    d = cfg["detector"]
    base = YoloDetector(d["weights"], d["imgsz"], d["conf"], d.get("classes", [COCO_PERSON]),
                        d.get("device"), d.get("half", True), d.get("max_det", 300))
    t = cfg.get("tiling", {})
    if t.get("enabled", False):
        return TiledDetector(base, t["tile"], t["overlap"], t.get("merge_threshold", 0.6),
                             t.get("merge_metric", "ios"), t.get("full_frame", True))
    return base
