"""Detection metrics written from scratch: matching, AP, operating point, recall by size.

Pipeline: match detections to ground truth in each frame, pool the
(score, is_tp) pairs of all frames, then read AP or a single operating point
from them. Greedy matching by decreasing score has a useful property: the
matches of the detections above a threshold do not depend on the detections below it,
so one matching with a low score threshold gives the whole curve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .boxes import as_boxes, box_iou

COCO_IOU_THRESHOLDS = np.round(np.arange(0.5, 0.951, 0.05), 2)


# --------------------------------------------------------------------------- matching
def match_greedy(gt, det, scores, iou_threshold: float):
    """COCO-style greedy matching in ONE frame.

    Visit the detections by decreasing score (stable sort); each one takes the
    still-unmatched ground-truth box with the highest IoU, provided that
    IoU >= iou_threshold. Returns (det_gt, gt_det): det_gt[j] is the index of the
    ground truth matched by detection j (or -1), gt_det[i] the index of the detection
    matched to ground truth i (or -1).
    """

    # Normaliser les bounding boxes au format NumPy (N, 4) et (M, 4)
    gt, det = as_boxes(gt), as_boxes(det)

    # Convertir les scores en tableau NumPy 1D
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    # Nombre de Ground Truth et de détections
    n_gt, n_det = len(gt), len(det)

    # Initialiser les associations avec -1 :
    # det_gt[j] = indice du Ground Truth associé à la détection j
    # gt_det[i] = indice de la détection associée au Ground Truth i
    # -1 signifie qu'aucune association n'existe
    det_gt = np.full(n_det, -1, dtype=np.int64)
    gt_det = np.full(n_gt, -1, dtype=np.int64)

    # Calculer les IoU entre toutes les détections et tous les Ground Truth
    # Shape : (n_det, n_gt)
    iou = box_iou(det, gt)

    # Trier les détections par score décroissant
    # Le mergesort garantit un tri stable en cas d'égalité des scores
    order = np.argsort(-scores, kind="mergesort")

    # Parcourir les détections dans l'ordre décroissant de confiance
    for j in order:

        # Récupérer les IoU de la détection j avec tous les Ground Truth
        # Shape : (n_gt,)
        candidates = iou[j].copy()

        # Éliminer les Ground Truth déjà associés à une détection
        # On remplace leur IoU par -1 pour qu'ils ne puissent plus être sélectionnés
        candidates[gt_det != -1] = -1.0

        # Vérifier qu'il reste au moins un Ground Truth disponible
        if np.all(gt_det != -1):
            continue

        # Trouver l'indice du Ground Truth disponible ayant la meilleure IoU
        best_gt = int(np.argmax(candidates))

        # Vérifier que la meilleure IoU atteint le seuil demandé
        if candidates[best_gt] >= iou_threshold:

            # Associer la détection j au Ground Truth best_gt
            det_gt[j] = best_gt

            # Enregistrer également l'association dans l'autre sens
            gt_det[best_gt] = j

    # Retourner les associations dans les deux directions
    return det_gt, gt_det


# --------------------------------------------------------------------------- precision, recall, AP
def precision_recall(scores, is_tp, n_gt: int):
    """Precision and recall after each detection, detections sorted by decreasing score."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    tp = np.asarray(is_tp, dtype=np.float64).reshape(-1)
    order = np.argsort(-scores, kind="mergesort")
    ctp, cfp = np.cumsum(tp[order]), np.cumsum(1.0 - tp[order])
    recall = ctp / n_gt if n_gt > 0 else np.zeros_like(ctp)
    precision = ctp / np.maximum(ctp + cfp, np.finfo(np.float64).eps)
    return precision, recall


def average_precision(scores, is_tp, n_gt: int, n_points: int = 101) -> float:
    """COCO AP: mean over r in {0, 0.01, ..., 1} of the interpolated precision
    p_interp(r) = max{precision(k) : recall(k) >= r}, and 0 if recall r is never reached.

    scores and is_tp are pooled over all frames; n_gt is the total number of ground
    truth boxes. Returns NaN if n_gt == 0.
    """
    if n_gt == 0:
        return float("nan")
    precision, recall = precision_recall(scores, is_tp, n_gt)

    # Calculer Precision et Recall à partir des scores et des TP/FP
    precision, recall = precision_recall(scores, is_tp, n_gt)

    # Aucune détection : AP égale à zéro
    if len(precision) == 0:
        return 0.0

    # Construire l'enveloppe de précision :
    # chaque position contient le maximum des précisions situées à sa droite
    envelope = np.maximum.accumulate(precision[::-1])[::-1]

    # Générer les n_points niveaux de rappel entre 0 et 1
    levels = np.linspace(0.0, 1.0, n_points)

    # Trouver le premier indice k tel que recall[k] >= chaque niveau demandé
    k = np.searchsorted(recall, levels, side="left")

    # Identifier les niveaux de rappel effectivement atteints
    reached = k < len(envelope)

    # Initialiser les précisions interpolées à zéro
    p = np.zeros(n_points)

    # Remplir les précisions correspondant aux niveaux atteints
    p[reached] = envelope[k[reached]]

    # Calculer la moyenne des précisions interpolées
    return float(p.mean())


# --------------------------------------------------------------------------- operating point
def operating_curve(scores, is_tp, n_gt: int, thresholds) -> pd.DataFrame:
    """Precision, recall and DetA = TP / (TP + FN + FP) when only the detections with
    score >= tau are kept, for each tau in thresholds (one row per tau).

    Valid because greedy matching is prefix-consistent (module docstring).
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    is_tp = np.asarray(is_tp, dtype=bool).reshape(-1)
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    # Convert inputs into NumPy arrays with the appropriate types
    # Build a boolean matrix indicating which detections are kept
    # scores[None, :]:       shape (1, N)
    # thresholds[:, None]:  shape (T, 1)
    # kept:                  shape (T, N)
    kept = scores[None, :] >= thresholds[:, None]

    # Count True Positives for each threshold: shape (T,)
    tp = (kept & is_tp[None, :]).sum(axis=1).astype(np.float64)

    # Count False Positives for each threshold: shape (T,)
    fp = (kept & ~is_tp[None, :]).sum(axis=1).astype(np.float64)

    # False Negatives = total Ground Truth - True Positives
    fn = n_gt - tp

    # Compute Precision, Recall and DetA
    with np.errstate(invalid="ignore", divide="ignore"):

        # Precision = TP / (TP + FP)
        # NaN when no detection is kept
        precision = np.where(
            tp + fp > 0,
            tp / (tp + fp),
            np.nan
        )

        # Recall = TP / total Ground Truth
        # NaN when there is no Ground Truth
        recall = (
            tp / n_gt
            if n_gt > 0
            else np.full_like(tp, np.nan)
        )

        # DetA = TP / (TP + FN + FP)
        det_a = np.where(
            tp + fn + fp > 0,
            tp / (tp + fn + fp),
            np.nan
        )

    # Return one row per confidence threshold
    return pd.DataFrame({
        "threshold": thresholds,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "det_a": det_a
    })


def recall_by_height(gt_heights, gt_best_score, bins, threshold: float) -> pd.DataFrame:
    """Recall of the ground truth boxes per height bin, keeping detections with score >= threshold.

    gt_best_score[i] is the score of the detection matched to ground truth i (-inf if none).
    """
    h = np.asarray(gt_heights, dtype=np.float64)
    found = np.asarray(gt_best_score, dtype=np.float64) >= threshold
    which = np.digitize(h, bins[1:-1], right=False)
    rows = []
    for b in range(len(bins) - 1):
        sel = which == b
        rows.append({"height_bin": f"[{bins[b]:g}, {bins[b + 1]:g})", "n_gt": int(sel.sum()),
                     "recall": float(found[sel].mean()) if sel.any() else float("nan")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- pooling
@dataclass
class PooledMatches:
    """Detections pooled over frames at one IoU threshold."""

    scores: np.ndarray         # (D,)
    is_tp: np.ndarray          # (D,) bool
    det_clip: np.ndarray       # (D,) clip index of each detection
    gt_heights: np.ndarray     # (G,)
    gt_best_score: np.ndarray  # (G,) score of the matched detection, -inf if missed
    gt_clip: np.ndarray        # (G,) clip index of each ground truth box
    n_gt: int


def pool_matches(frames, iou_threshold: float) -> PooledMatches:
    """Match every frame and pool the results.

    frames: iterable of (clip_index, gt_boxes (G, 4), det_boxes (D, 4), det_scores (D,)).
    """
    scores, tps, dclip, heights, best, gclip = [], [], [], [], [], []
    for clip, gt, det, sc in frames:
        gt, det = as_boxes(gt), as_boxes(det)
        sc = np.asarray(sc, dtype=np.float64).reshape(-1)
        det_gt, gt_det = match_greedy(gt, det, sc, iou_threshold)
        scores.append(sc)
        tps.append(det_gt >= 0)
        dclip.append(np.full(len(sc), clip))
        heights.append(gt[:, 3] - gt[:, 1])
        best.append(np.where(gt_det >= 0, sc[np.maximum(gt_det, 0)] if len(sc) else -np.inf, -np.inf))
        gclip.append(np.full(len(gt), clip))
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)  # noqa: E731
    gt_heights = cat(heights, np.float64)
    return PooledMatches(cat(scores, np.float64), cat(tps, bool), cat(dclip, np.int64),
                         gt_heights, cat(best, np.float64), cat(gclip, np.int64), len(gt_heights))


def coco_ap(frames, iou_thresholds=COCO_IOU_THRESHOLDS) -> dict:
    """AP at each IoU threshold and their mean (AP@[.5:.95]). `frames` must be re-iterable."""
    frames = list(frames)
    aps = {}
    for t in iou_thresholds:
        m = pool_matches(frames, float(t))
        aps[f"AP{int(round(100 * t))}"] = average_precision(m.scores, m.is_tp, m.n_gt)
    aps["AP"] = float(np.mean([aps[f"AP{int(round(100 * t))}"] for t in iou_thresholds]))
    return aps


def cluster_bootstrap_ratio(num, den, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile CI of sum(num) / sum(den) when whole clips are resampled with replacement.

    num[c], den[c]: per-clip numerator and denominator (e.g. TP and TP + FN for recall).
    Clips, not frames, are the independent units: consecutive frames are strongly correlated.
    """
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    if den.sum() <= 0 or np.all(np.isnan(stats)):
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(num.sum() / den.sum()), float(lo), float(hi)
