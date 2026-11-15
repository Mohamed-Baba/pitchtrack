"""Tracking metrics: HOTA written from scratch.

The sequence format is the one of TrackEval, so that every number can be checked against
the reference implementation (pip install trackeval): a dict with, per frame t,
gt_ids[t] (G_t,), tracker_ids[t] (P_t,) (ids relabelled 0..n-1 within the sequence) and
similarity_scores[t] (G_t, P_t) (IoU), plus num_gt_ids, num_tracker_ids, num_gt_dets,
num_tracker_dets. HOTA is computed for 19 localisation thresholds alpha in
{0.05, ..., 0.95} and averaged over alpha.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .boxes import box_iou

ALPHAS = np.arange(0.05, 0.99, 0.05)   # exactly TrackEval's thresholds
EPS = np.finfo(float).eps


def sequence_data(gt: pd.DataFrame, pred: pd.DataFrame, frames, name: str = "") -> dict:
    """TrackEval-style data of one clip from two tables (frame, track_id, x1, y1, x2, y2)."""
    frames = np.asarray(sorted(frames))
    gt = gt[gt["frame"].isin(frames)]
    pred = pred[pred["frame"].isin(frames)]
    gt_map = {v: k for k, v in enumerate(np.unique(gt["track_id"]))}
    pr_map = {v: k for k, v in enumerate(np.unique(pred["track_id"]))}
    gtg, prg = dict(tuple(gt.groupby("frame"))), dict(tuple(pred.groupby("frame")))
    cols = ["x1", "y1", "x2", "y2"]
    data = {"gt_ids": [], "tracker_ids": [], "similarity_scores": [], "seq": name,
            "num_gt_ids": len(gt_map), "num_tracker_ids": len(pr_map),
            "num_gt_dets": len(gt), "num_tracker_dets": len(pred), "num_timesteps": len(frames)}
    for f in frames:
        g, p = gtg.get(f), prg.get(f)
        g_ids = np.array([gt_map[i] for i in g["track_id"]], dtype=int) if g is not None else np.zeros(0, int)
        p_ids = np.array([pr_map[i] for i in p["track_id"]], dtype=int) if p is not None else np.zeros(0, int)
        g_box = g[cols].to_numpy() if g is not None else np.zeros((0, 4))
        p_box = p[cols].to_numpy() if p is not None else np.zeros((0, 4))
        data["gt_ids"].append(g_ids)
        data["tracker_ids"].append(p_ids)
        data["similarity_scores"].append(box_iou(g_box, p_box))
    return data


def _final_fields(res: dict) -> dict:
    tp, fn, fp = res["HOTA_TP"], res["HOTA_FN"], res["HOTA_FP"]
    res["DetRe"] = tp / np.maximum(1, tp + fn)
    res["DetPr"] = tp / np.maximum(1, tp + fp)
    res["DetA"] = tp / np.maximum(1, tp + fn + fp)
    res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
    return res


# --------------------------------------------------------------------------- HOTA
def hota(data: dict, alphas=ALPHAS) -> dict:
    """HOTA of one sequence, as arrays over alphas: HOTA, DetA, AssA, DetRe, DetPr, AssRe,
    AssPr, LocA and the counts HOTA_TP, HOTA_FN, HOTA_FP.

    1. Global alignment between ids: for every frame, sim_iou = S / (row sums + column sums - S)
       (0 where the denominator is 0) accumulated in potential[gt_id, tr_id]; with the
       numbers of detections n_gt[gt_id] and n_tr[tr_id]:
       align = potential / (n_gt + n_tr - potential).
    2. For every frame: Hungarian matching maximising align[gt_ids, tr_ids] * S; for each
       alpha, the matched pairs with S >= alpha - EPS are true positives (count them, add S
       to the LocA sum, add 1 to matches[alpha][gt_id, tr_id]); the others are FN and FP.
       A frame without ground truth only adds FP; a frame without predictions only FN.
    3. For each alpha: A(gt_id, tr_id) = m / max(1, n_gt + n_tr - m) with m the match count;
       AssA = sum(m * A) / max(1, TP); AssRe and AssPr with n_gt and n_tr as denominators;
       LocA = max(1e-10, LocA sum) / max(1e-10, TP). Then _final_fields.
    Empty tracker (or empty ground truth): every detection is a FN (FP), LocA = 1, the other
    association fields 0.
    """
    k = len(alphas)
    res = {f: np.zeros(k) for f in ("HOTA_TP", "HOTA_FN", "HOTA_FP", "AssA", "AssRe", "AssPr", "LocA")}
    if data["num_tracker_dets"] == 0 or data["num_gt_dets"] == 0:
        res["HOTA_FN"] += data["num_gt_dets"]
        res["HOTA_FP"] += data["num_tracker_dets"]
        res["LocA"] = np.ones(k)
        return _final_fields(res)

    
    # ------------------------------------------------------------
    # Step 1: Global alignment between ground-truth and tracker IDs
    # ------------------------------------------------------------

    n_gt_ids = data["num_gt_ids"]
    n_tr_ids = data["num_tracker_ids"]

    # Total number of detections for each identity
    n_gt = np.zeros(n_gt_ids)
    n_tr = np.zeros(n_tr_ids)

    # Accumulated similarity between every pair of identities
    potential = np.zeros((n_gt_ids, n_tr_ids))

    # Process every frame
    for gt_ids, tr_ids, S in zip(
        data["gt_ids"],
        data["tracker_ids"],
        data["similarity_scores"]
    ):
        # Count the detections of each identity
        n_gt[gt_ids] += 1
        n_tr[tr_ids] += 1

        # Skip frames where one side is empty
        if len(gt_ids) == 0 or len(tr_ids) == 0:
            continue

        # Compute the sum of each row and each column
        row_sum = S.sum(axis=1, keepdims=True)
        col_sum = S.sum(axis=0, keepdims=True)

        # Compute the denominator of the IoU-like similarity
        denominator = row_sum + col_sum - S

        # Compute sim_iou, with 0 when denominator is 0
        sim_iou = np.divide(
            S,
            denominator,
            out=np.zeros_like(S),
            where=denominator > 0
        )

        # Accumulate similarities using global identity indices
        potential[np.ix_(gt_ids, tr_ids)] += sim_iou

    # Reshape identity detection counts for broadcasting
    n_gt = n_gt[:, None]
    n_tr = n_tr[None, :]

    # Compute global alignment between identities
    align_denominator = n_gt + n_tr - potential

    align = np.divide(
        potential,
        align_denominator,
        out=np.zeros_like(potential),
        where=align_denominator > 0
    )

    # ------------------------------------------------------------
    # Step 2: Per-frame matching, then counts for every alpha
    # ------------------------------------------------------------

    matches = [np.zeros_like(potential) for _ in range(k)]

    for gt_ids, tr_ids, S in zip(
        data["gt_ids"],
        data["tracker_ids"],
        data["similarity_scores"]
    ):
        # A frame without ground truth only adds false positives,
        # a frame without predictions only adds false negatives
        if len(gt_ids) == 0:
            res["HOTA_FP"] += len(tr_ids)
            continue
        if len(tr_ids) == 0:
            res["HOTA_FN"] += len(gt_ids)
            continue

        # Hungarian matching that maximises alignment x similarity
        score = align[np.ix_(gt_ids, tr_ids)] * S
        rows, cols = linear_sum_assignment(-score)

        for a, alpha in enumerate(alphas):
            # A matched pair counts only if its similarity reaches alpha
            ok = S[rows, cols] >= alpha - EPS
            r, c = rows[ok], cols[ok]
            n = len(r)

            res["HOTA_TP"][a] += n
            res["HOTA_FN"][a] += len(gt_ids) - n
            res["HOTA_FP"][a] += len(tr_ids) - n

            if n > 0:
                res["LocA"][a] += S[r, c].sum()
                matches[a][gt_ids[r], tr_ids[c]] += 1

    # ------------------------------------------------------------
    # Step 3: Association scores for every alpha
    # ------------------------------------------------------------

    for a in range(k):
        m = matches[a]
        tp = max(1.0, res["HOTA_TP"][a])

        res["AssA"][a] = np.sum(m * m / np.maximum(1, n_gt + n_tr - m)) / tp
        res["AssRe"][a] = np.sum(m * m / np.maximum(1, n_gt)) / tp
        res["AssPr"][a] = np.sum(m * m / np.maximum(1, n_tr)) / tp

    res["LocA"] = np.maximum(1e-10, res["LocA"]) / np.maximum(1e-10, res["HOTA_TP"])

    return _final_fields(res)



def combine_hota(results: list[dict]) -> dict:
    """Combine sequences like TrackEval: sum the counts, weight AssA, AssRe, AssPr, LocA by TP."""
    res = {f: sum(r[f] for r in results) for f in ("HOTA_TP", "HOTA_FN", "HOTA_FP")}
    tp = np.maximum(1, res["HOTA_TP"])
    for f in ("AssA", "AssRe", "AssPr"):
        res[f] = sum(r[f] * r["HOTA_TP"] for r in results) / tp
    loca = sum(r["LocA"] * r["HOTA_TP"] for r in results)
    res["LocA"] = np.maximum(1e-10, loca) / np.maximum(1e-10, res["HOTA_TP"])
    return _final_fields(res)


def summarize(res: dict) -> dict:
    """Scalar summary: mean over alpha of HOTA, DetA, AssA, LocA (TrackEval's convention)."""
    return {k: float(np.mean(res[k])) for k in ("HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr")}


def clear_identity(data: dict) -> dict:
    """MOTA, IDSW, Frag (CLEAR MOT) and IDF1 (Identity) from TrackEval, if it is installed."""
    try:
        from trackeval.metrics import CLEAR, Identity
    except ImportError:
        return {}
    clear = CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(data)
    ident = Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(data)
    out = {k: clear[k] for k in ("MOTA", "IDSW", "Frag", "CLR_TP", "CLR_FN", "CLR_FP")}
    out.update({k: ident[k] for k in ("IDF1", "IDTP", "IDFN", "IDFP")})
    return out


def combine_clear_identity(results: list[dict]) -> dict:
    """Pool CLEAR and Identity counts over sequences."""
    if not results or not results[0]:
        return {}
    s = {k: sum(r[k] for r in results) for k in ("IDSW", "Frag", "CLR_TP", "CLR_FN", "CLR_FP",
                                                    "IDTP", "IDFN", "IDFP")}
    n_gt = s["CLR_TP"] + s["CLR_FN"]
    s["MOTA"] = (s["CLR_TP"] - s["CLR_FP"] - s["IDSW"]) / max(1, n_gt)
    s["IDF1"] = s["IDTP"] / max(1.0, s["IDTP"] + 0.5 * s["IDFP"] + 0.5 * s["IDFN"])
    return s
