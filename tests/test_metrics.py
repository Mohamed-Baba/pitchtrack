import numpy as np
import pytest

from pitchtrack.metrics import (average_precision, cluster_bootstrap_ratio, coco_ap, match_greedy,
                                operating_curve, pool_matches, precision_recall, recall_by_height)

RNG = np.random.default_rng(1)


# ----------------------------------------------------------------- matching
def test_match_simple_and_duplicate():
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float)
    det = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [21, 0, 31, 10]], float)
    det_gt, gt_det = match_greedy(gt, det, [0.9, 0.95, 0.5], 0.5)
    assert det_gt.tolist() == [-1, 0, 1]        # the 0.95 box takes gt 0, the 0.9 copy is a FP
    assert gt_det.tolist() == [1, 2]


def test_match_threshold_inclusive_and_empty():
    gt = np.array([[0, 0, 10, 10]], float)
    det = np.array([[0, 0, 10, 5]], float)       # IoU exactly 0.5
    assert match_greedy(gt, det, [1.0], 0.5)[0].tolist() == [0]
    assert match_greedy(gt, det, [1.0], 0.51)[0].tolist() == [-1]
    d, g = match_greedy(np.zeros((0, 4)), det, [1.0], 0.5)
    assert d.tolist() == [-1] and g.shape == (0,)
    d, g = match_greedy(gt, np.zeros((0, 4)), [], 0.5)
    assert d.shape == (0,) and g.tolist() == [-1]


def test_greedy_is_not_optimal():
    """Greedy by score can lose a match that an optimal assignment would keep."""
    gt = np.array([[0, 0, 10, 10], [6, 0, 16, 10]], float)
    det = np.array([[4, 0, 14, 10], [8, 0, 18, 10]], float)
    det_gt, _ = match_greedy(gt, det, [0.9, 0.8], 0.4)
    assert (det_gt >= 0).sum() == 1                # greedy: one TP
    from scipy.optimize import linear_sum_assignment
    from pitchtrack.boxes import box_iou
    iou = box_iou(det, gt)
    r, c = linear_sum_assignment(-iou)
    assert (iou[r, c] >= 0.4).sum() == 2           # optimal: two TP


# ----------------------------------------------------------------- AP
def test_ap_perfect_and_hopeless():
    assert average_precision([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)
    assert average_precision([0.9, 0.8], [False, False], 2) == 0.0
    assert average_precision([], [], 3) == 0.0
    assert np.isnan(average_precision([0.5], [False], 0))


def test_ap_hand_computed():
    # ranked: TP, FP, TP with 4 ground truth boxes -> precision 1, 1/2, 2/3; recall 1/4, 1/4, 1/2
    # envelope: 1, 2/3, 2/3; levels r <= 0.25 -> 1 (26 levels), 0.25 < r <= 0.5 -> 2/3 (25 levels)
    ap = average_precision([0.9, 0.8, 0.7], [True, False, True], 4)
    assert ap == pytest.approx((26 * 1.0 + 25 * 2 / 3) / 101)


def test_ap_ignores_order_of_input():
    s = RNG.uniform(size=50)
    tp = RNG.uniform(size=50) < 0.6
    p = RNG.permutation(50)
    assert average_precision(s, tp, 40) == pytest.approx(average_precision(s[p], tp[p], 40))


def test_precision_recall_shapes():
    p, r = precision_recall([0.2, 0.9], [True, False], 1)
    np.testing.assert_allclose(p, [0.0, 0.5])
    np.testing.assert_allclose(r, [0.0, 1.0])


# ----------------------------------------------------------------- operating curve
def test_operating_curve_and_deta_identity():
    s = RNG.uniform(size=200)
    tp = RNG.uniform(size=200) < s                     # higher scores are more often right
    curve = operating_curve(s, tp, 150, [0.0, 0.3, 0.6, 2.0])
    row = curve.iloc[1]
    assert row.tp == (tp & (s >= 0.3)).sum() and row.fp == (~tp & (s >= 0.3)).sum()
    assert row.fn == 150 - row.tp
    p, r = row.precision, row.recall
    assert row.det_a == pytest.approx(1.0 / (1.0 / p + 1.0 / r - 1.0))     # DetA = F1 / (2 - F1)
    assert row.det_a <= min(p, r)
    assert np.isnan(curve.iloc[3].precision) and curve.iloc[3].recall == 0.0


def test_prefix_consistency_of_greedy_matching():
    """Matching once at a low threshold == matching again after filtering."""
    gt = RNG.uniform(0, 100, (30, 2))
    gt = np.hstack([gt, gt + RNG.uniform(5, 15, (30, 2))])
    det = gt[RNG.integers(0, 30, 60)] + RNG.normal(0, 3, (60, 4))
    sc = RNG.uniform(size=60)
    det_gt, _ = match_greedy(gt, det, sc, 0.5)
    keep = sc >= 0.5
    det_gt2, _ = match_greedy(gt, det[keep], sc[keep], 0.5)
    np.testing.assert_array_equal(det_gt[keep], det_gt2)


def test_recall_by_height():
    df = recall_by_height([10, 50, 55, 200], [0.9, -np.inf, 0.4, 0.8], [0, 40, 100, 1e4], 0.5)
    assert df["n_gt"].tolist() == [1, 2, 1]
    assert df["recall"].tolist() == [1.0, 0.0, 1.0]


# ----------------------------------------------------------------- pooling, bootstrap
def make_frames(n_frames=30, seed=2):
    rng = np.random.default_rng(seed)
    frames = []
    for f in range(n_frames):
        n = rng.integers(0, 8)
        xy = rng.uniform(0, 400, (n, 2))
        gt = np.hstack([xy, xy + rng.uniform(8, 60, (n, 2))])
        keep = rng.uniform(size=n) < 0.8
        det = gt[keep] + rng.normal(0, 2.0, (keep.sum(), 4))
        fp = rng.uniform(0, 400, (rng.integers(0, 4), 2))
        det = np.vstack([det, np.hstack([fp, fp + 20])])
        frames.append((f % 3, gt, det, rng.uniform(size=len(det))))
    return frames


def test_pool_matches_counts():
    frames = make_frames()
    m = pool_matches(frames, 0.5)
    assert m.n_gt == sum(len(g) for _, g, _, _ in frames)
    assert len(m.scores) == sum(len(d) for _, _, d, _ in frames)
    assert m.is_tp.sum() == np.isfinite(m.gt_best_score).sum()


def test_cluster_bootstrap_ratio():
    est, lo, hi = cluster_bootstrap_ratio([5, 7, 9], [10, 10, 10], n_boot=500, seed=0)
    assert est == pytest.approx(0.7) and lo <= est <= hi


def test_ap_matches_pycocotools():
    pycocotools = pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    frames = make_frames(40, seed=5)
    images, anns, dets = [], [], []
    for k, (_, gt, det, sc) in enumerate(frames):
        images.append({"id": k, "width": 500, "height": 500})
        for g in gt:
            anns.append({"id": len(anns) + 1, "image_id": k, "category_id": 1, "iscrowd": 0,
                         "bbox": [g[0], g[1], g[2] - g[0], g[3] - g[1]],
                         "area": (g[2] - g[0]) * (g[3] - g[1])})
        for d, s in zip(det, sc):
            dets.append({"image_id": k, "category_id": 1, "score": float(s),
                         "bbox": [d[0], d[1], d[2] - d[0], d[3] - d[1]]})
    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}
    coco.createIndex()
    ev = COCOeval(coco, coco.loadRes(dets), "bbox")
    ev.params.maxDets = [1000]
    ev.params.areaRng, ev.params.areaRngLbl = [[0, 1e10]], ["all"]
    ev.evaluate()
    ev.accumulate()
    precision = ev.eval["precision"][:, :, 0, 0, 0]          # (iou thresholds, recall levels)
    ours = coco_ap(frames)
    assert ours["AP50"] == pytest.approx(precision[0].mean(), abs=1e-9)
    assert ours["AP"] == pytest.approx(precision.mean(), abs=1e-9)
