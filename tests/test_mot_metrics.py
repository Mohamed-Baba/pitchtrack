import numpy as np
import pandas as pd
import pytest

from pitchtrack.mot_metrics import combine_hota, hota, sequence_data, summarize

RNG = np.random.default_rng(5)


def table(rows):
    return pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2"])


def one_track(n, ids):
    """GT track 1 over n frames, prediction ids[f] (None = missed) on exactly the same boxes."""
    gt = table([(f, 1, 0, 0, 10, 10) for f in range(n)])
    pr = table([(f, ids[f], 0, 0, 10, 10) for f in range(n) if ids[f] is not None])
    return sequence_data(gt, pr, range(n))


# ----------------------------------------------------------------- HOTA
def test_perfect_tracker():
    s = summarize(hota(one_track(20, [7] * 20)))
    assert s["HOTA"] == pytest.approx(1.0) and s["AssA"] == pytest.approx(1.0)


def test_id_switch_halfway():          # one identity switch halfway
    s = summarize(hota(one_track(100, [7] * 50 + [8] * 50)))
    assert s["DetA"] == pytest.approx(1.0) and s["AssA"] == pytest.approx(0.5)
    assert s["HOTA"] == pytest.approx(np.sqrt(0.5))


def test_half_missed():                # half of the frames missed
    s = summarize(hota(one_track(100, [7] * 50 + [None] * 50)))
    assert s["DetA"] == pytest.approx(0.5) and s["AssA"] == pytest.approx(0.5)
    assert s["HOTA"] == pytest.approx(0.5)


def test_empty_tracker():
    res = hota(one_track(10, [None] * 10))
    assert np.all(res["HOTA"] == 0) and np.all(res["HOTA_FN"] == 10)


def random_sequence(n_frames=40, n_tracks=6, seed=0):
    rng = np.random.default_rng(seed)
    start = rng.uniform(0, 300, (n_tracks, 2))
    vel = rng.normal(0, 3, (n_tracks, 2))
    gt, pr = [], []
    for f in range(n_frames):
        for k in range(n_tracks):
            if rng.uniform() < 0.1:
                continue
            x, y = start[k] + f * vel[k]
            gt.append((f, k, x, y, x + 20, y + 50))
            if rng.uniform() < 0.85:
                pid = k + 100 * (f > n_frames // 2 and k % 3 == 0)      # some id switches
                j = rng.normal(0, 4, 4)
                pr.append((f, pid, x + j[0], y + j[1], x + 20 + j[2], y + 50 + j[3]))
        for _ in range(rng.poisson(0.5)):                                # false positives
            x, y = rng.uniform(0, 300, 2)
            pr.append((f, 1000 + f, x, y, x + 20, y + 50))
    return sequence_data(table(gt), table(pr), range(n_frames), f"seq{seed}")


def test_matches_trackeval():
    trackeval = pytest.importorskip("trackeval")
    metric = trackeval.metrics.HOTA()
    ours, refs = [], []
    for seed in range(4):
        data = random_sequence(seed=seed)
        mine, ref = hota(data), metric.eval_sequence(data)
        for key in ("HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "AssRe", "AssPr"):
            np.testing.assert_allclose(mine[key], ref[key], atol=1e-12, err_msg=key)
        ours.append(mine)
        refs.append(ref)
    combined = combine_hota(ours)
    ref_c = metric.combine_sequences({f"s{i}": r for i, r in enumerate(refs)})
    for key in ("HOTA", "DetA", "AssA", "LocA"):
        np.testing.assert_allclose(combined[key], ref_c[key], atol=1e-12, err_msg=key)


# ----------------------------------------------------------------- data format
def test_sequence_data_relabels_ids():
    gt = table([(0, 11, 0, 0, 10, 10), (0, 42, 20, 0, 30, 10), (1, 42, 21, 0, 31, 10)])
    pr = table([(1, 5, 21, 0, 31, 10)])
    d = sequence_data(gt, pr, [0, 1])
    assert d["num_gt_ids"] == 2 and d["num_tracker_ids"] == 1 and d["num_gt_dets"] == 3
    assert d["gt_ids"][1].tolist() == [1] and d["tracker_ids"][0].tolist() == []
    assert d["similarity_scores"][1].shape == (1, 1) and d["similarity_scores"][1][0, 0] == pytest.approx(1.0)
