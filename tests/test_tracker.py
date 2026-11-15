import numpy as np
import pytest

from pitchtrack.boxes import box_iou
from pitchtrack.detector import Detections
from pitchtrack.tracker import CONFIRMED, LOST, TENTATIVE, Tracker, TrackerConfig

DT = 0.04


def box_at(cx, cy, w=20.0, h=50.0):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def run(tracker, frames):
    """frames: list of (boxes, scores). Returns {frame: {track_id: box}}."""
    out = {}
    for f, (boxes, scores) in enumerate(frames):
        dets = Detections(np.array(boxes, float).reshape(-1, 4), scores, np.zeros(len(scores), int))
        ids, b, _ = tracker.update(dets, f * DT)
        out[f] = dict(zip(ids.tolist(), b))
    return out


def ids_seen(out):
    return sorted({i for d in out.values() for i in d})


# ----------------------------------------------------------------- association
def test_first_stage_matches_and_gates():
    tr = Tracker(TrackerConfig(n_init=1))
    run(tr, [([box_at(100, 100), box_at(300, 100)], [0.9, 0.9])])
    matches, ut, ud = tr.associate(Detections([box_at(102, 100), box_at(299, 101), box_at(600, 400)],
                                              [0.9, 0.8, 0.95], [0, 0, 0]))
    assert sorted(matches) == [(0, 0), (1, 1)]
    assert ut == [] and ud == [2]


def test_second_stage_uses_low_scores():
    for two_stage, expected in ((True, [(0, 0)]), (False, [])):
        tr = Tracker(TrackerConfig(n_init=1, two_stage=two_stage))
        run(tr, [([box_at(100, 100)], [0.9]), ([box_at(101, 100)], [0.9])])
        tr.predict(2 * DT)
        matches, ut, ud = tr.associate(Detections([box_at(102, 100)], [0.3], [0]))
        assert matches == expected and ud == []


def test_tentative_tracks_skip_the_second_stage():
    tr = Tracker(TrackerConfig(n_init=3))
    run(tr, [([box_at(100, 100)], [0.9])])                 # one tentative track
    tr.predict(DT)
    matches, ut, _ = tr.associate(Detections([box_at(101, 100)], [0.3], [0]))
    assert matches == [] and ut == [0]


# ----------------------------------------------------------------- life cycle
def test_confirmation_and_removal_of_tentative_tracks():
    tr = Tracker(TrackerConfig(n_init=3))
    out = run(tr, [([box_at(100, 100)], [0.9])] * 2)
    assert out[1] == {} and tr.tracks[0].state == TENTATIVE
    run(tr, [([box_at(100, 100)], [0.9])])
    assert tr.tracks[0].state == CONFIRMED
    tr2 = Tracker(TrackerConfig(n_init=3))
    run(tr2, [([box_at(100, 100)], [0.9]), ([], [])])     # a one-frame false positive
    assert tr2.tracks == []


def test_lost_then_removed_and_ids_never_reused():
    tr = Tracker(TrackerConfig(n_init=1, max_lost_s=0.2))
    run(tr, [([box_at(100, 100)], [0.9])])
    assert tr.tracks[0].state == CONFIRMED
    for k in range(1, 6):                                  # 0.2 s without detection
        tr.update(Detections.empty(), k * DT)
        assert tr.tracks[0].state == LOST
    tr.update(Detections.empty(), 6 * DT)
    assert tr.tracks == []
    ids, _, _ = tr.update(Detections([box_at(100, 100)], [0.9], [0]), 7 * DT)
    assert ids.tolist() == [2]


def test_low_scores_never_start_tracks():
    tr = Tracker(TrackerConfig(n_init=1))
    run(tr, [([box_at(100, 100)], [0.55])])                # above high_thresh, below new_track_thresh
    assert tr.tracks == []


def test_two_crossing_players_keep_their_ids():
    rng = np.random.default_rng(4)
    frames = []
    for k in range(60):
        a = box_at(100 + 4 * k, 200) + rng.normal(0, 0.8, 4)
        b = box_at(340 - 4 * k, 210) + rng.normal(0, 0.8, 4)
        frames.append(([a, b], [0.9, 0.85]))
    out = run(Tracker(), frames)
    assert len(ids_seen(out)) == 2
    first_left = min(out[10], key=lambda i: out[10][i][0])
    assert min(out[55], key=lambda i: out[55][i][0]) != first_left   # they crossed, ids followed


@pytest.mark.parametrize("motion, same_id", [("kalman", True), ("none", False)])
def test_occlusion_needs_motion_prediction(motion, same_id):
    frames = [([box_at(100 + 6 * k, 200)], [0.9]) for k in range(20)]
    frames += [([], [])] * 10                              # 0.4 s behind another player
    frames += [([box_at(100 + 6 * k, 200)], [0.9]) for k in range(30, 40)]
    out = run(Tracker(TrackerConfig(motion=motion)), frames)
    assert (len(ids_seen(out)) == 1) == same_id


def test_output_only_confirmed_and_updated():
    tr = Tracker(TrackerConfig(n_init=2))
    ids, boxes, scores = tr.update(Detections([box_at(50, 50)], [0.9], [0]), 0.0)
    assert ids.size == 0 and boxes.shape == (0, 4)
    ids, boxes, scores = tr.update(Detections([box_at(51, 50)], [0.8], [0]), DT)
    assert ids.tolist() == [1] and scores.tolist() == [0.8]
    assert box_iou(boxes, [box_at(51, 50)])[0, 0] > 0.8
