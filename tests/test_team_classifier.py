"""Shirt colours, k-means, team per track."""
import itertools

import cv2
import numpy as np

from pitchtrack.team_classifier import (UNKNOWN, assign_teams, colour_space, fit_team_colours, kmeans,
                                        match_clusters_to_labels, torso_colour, torso_box)

RED, WHITE, GRASS = (40, 40, 200), (235, 235, 235), (50, 150, 60)


def lab_of(bgr):
    return cv2.cvtColor(np.array([[bgr]], np.float32) / 255.0, cv2.COLOR_BGR2Lab)[0, 0]


def player_image():
    img = np.full((200, 300, 3), GRASS, np.uint8)
    cv2.rectangle(img, (40, 40), (60, 100), RED, -1)                  # box (36, 30, 64, 110): red torso
    cv2.rectangle(img, (40, 40), (47, 100), GRASS, -1)                # grass in the left part of the torso
    cv2.rectangle(img, (150, 40), (170, 100), WHITE, -1)              # box (146, 30, 174, 110): white torso
    return img


def test_torso_box_is_clipped():
    assert torso_box((-20, -40, 20, 40), (100, 100)) == (0, 0, 10, 0)
    assert torso_box((0, 0, 100, 100), (50, 300)) == (25, 15, 75, 50)


def test_median_lab_without_grass():
    img = player_image()
    feats, valid = torso_colour(img, [[36, 30, 64, 110], [146, 30, 174, 110], [230, 30, 258, 110]])
    assert valid.tolist() == [True, True, False]                       # the third torso is only grass
    assert np.allclose(feats[0], lab_of(RED), atol=1.0)
    assert np.allclose(feats[1], lab_of(WHITE), atol=1.0)
    assert np.isnan(feats[2]).all()


def test_boxes_partly_outside_and_tiny_boxes():
    img = np.full((200, 300, 3), GRASS, np.uint8)
    cv2.rectangle(img, (0, 40), (15, 100), RED, -1)                     # player cut by the left border
    feats, valid = torso_colour(img, [[-15, 30, 25, 110], [150, 50, 152, 52]], min_pixels=20)
    assert valid.tolist() == [True, False]
    assert np.allclose(feats[0], lab_of(RED), atol=1.0)


def test_recovers_separated_blobs_and_is_reproducible():
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    x = np.concatenate([c + rng.normal(0, 0.5, (50, 2)) for c in centres])
    truth = np.repeat(np.arange(3), 50)
    c, labels, inertia = kmeans(x, 3, seed=1)
    assert len(set(zip(truth, labels))) == 3                            # one cluster per blob
    c2, labels2, inertia2 = kmeans(x, 3, seed=1)
    assert np.array_equal(labels, labels2) and inertia == inertia2
    assert np.isclose(inertia, ((x - c[labels]) ** 2).sum())


def test_finds_the_optimum_of_a_small_problem():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (9, 2))
    best = min((((x - np.array([x[np.array(s) == j].mean(0) for j in (0, 1)])[np.array(s)]) ** 2).sum())
               for s in itertools.product((0, 1), repeat=9) if 0 < sum(s) < 9)
    _, _, inertia = kmeans(x, 2, n_init=20, seed=0)
    assert np.isclose(inertia, best)


def test_seeding_never_duplicates_a_centre():
    x = np.array([[0.0, 0.0]] * 20 + [[5.0, 5.0]])
    for seed in range(10):
        c, labels, _ = kmeans(x, 2, n_init=1, max_iter=1, seed=seed)
        assert {tuple(v) for v in c} == {(0.0, 0.0), (5.0, 5.0)}


def test_track_level_beats_frame_level_and_says_unknown():
    centres = colour_space(np.array([lab_of(RED), lab_of(WHITE), [20.0, 0.0, 0.0]]))
    rng = np.random.default_rng(0)
    f_red = lab_of(RED) + rng.normal(0, 25, (40, 3))                    # noisy frames of a red player
    f_ref = np.tile([20.0, 0.0, 0.0], (10, 1)) + rng.normal(0, 2, (10, 3))   # dark referee
    feats = np.concatenate([f_red, f_ref, [lab_of(WHITE)] * 2])
    tracks = np.array([1] * 40 + [2] * 10 + [3] * 2)
    valid = np.ones(len(feats), bool)
    frame = assign_teams(feats, valid, tracks, centres, [0, 1], level="frame")
    track = assign_teams(feats, valid, tracks, centres, [0, 1], level="track", min_frames=3)
    assert (frame[:40] != 0).sum() >= 3                                  # some frames wrong or unknown
    assert (track[:40] == 0).all()
    assert (track[40:50] == UNKNOWN).all()                               # far from both teams
    assert (track[50:] == UNKNOWN).all()                                 # too few frames


def test_margin_and_invalid_rows():
    centres = np.array([[0.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    feats = np.array([[0.0, 10.0, 0.0], [0.0, 2.0, 0.0], [np.nan] * 3])
    out = assign_teams(feats, [True, True, False], [1, 2, 3], centres, [0, 1], level="frame", l_weight=1.0)
    assert out.tolist() == [UNKNOWN, 0, UNKNOWN]


def test_fit_team_colours_and_matching():
    rng = np.random.default_rng(1)
    feats = np.concatenate([lab_of(RED) + rng.normal(0, 2, (30, 3)), lab_of(WHITE) + rng.normal(0, 2, (25, 3)),
                            np.tile([20.0, 0.0, 0.0], (5, 1))])
    centres, teams, labels = fit_team_colours(feats, k=3, seed=0)
    assert set(labels[:30]) == {teams[0]} and set(labels[30:55]) == {teams[1]}
    m = match_clusters_to_labels([1, 1, 0, UNKNOWN], ["left", "left", "right", "left"])
    assert m == {0: "right", 1: "left"}
