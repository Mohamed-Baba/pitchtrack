"""Pitch-line annotations <-> label maps, and the calibration of a predicted label map."""
import numpy as np
import pytest
from conftest import broadcast_camera, pitch_line_annotation

from pitchtrack.line_annotations import (CLASSES, FLIP_LUT, LABEL_OF, crop_points, flip_points, label_map_to_lines,
                                         mirror_name, normalised_points, render_label_map, to_annotation)
from pitchtrack.pitch_calibration import calibrate_frame, image_to_pitch, pitch_to_image

VIEWS = [((-30.0, 0.0), (-20.0, 74.0, -18.0)), ((40.0, -5.0), (25.0, 74.0, -16.0))]


def test_classes_and_mirror():
    assert len(CLASSES) == 20 and CLASSES[0] == "Side line top" and LABEL_OF["Circle right"] == 20
    assert mirror_name("Big rect. left top") == "Big rect. right top"
    assert mirror_name("Circle right") == "Circle left" and mirror_name("Middle line") == "Middle line"
    assert np.array_equal(FLIP_LUT[FLIP_LUT], np.arange(len(CLASSES) + 1))       # flipping twice: identity


def test_points_follow_flip_and_crop():
    pts = {"Side line left": np.array([[0.1, 0.2], [0.3, 0.9]])}
    f = flip_points(pts)
    assert list(f) == ["Side line right"] and np.allclose(f["Side line right"], [[0.9, 0.2], [0.7, 0.9]])
    c = crop_points(pts, (0.1, 0.0, 0.6, 0.5))
    assert np.allclose(c["Side line left"], [[0.0, 0.4], [0.4, 1.8]])
    raw = {"Middle line ": [{"x": 0.5, "y": 0.1}], "Goal left crossbar": [{"x": 0.2, "y": 0.2}]}
    assert list(normalised_points(raw)) == ["Middle line"]                         # unknown classes are dropped
    assert to_annotation({"Middle line": [[0.5, 0.1]]}) == {"Middle line": [{"x": 0.5, "y": 0.1}]}


def test_render_label_map_and_its_mirror():
    Hm, _, size = broadcast_camera()
    pts = normalised_points(pitch_line_annotation(Hm, size))
    lab = render_label_map(pts, (960, 544), 3)
    assert lab.shape == (544, 960) and lab.dtype == np.uint8
    assert set(np.unique(lab)) == {0} | {LABEL_OF[n] for n in pts}
    mirrored = render_label_map(flip_points(pts), (960, 544), 3)
    assert (FLIP_LUT[lab[:, ::-1]] == mirrored).mean() > 0.999                     # rasterisation differs by a pixel


@pytest.mark.parametrize("target,cam", VIEWS)
def test_label_map_back_to_lines_gives_the_homography(target, cam):
    """A perfect label map -> points on the centre of each line -> calibrate_frame -> the camera."""
    Hm, _, size = broadcast_camera(target, cam)
    lab = render_label_map(normalised_points(pitch_line_annotation(Hm, size)), (960, 544), 3)
    lines = label_map_to_lines(lab)
    cal = calibrate_frame(lines, size)
    assert cal.ok, cal.reason
    gx, gy = np.meshgrid(np.linspace(-52, 52, 30), np.linspace(-33, 33, 20))
    xy = np.c_[gx.ravel(), gy.ravel()]
    uv = pitch_to_image(Hm, xy)
    seen = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < size[1])
    back, _ = image_to_pitch(cal.G, uv[seen], margin=50.0)
    assert np.nanmedian(np.linalg.norm(back - xy[seen], axis=1)) < 0.1             # metres


def test_imperfect_label_map_still_calibrates():
    """The errors of a network: bands too wide, gaps (players), blobs, and one line given the class
    of its mirror image. Components, run centres and RANSAC on lines absorb them."""
    import cv2
    Hm, _, size = broadcast_camera(*VIEWS[0])
    lab = render_label_map(normalised_points(pitch_line_annotation(Hm, size)), (960, 544), 3)
    lab = cv2.dilate(lab, np.ones((3, 3), np.uint8))                               # 5-pixel bands
    rng = np.random.default_rng(0)
    for u in rng.integers(0, 940, 25):                                              # gaps
        lab[:, u:u + 12][rng.random((544, 12)) < 0.9] = 0
    for _ in range(6):                                                              # blobs
        v, u = rng.integers(20, 520), rng.integers(20, 940)
        lab[v:v + 6, u:u + 6] = rng.integers(1, len(CLASSES) + 1)
    lab[lab == LABEL_OF["Small rect. left main"]] = LABEL_OF["Small rect. right main"]   # left / right confusion
    conf = np.where(lab > 0, rng.uniform(0.55, 1.0, lab.shape), 1.0).astype(np.float32)
    cal = calibrate_frame(label_map_to_lines(lab, conf), size)
    assert cal.ok, cal.reason
    gx, gy = np.meshgrid(np.linspace(-52, 52, 30), np.linspace(-33, 33, 20))
    xy = np.c_[gx.ravel(), gy.ravel()]
    uv = pitch_to_image(Hm, xy)
    seen = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < size[1])
    back, _ = image_to_pitch(cal.G, uv[seen], margin=50.0)
    assert np.nanmedian(np.linalg.norm(back - xy[seen], axis=1)) < 0.3


def test_isolated_blobs_and_low_confidence_are_dropped():
    lab = np.zeros((100, 200), np.uint8)
    k = LABEL_OF["Middle line"]
    lab[10:90, 99:102] = k                                                          # the line
    lab[5:12, 10:17] = k                                                            # a blob far from it
    conf = np.ones(lab.shape, np.float32)
    lines = label_map_to_lines(lab, conf, min_pixels=20)
    u = np.array([p["x"] for p in lines["Middle line"]]) * 199
    assert np.all(np.abs(u - 100.0) < 1.0)                                          # centre of the 3-pixel band
    conf[:] = 0.3
    assert label_map_to_lines(lab, conf, min_conf=0.5) == {}
