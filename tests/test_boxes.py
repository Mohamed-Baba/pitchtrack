import numpy as np
import pytest

from pitchtrack.boxes import (BoxTransform, as_boxes, box_area, box_intersection, box_ios, box_iou,
                              cxcywh_to_xyxy, foot_points, letterbox_transform, tile_grid,
                              to_yolo_labels, xywh_to_xyxy, xyxy_to_cxcywh, xyxy_to_xywh)

RNG = np.random.default_rng(0)


def random_boxes(n, size=100.0):
    xy = RNG.uniform(0, size, (n, 2))
    wh = RNG.uniform(1, size / 3, (n, 2))
    return np.concatenate([xy, xy + wh], axis=1)


# ----------------------------------------------------------------- conversions
def test_xywh_roundtrip_and_values():
    b = np.array([[10.0, 20.0, 5.0, 8.0]])
    np.testing.assert_allclose(xywh_to_xyxy(b), [[10, 20, 15, 28]])
    boxes = random_boxes(50)
    np.testing.assert_allclose(xywh_to_xyxy(xyxy_to_xywh(boxes)), boxes)


def test_conversions_do_not_modify_input():
    b = np.array([[10.0, 20.0, 5.0, 8.0]])
    xywh_to_xyxy(b)
    xyxy_to_xywh(b)
    np.testing.assert_array_equal(b, [[10.0, 20.0, 5.0, 8.0]])


def test_cxcywh_roundtrip():
    boxes = random_boxes(50)
    c = xyxy_to_cxcywh(boxes)
    np.testing.assert_allclose(c[0, 2:], boxes[0, 2:] - boxes[0, :2])
    np.testing.assert_allclose(cxcywh_to_xyxy(c), boxes)


def test_area_and_degenerate_boxes():
    b = np.array([[0, 0, 4, 5], [3, 3, 3, 9], [5, 5, 1, 1]], dtype=float)
    np.testing.assert_allclose(box_area(b), [20, 0, 0])


def test_empty_inputs_keep_shapes():
    assert as_boxes([]).shape == (0, 4)
    assert xywh_to_xyxy(np.zeros((0, 4))).shape == (0, 4)
    assert box_area(np.zeros((0, 4))).shape == (0,)


# ----------------------------------------------------------------- IoU and IoS
def test_iou_identical_disjoint_nested():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    assert box_iou(a, a)[0, 0] == pytest.approx(1.0)
    assert box_iou(a, [[20, 20, 30, 30]])[0, 0] == 0.0
    assert box_iou(a, [[10, 0, 20, 10]])[0, 0] == 0.0          # touching edges
    assert box_iou(a, [[2, 2, 7, 7]])[0, 0] == pytest.approx(25 / 100)


def test_iou_zero_area_is_zero_not_nan():
    z = np.array([[5, 5, 5, 5]], dtype=float)
    iou = box_iou(z, np.vstack([z, [[0, 0, 10, 10]]]))
    assert np.all(np.isfinite(iou)) and np.all(iou == 0.0)


def test_iou_matrix_shape_symmetry_and_range():
    a, b = random_boxes(7), random_boxes(5)
    iou = box_iou(a, b)
    assert iou.shape == (7, 5)
    np.testing.assert_allclose(iou, box_iou(b, a).T)
    assert iou.min() >= 0.0 and iou.max() <= 1.0
    assert box_iou(np.zeros((0, 4)), b).shape == (0, 5)
    assert box_iou(a, np.zeros((0, 4))).shape == (7, 0)


def test_iou_matches_loop_reference():
    a, b = random_boxes(20), random_boxes(15)
    ref = np.zeros((20, 15))
    for i, p in enumerate(a):
        for j, q in enumerate(b):
            iw = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
            ih = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
            inter = iw * ih
            union = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
            ref[i, j] = inter / union
    np.testing.assert_allclose(box_iou(a, b), ref, atol=1e-12)
    assert (box_intersection(a, b) >= 0).all()


def test_shifted_box_formula():
    w, h = 12.0, 30.0
    for d in (0.0, 2.0, 4.0, 6.0):
        iou = box_iou([[0, 0, w, h]], [[d, 0, w + d, h]])[0, 0]
        assert iou == pytest.approx((w - d) / (w + d))


def test_ios_partial_box_inside_full_box():
    full = np.array([[100, 100, 120, 160]], dtype=float)
    half = np.array([[100, 100, 120, 130]], dtype=float)   # the upper half, cut by a tile border
    assert box_iou(full, half)[0, 0] == pytest.approx(0.5)
    assert box_ios(full, half)[0, 0] == pytest.approx(1.0)
    assert box_ios(full, [[0, 0, 0, 0]])[0, 0] == 0.0


# ----------------------------------------------------------------- box transforms
def test_transform_inverse_and_compose():
    t = BoxTransform(0.5, 2.0, 3.0, -4.0)
    boxes = random_boxes(10)
    np.testing.assert_allclose(t.inverse().apply(t.apply(boxes)), boxes, atol=1e-12)
    np.testing.assert_allclose(t.inverse().matrix() @ t.matrix(), np.eye(3), atol=1e-12)
    u = BoxTransform(3.0, 3.0, 1.0, 1.0)
    np.testing.assert_allclose(u.compose(t).apply(boxes), u.apply(t.apply(boxes)))
    np.testing.assert_allclose(u.compose(t).matrix(), u.matrix() @ t.matrix())


def test_transform_preserves_iou():
    t = BoxTransform(0.37, 1.9, 12.0, -7.0)
    a, b = random_boxes(8), random_boxes(6)
    np.testing.assert_allclose(box_iou(t.apply(a), t.apply(b)), box_iou(a, b), atol=1e-12)


def test_letterbox_numbers():
    t = letterbox_transform((1080, 1920), (1280, 1280))
    assert t.sx == pytest.approx(2 / 3) and t.sy == pytest.approx(2 / 3)
    assert (t.tx, t.ty) == (pytest.approx(0.0), pytest.approx(280.0))
    corners = t.apply([[0, 0, 1920, 1080]])
    np.testing.assert_allclose(corners, [[0, 280, 1280, 1000]])
    tall = letterbox_transform((1000, 500), (640, 640))      # portrait: horizontal padding
    np.testing.assert_allclose(tall.apply([[0, 0, 500, 1000]]), [[160, 0, 480, 640]])


def test_points_and_feet():
    np.testing.assert_allclose(foot_points([[10, 20, 30, 60]]), [[20, 60]])
    t = BoxTransform(2.0, 2.0, 1.0, 1.0)
    np.testing.assert_allclose(t.apply_points([[1, 2]]), [[3, 5]])


def test_yolo_labels_normalised_and_clipped():
    lines = to_yolo_labels([[96, 54, 192, 108], [-10, 0, 10, 20], [5, 5, 5, 9]], 1920, 1080)
    assert len(lines) == 2                                    # the degenerate box is dropped
    assert lines[0] == "0 0.075000 0.075000 0.050000 0.050000"
    assert lines[1].split()[1] == f"{5 / 1920:.6f}"           # clipped to x in [0, 10]


# ----------------------------------------------------------------- tiles
def test_tile_grid_covers_frame_with_overlap():
    tiles = tile_grid((1080, 1920), 640, 0.2)
    assert tiles[:, 2].max() == 1920 and tiles[:, 3].max() == 1080
    assert tiles[:, 0].min() == 0 and tiles[:, 1].min() == 0
    xs = np.unique(tiles[:, 0])
    assert np.all(np.diff(xs) <= 640 * 0.8 + 1)   # overlap at least 20 %
    assert len(tiles) == 8
    assert len(tile_grid((300, 300), 640, 0.2)) == 1
