import numpy as np
import pytest

from pitchtrack.boxes import box_iou
from pitchtrack.detector import Detections, TiledDetector, batched_nms, merge_tile_detections, nms


# ----------------------------------------------------------------- NMS
def test_nms_keeps_best_of_a_cluster_and_far_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0.5, 0, 10.5, 10]], float)
    scores = np.array([0.8, 0.9, 0.5, 0.7])
    keep = nms(boxes, scores, 0.5)
    assert keep.tolist() == [1, 2]


def test_nms_threshold_is_strict():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], float)        # IoU = 1/3
    iou = box_iou(boxes[:1], boxes[1:])[0, 0]
    assert nms(boxes, [0.9, 0.8], iou).tolist() == [0, 1]             # not > threshold
    assert nms(boxes, [0.9, 0.8], iou - 1e-9).tolist() == [0]


def test_nms_ios_removes_partial_box():
    full, part = [100, 100, 120, 160], [100, 100, 120, 130]
    boxes, scores = np.array([full, part], float), np.array([0.9, 0.95])
    assert nms(boxes, scores, 0.6, "iou").tolist() == [1, 0]          # IoU 0.5: both survive
    assert nms(boxes, scores, 0.6, "ios").tolist() == [1]             # IoS 1.0: one survives


def test_nms_stable_ties_and_empty():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
    assert nms(boxes, [0.5, 0.5], 0.5).tolist() == [0]
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5).shape == (0,)


def test_batched_nms_is_classwise():
    d = Detections([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8], [0, 32])
    assert len(batched_nms(d, 0.5)) == 2


# ----------------------------------------------------------------- tiled inference
TILES = np.array([[0, 0, 100, 100], [60, 0, 160, 100]], float)     # frame 100 x 160


def test_merge_translates_and_deduplicates():
    # the same person at frame x in [70, 90], fully visible in both tiles
    t0 = Detections([[70, 10, 90, 60]], [0.8], [0])
    t1 = Detections([[10, 10, 30, 60]], [0.9], [0])
    merged = merge_tile_detections([t0, t1], TILES, (100, 160), threshold=0.6, metric="ios")
    assert len(merged) == 1
    np.testing.assert_allclose(merged.boxes, [[70, 10, 90, 60]])
    assert merged.scores[0] == pytest.approx(0.9)


def test_merge_drops_boxes_cut_by_an_inner_side():
    # person at frame x in [85, 105]: cut by the right side of tile 0, complete in tile 1.
    # The truncated box has the HIGHER score: NMS alone would keep it.
    t0 = Detections([[85, 10, 100, 60]], [0.95], [0])
    t1 = Detections([[25, 10, 45, 60]], [0.9], [0])
    merged = merge_tile_detections([t0, t1], TILES, (100, 160))
    np.testing.assert_allclose(merged.boxes, [[85, 10, 105, 60]])
    # sides of the frame are not inner sides: a box touching the frame border survives
    edge = merge_tile_detections([Detections([[0, 10, 12, 60]], [0.9], [0]), Detections.empty()],
                                 TILES, (100, 160))
    assert len(edge) == 1


def test_merge_keeps_distinct_people():
    t0 = Detections([[5, 5, 20, 40]], [0.7], [0])
    t1 = Detections([[80, 5, 95, 40]], [0.6], [0])
    merged = merge_tile_detections([t0, t1], TILES, (100, 160))
    np.testing.assert_allclose(np.sort(merged.boxes[:, 0]), [5, 140])


def test_tiled_detector_with_fake_base():
    """A fake detector that 'sees' one 20 x 40 person at a fixed frame position."""
    person = np.array([300.0, 200.0, 320.0, 240.0])

    def fake_base(images):
        out = []
        for img in images:
            # crops are views of the frame: recover their offset from the stamped coordinates
            oy, ox = int(img[0, 0, 0]) * 10, int(img[0, 0, 1]) * 10
            b = person - np.array([ox, oy, ox, oy])
            h, w = img.shape[:2]
            if b[0] >= 0 and b[1] >= 0 and b[2] <= w and b[3] <= h:
                out.append(Detections([b], [0.9], [0]))
            else:
                out.append(Detections.empty())
        return out

    frame = np.zeros((360, 640, 3), np.uint8)
    yy, xx = np.mgrid[0:360, 0:640]
    frame[..., 0], frame[..., 1] = yy // 10, xx // 10      # pixel stamps its own offset / 10
    det = TiledDetector(fake_base, tile=320, overlap=0.25, full_frame=True)
    out = det([frame])[0]
    assert len(out) == 1
    np.testing.assert_allclose(out.boxes[0], person)
