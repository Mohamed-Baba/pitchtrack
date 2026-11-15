import cv2
import numpy as np
import pytest

from pitchtrack.pitch_mask import distance_to_mask, foot_distance_to_pitch, pitch_mask


def polygon_mask(shape, poly):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [poly], 1)
    return m


# ----------------------------------------------------------------- pitch mask
def test_mask_recovers_the_pitch(pitch_image):
    img, poly = pitch_image
    mask = pitch_mask(img, close_px=15, scale=0.5)
    assert mask.shape == img.shape[:2] and mask.dtype == np.uint8 and set(np.unique(mask)) <= {0, 1}
    truth = polygon_mask(img.shape[:2], poly)
    inter = np.logical_and(mask, truth).sum()
    union = np.logical_or(mask, truth).sum()
    assert inter / union > 0.95


def test_mask_fills_players_and_lines(pitch_image):
    img, _ = pitch_image
    mask = pitch_mask(img, close_px=15, scale=0.5)
    assert mask[220, 307] == 1          # inside the red player
    assert mask[200, 320] == 1          # on the white line
    assert mask[60, 47] == 0            # the person in the stands


def test_mask_without_green_is_empty():
    img = np.full((100, 200, 3), 128, np.uint8)
    assert pitch_mask(img).sum() == 0


def test_distance_to_mask():
    mask = np.zeros((100, 100), np.uint8)
    mask[50:, :] = 1
    d = distance_to_mask([[10, 70], [10, 40.5], [500, 500]], mask)
    assert d[0] == 0.0 and d[2] == 0.0                  # inside; clipped into the image
    assert d[1] == pytest.approx(10.0, abs=0.6)
    assert np.all(np.isinf(distance_to_mask([[1, 1]], np.zeros((10, 10), np.uint8))))


def test_foot_distance(pitch_image):
    img, _ = pitch_image
    mask = pitch_mask(img, close_px=15, scale=0.5)
    boxes = np.array([[300, 200, 315, 240], [40, 40, 55, 80]], float)
    d = foot_distance_to_pitch(boxes, mask)
    assert d[0] == 0.0 and d[1] > 30.0
