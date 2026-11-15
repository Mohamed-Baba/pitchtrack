import cv2
import numpy as np
import pytest

from pitchtrack.camera_motion import (IDENTITY, CameraMotionEstimator, estimate_camera_motion, warp_box,
                                      warp_track_state)


def texture(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.uniform(0, 255, (h // 8, w // 8)).astype(np.float32)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(img, 0, 255).astype(np.uint8)


def shift(img, A):
    return cv2.warpAffine(img, np.asarray(A, np.float64), (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ----------------------------------------------------------------- camera motion
def test_recovers_a_translation():
    a = texture()
    b = shift(a, [[1, 0, 5], [0, 1, -3]])
    A, ok = estimate_camera_motion(a, b)
    assert ok
    np.testing.assert_allclose(A, [[1, 0, 5], [0, 1, -3]], atol=0.3)


def test_recovers_a_zoom():
    a = texture(seed=1)
    Z = cv2.getRotationMatrix2D((160, 120), 0, 1.04)
    A, ok = estimate_camera_motion(a, shift(a, Z))
    assert ok and np.sqrt(abs(np.linalg.det(A[:, :2]))) == pytest.approx(1.04, abs=0.01)


def test_ignores_the_excluded_boxes():
    a = texture(seed=2)
    b = shift(a, [[1, 0, 4], [0, 1, 0]])
    b[60:180, 100:220] = shift(a, [[1, 0, -10], [0, 1, 0]])[60:180, 100:220]   # a big "player"
    A, ok = estimate_camera_motion(a, b, exclude_boxes=[[90, 50, 230, 190]])
    assert ok and A[0, 2] == pytest.approx(4, abs=0.4)


def test_fails_on_a_blank_frame():
    blank = np.full((240, 320), 128, np.uint8)
    A, ok = estimate_camera_motion(blank, blank)
    assert not ok and np.array_equal(A, IDENTITY)


# ----------------------------------------------------------------- tracks in the new frame
def test_warp_state():
    mean = np.array([100, 50, 20, 40, 10, -5, 1, 2], float)
    cov = np.diag(np.arange(1, 9, dtype=float))
    m, c = warp_track_state(mean, cov, IDENTITY)
    np.testing.assert_allclose(m, mean)
    np.testing.assert_allclose(c, cov)
    m, c = warp_track_state(mean, cov, [[1, 0, 7], [0, 1, -2]])
    np.testing.assert_allclose(m, [107, 48, 20, 40, 10, -5, 1, 2])
    np.testing.assert_allclose(c, cov)
    m, c = warp_track_state(mean, cov, [[2, 0, 0], [0, 2, 0]])
    np.testing.assert_allclose(m, [200, 100, 40, 80, 20, -10, 2, 4])
    np.testing.assert_allclose(c, 4 * cov)
    R = cv2.getRotationMatrix2D((0, 0), 30, 1.0)
    m, c = warp_track_state(mean, cov, R)
    assert np.linalg.norm(m[:2]) == pytest.approx(np.linalg.norm(mean[:2]))
    np.testing.assert_allclose(c, c.T)


def test_estimator_works_at_half_resolution():
    np.testing.assert_allclose(warp_box([0, 0, 10, 20], [[1, 0, 5], [0, 1, 5]]), [5, 5, 15, 25])
    est = CameraMotionEstimator(scale=0.5)
    a = cv2.cvtColor(texture(480, 640, seed=3), cv2.COLOR_GRAY2BGR)
    b = cv2.cvtColor(shift(texture(480, 640, seed=3), [[1, 0, 8], [0, 1, 6]]), cv2.COLOR_GRAY2BGR)
    est(a)
    A, ok = est(b)
    assert ok and A[0, 2] == pytest.approx(8, abs=0.6) and A[1, 2] == pytest.approx(6, abs=0.6)
