import numpy as np
import pytest

from pitchtrack.kalman import CHI2_4_95, H_MAT, KalmanBoxFilter, box_from_mean


# ----------------------------------------------------------------- prediction
def test_motion_matrices():
    kf = KalmanBoxFilter(q_pos=4.0, q_size=1.0)
    dt, h = 0.04, 50.0
    F, Q = kf.motion(dt, h)
    assert F.shape == Q.shape == (8, 8)
    np.testing.assert_allclose(F[:4, 4:], dt * np.eye(4))
    np.testing.assert_allclose(F[:4, :4], np.eye(4))
    s2 = (4.0 * h) ** 2                                     # centre x: sigma = q_pos h
    np.testing.assert_allclose(Q[np.ix_([0, 4], [0, 4])], s2 * np.array([[dt**4 / 4, dt**3 / 2],
                                                                         [dt**3 / 2, dt**2]]))
    np.testing.assert_allclose(Q, Q.T)
    assert np.linalg.eigvalsh(Q).min() > -1e-9
    _, Q2 = kf.motion(dt, 2 * h)
    np.testing.assert_allclose(Q2, 4 * Q)                   # std proportional to h


def test_predict_constant_velocity():
    kf = KalmanBoxFilter()
    mean = np.array([100, 200, 20, 50, 30, -10, 0, 0], float)
    cov = np.eye(8)
    m2, c2 = kf.predict(mean, cov, 0.5)
    np.testing.assert_allclose(m2[:4], [115, 195, 20, 50])
    np.testing.assert_allclose(m2[4:], mean[4:])
    assert np.trace(c2) > np.trace(cov) and np.allclose(c2, c2.T)
    np.testing.assert_array_equal(mean, [100, 200, 20, 50, 30, -10, 0, 0])   # inputs untouched


# ----------------------------------------------------------------- correction
def test_update_moves_towards_measurement_and_shrinks():
    kf = KalmanBoxFilter()
    mean, cov = kf.initiate([100, 100, 20, 50])
    mean, cov = kf.predict(mean, cov, 0.04)
    z = np.array([103, 99, 21, 50])
    m2, c2 = kf.update(mean, cov, z)
    assert np.all(np.abs(m2[:4] - z) <= np.abs(mean[:4] - z) + 1e-12)
    assert np.trace(c2) < np.trace(cov)
    np.testing.assert_allclose(c2, c2.T)
    assert np.linalg.eigvalsh(c2).min() > 0


def test_precise_sensor_follows_measurements():
    kf = KalmanBoxFilter(r_pos=1e-6, r_size=1e-6)
    mean, cov = kf.initiate([0, 0, 20, 50])
    m2, _ = kf.update(*kf.predict(mean, cov, 0.04), [5, -2, 20, 50])
    np.testing.assert_allclose(m2[:4], [5, -2, 20, 50], atol=1e-3)


def test_velocity_converges():
    rng = np.random.default_rng(0)
    kf = KalmanBoxFilter()
    v = np.array([120.0, -40.0])
    mean, cov = kf.initiate([300, 300, 20, 50])
    for k in range(1, 60):
        t_center = np.array([300, 300]) + v * 0.04 * k
        z = np.r_[t_center + rng.normal(0, 1.5, 2), 20, 50]
        mean, cov = kf.update(*kf.predict(mean, cov, 0.04), z)
    np.testing.assert_allclose(mean[4:6], v, atol=15)


def test_nees_is_consistent():
    """Average NEES ~ 8 when the data follow the model (normalised estimation error squared)."""
    rng = np.random.default_rng(1)
    kf = KalmanBoxFilter(q_pos=4.0, q_size=1e-3, r_pos=0.05, r_size=0.05, v0=0.5)
    dt, nees = 0.04, []
    for _ in range(200):
        mean, cov = kf.initiate([500, 300, 20, 50])
        x = rng.multivariate_normal(mean, cov)               # truth drawn from the prior
        for k in range(40):
            F, Q = kf.motion(dt, x[3])
            x = F @ x + rng.multivariate_normal(np.zeros(8), Q)
            z = H_MAT @ x + rng.multivariate_normal(np.zeros(4), kf.measurement_cov(x[3]))
            mean, cov = kf.update(*kf.predict(mean, cov, dt), z)
        e = x - mean
        nees.append(e @ np.linalg.solve(cov, e))
    assert 7.0 < np.mean(nees) < 9.0


# ----------------------------------------------------------------- gating
def test_gating_distance_formula():
    kf = KalmanBoxFilter()
    mean, cov = kf.initiate([100, 100, 20, 50])
    zs = np.array([[100, 100, 20, 50], [110, 95, 22, 48], [300, 300, 20, 50]], float)
    d2 = kf.gating_distance(mean, cov, zs)
    S = H_MAT @ cov @ H_MAT.T + kf.measurement_cov(mean[3])
    ref = [(z - mean[:4]) @ np.linalg.inv(S) @ (z - mean[:4]) for z in zs]
    np.testing.assert_allclose(d2, ref, rtol=1e-10)
    assert d2[0] == pytest.approx(0.0) and d2[2] > CHI2_4_95


def test_gate_keeps_95_percent():
    rng = np.random.default_rng(2)
    kf = KalmanBoxFilter()
    mean, cov = kf.predict(*kf.initiate([100, 100, 20, 50]), 0.04)
    S = H_MAT @ cov @ H_MAT.T + kf.measurement_cov(mean[3])
    zs = rng.multivariate_normal(mean[:4], S, size=20000)
    inside = kf.gating_distance(mean, cov, zs) <= CHI2_4_95
    assert inside.mean() == pytest.approx(0.95, abs=0.01)


def test_box_from_mean():
    np.testing.assert_allclose(box_from_mean([10, 20, 4, 8, 0, 0, 0, 0]), [8, 16, 12, 24])
