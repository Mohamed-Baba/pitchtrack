"""Distance and speed of a track."""
import numpy as np

from pitchtrack.statistics import heatmap, noise_distance_bias, smooth, split_segments, track_kinematics


def test_split_and_smooth():
    t = np.array([0.0, 0.04, 0.08, 1.0, 1.04])
    assert [s.tolist() for s in split_segments(t, 0.5)] == [[0, 1, 2], [3, 4]]
    line = np.c_[np.arange(5.0), np.zeros(5)]
    assert np.allclose(smooth(np.arange(5.0), line, 2.0), line)              # a straight run is unchanged
    bump = np.c_[[0.0, 0.0, 3.0, 0.0, 0.0], np.zeros(5)]
    assert np.allclose(smooth(np.arange(5.0), bump, 2.0)[:, 0], [0.0, 1.0, 1.0, 1.0, 0.0])


def test_constant_speed():
    t = np.arange(0, 10, 0.04)
    xy = np.c_[5.0 * t, np.zeros_like(t)]
    k = track_kinematics(t, xy)
    assert abs(k.distance_m - 5.0 * t[-1]) < 0.3
    assert abs(k.mean_speed - 5.0) < 0.05 and abs(k.max_speed - 5.0) < 0.05
    assert k.n_segments == k.n_reliable == 1 and k.coverage > 0.99


def test_no_distance_across_a_gap():
    t = np.r_[np.arange(0, 4, 0.04), np.arange(5, 9, 0.04)]
    xy = np.c_[np.where(t < 4.5, 2.0 * t, 2.0 * t + 20.0), np.zeros_like(t)]   # teleports 20 m in the gap
    k = track_kinematics(t, xy)
    assert k.n_segments == 2 and k.n_reliable == 2
    assert abs(k.distance_m - 2.0 * 2 * (4 - 0.04)) < 0.5


def test_noise_bias_and_its_cure():
    rng = np.random.default_rng(0)
    t = np.arange(0, 60, 0.04)
    xy = rng.normal(0.0, 0.2, (len(t), 2))                                    # a static player, 0.2 m noise
    raw = track_kinematics(t, xy, smooth_s=0.0, step_s=0.0, max_speed=1e9)
    assert abs(raw.distance_m / 60.0 - noise_distance_bias(0.2, 25.0)) < 0.05 * noise_distance_bias(0.2, 25.0)
    cured = track_kinematics(t, xy, smooth_s=1.0, step_s=0.5)
    assert cured.distance_m < 0.05 * raw.distance_m


def test_implausible_jump_and_short_segments():
    t = np.arange(0, 6, 0.04)
    xy = np.c_[1.0 * t, np.zeros_like(t)]
    xy[len(t) // 2:, 0] += 5.0                                                # calibration glitch: +5 m in one frame
    k = track_kinematics(t, xy, smooth_s=0.0)
    assert k.n_implausible_steps >= 1 and k.n_reliable == 0 and k.distance_m == 0.0
    assert np.isnan(k.mean_speed)
    nan_rows = np.c_[np.full(10, np.nan), np.zeros(10)]
    short = track_kinematics(np.arange(10) * 0.04, nan_rows)
    assert short.n_segments == 0 and short.distance_m == 0.0


def test_heatmap_sums_to_one():
    h = heatmap([[0.0, 0.0], [10.0, 5.0], [np.nan, 1.0]], bins=(4, 2))
    assert h.shape == (4, 2) and np.isclose(h.sum(), 1.0)
