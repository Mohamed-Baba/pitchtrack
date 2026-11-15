"""Pitch model, homography from lines, RANSAC, projection, uncertainty."""
import numpy as np
import pytest
from conftest import broadcast_camera, visible_line_points

from pitchtrack import pitch_model as pm
from pitchtrack.pitch_calibration import (calibrate_frame, dlt_from_lines, homogeneous, image_to_pitch,
                                          line_residuals, parse_lines, pitch_covariance, pitch_to_image,
                                          propagate, ransac_lines, well_posed)

PENALTY_VIEW = ["Side line top", "Side line left", "Big rect. left main", "Big rect. left top",
                "Big rect. left bottom", "Small rect. left main", "Small rect. left top", "Small rect. left bottom"]


def lines_of(names):
    return np.array([pm.LINES[n] for n in names])


def pitch_error(G, Hm, xy):
    """Metres between the pitch points xy and their round trip pitch -> image (true) -> pitch (G)."""
    back, ok = image_to_pitch(G, pitch_to_image(Hm, xy), margin=100.0)
    return np.where(ok, np.hypot(*(np.nan_to_num(back) - xy).T), np.inf)


def test_pitch_model_keys_and_samples():
    assert pm.LINE_KEY["Big rect. left top"] == pm.LINE_KEY["Big rect. right top"] == ("y", -20.16)
    assert pm.LINE_KEY["Small rect. right main"] == ("x", 47.0)
    for name, xy in pm.sample_segments(1.0).items():
        if pm.is_ground_line(name):
            assert np.abs(homogeneous(xy) @ pm.LINES[name]).max() < 1e-9
        else:
            assert pm.circle_residual(name, xy).max() < 1e-9


def test_parse_lines_scales_like_soccernet():
    lp = parse_lines({"Middle line": [{"x": 0.0, "y": 0.5}, {"x": 1.0, "y": 1.0}],
                      "Circle central": [{"x": 0.5, "y": 0.5}], "Goal left crossbar": [{"x": 0.1, "y": 0.1}],
                      "Goal left post left ": [{"x": 0.2, "y": 0.2}]}, 1920, 1080)
    assert np.allclose(lp.pts, [[0.0, 539.5], [1919.0, 1079.0]])
    assert lp.names == ["Middle line"] * 2 and lp.circle_names == ["Circle central"]
    assert np.allclose(lp.lines, [[1.0, 0.0, 0.0]] * 2)


def test_well_posed_needs_two_lines_of_each_direction():
    assert well_posed(lines_of(["Side line top", "Big rect. left top", "Middle line", "Big rect. left main"]))
    # three y-lines (parallel, concurrent at infinity) and one x-line: degenerate
    assert not well_posed(lines_of(["Side line top", "Big rect. left top", "Big rect. left bottom", "Middle line"]))
    # two classes on the same infinite line count once
    assert not well_posed(lines_of(["Big rect. left top", "Big rect. right top", "Middle line", "Big rect. left main"]))


def test_exact_recovery_and_sign():
    Hm, _, size = broadcast_camera()
    pts, names = visible_line_points(Hm, size, PENALTY_VIEW)
    G = dlt_from_lines(pts, lines_of(names))
    assert G is not None and np.isclose(np.linalg.norm(G), 1.0)
    assert (homogeneous(pts) @ G[2] > 0).all()
    grid = np.stack(np.meshgrid(np.linspace(-52, -20, 9), np.linspace(-33, 33, 9)), -1).reshape(-1, 2)
    assert pitch_error(G, Hm, grid).max() < 1e-6
    assert line_residuals(G, pts, lines_of(names)).max() < 1e-6


def test_degenerate_returns_none():
    Hm, _, size = broadcast_camera()
    pts, names = visible_line_points(Hm, size, ["Side line top", "Big rect. left top", "Big rect. left bottom",
                                                "Big rect. left main"])
    assert dlt_from_lines(pts, lines_of(names)) is None
    assert dlt_from_lines(pts[:5], lines_of(names[:5])) is None


def test_invariant_to_image_similarity():
    """Hartley normalisation: the estimate does not depend on the units of the image."""
    Hm, _, size = broadcast_camera()
    pts, names = visible_line_points(Hm, size, PENALTY_VIEW, noise_px=1.5, seed=3)
    L = lines_of(names)
    G = dlt_from_lines(pts, L)
    S = np.array([[1e3, 0.0, 5e6], [0.0, 1e3, -2e6], [0.0, 0.0, 1.0]])   # millipixels, shifted
    G2 = dlt_from_lines(homogeneous(pts) @ S.T[:, :2], L) @ S
    G2 *= np.sign(np.sum(G2 * G)) / np.linalg.norm(G2)
    assert np.abs(G2 - G).max() < 1e-9
    grid = np.stack(np.meshgrid(np.linspace(-50, -25, 6), np.linspace(-30, 30, 6)), -1).reshape(-1, 2)
    assert np.median(pitch_error(G, Hm, grid)) < 0.3


def best_by_points_and_by_lines(pts, L, thr_px=4.0):
    """Exhaustive search over all samples (2 x-lines, 2 y-lines): the inlier mask of the model with
    the most inlier points, and of the model with the most inlier lines."""
    import itertools
    from pitchtrack.pitch_calibration import line_groups
    groups = line_groups(L)
    xk = [k for k in groups if k[0] == "x"]
    yk = [k for k in groups if k[0] == "y"]
    best_p = best_l = None
    for xs in itertools.combinations(xk, 2):
        for ys in itertools.combinations(yk, 2):
            idx = np.concatenate([groups[k] for k in xs + ys])
            G = dlt_from_lines(pts[idx], L[idx])
            if G is None or np.linalg.det(G) <= 0:
                continue
            inl = line_residuals(G, pts, L) < thr_px
            n_lines = sum(inl[g].mean() >= 0.5 for g in groups.values())
            if best_p is None or (inl.sum(), n_lines) > best_p[0]:
                best_p = ((inl.sum(), n_lines), inl)
            if best_l is None or (n_lines, inl.sum()) > best_l[0]:
                best_l = ((n_lines, inl.sum()), inl)
    return best_p[1], best_l[1]


def test_ransac_counts_lines_before_points():
    """The halfway line (60 points) labelled as the line of the right penalty area (x = +36): a
    wrong model explains MORE points than the right one, but FEWER lines."""
    Hm, _, size = broadcast_camera(target=(-20.0, 0.0), cam=(-15.0, 74.0, -18.0))
    pts, names = visible_line_points(Hm, size, noise_px=0.5, seed=1)
    wrong = np.array([n == "Middle line" for n in names])
    L = lines_of(["Big rect. right main" if w else n for n, w in zip(names, wrong)])
    grid = np.stack(np.meshgrid(np.linspace(-45, -5, 6), np.linspace(-30, 30, 6)), -1).reshape(-1, 2)
    assert np.median(pitch_error(dlt_from_lines(pts, L), Hm, grid)) > 1.0      # plain DLT is ruined
    by_points, by_lines = best_by_points_and_by_lines(pts, L)
    assert by_points[wrong].mean() > 0.5 and by_lines[wrong].mean() < 0.2     # counting points is fooled
    G, inl, it = ransac_lines(pts, L, thr_px=4.0, seed=0)
    assert G is not None and it >= 1 and np.linalg.det(G) > 0
    assert inl[~wrong].mean() > 0.95 and inl[wrong].mean() < 0.2
    assert np.median(pitch_error(G, Hm, grid)) < 0.15


def test_clean_data_stops_early_and_calibrate_frame_end_to_end():
    Hm, _, size = broadcast_camera()
    pts, names = visible_line_points(Hm, size, PENALTY_VIEW, noise_px=0.3, seed=2)
    G, inl, it = ransac_lines(pts, lines_of(names), seed=0)
    assert inl.all() and it <= 3
    W, H = size
    raw = {}
    for (u, v), n in zip(pts, names):
        raw.setdefault(n, []).append({"x": u / (W - 1), "y": v / (H - 1)})
    circ = pitch_to_image(Hm, pm.sample_segments(1.0)["Circle left"])
    raw["Circle left"] = [{"x": u / (W - 1), "y": v / (H - 1)} for u, v in circ]
    cal = calibrate_frame(raw, size)
    assert cal.ok, cal.reason
    assert cal.circle_m < 0.1 and cal.n_xlines >= 2 and cal.n_ylines >= 2
    bad = calibrate_frame({k: v for k, v in raw.items() if "top" in k or "Middle" in k}, size)
    assert not bad.ok


def test_projection_horizon_and_margin():
    Hm, _, size = broadcast_camera(target=(0.0, 0.0), cam=(0.0, 60.0, -12.0), hfov_deg=60.0)
    G = np.linalg.inv(Hm)
    G /= np.linalg.norm(G)
    if np.median(homogeneous([[960, 900]]) @ G[2]) < 0:
        G = -G
    xy_true = np.array([[0.0, 0.0], [-30.0, 20.0], [40.0, -30.0]])
    xy, ok = image_to_pitch(G, pitch_to_image(Hm, xy_true))
    assert ok.all() and np.allclose(xy, xy_true, atol=1e-6)
    sky = np.array([[960.0, 5.0]])                       # above the horizon in this wide view
    assert (homogeneous(sky) @ G[2] < 0).all()
    xy_sky, ok_sky = image_to_pitch(G, sky)
    assert not ok_sky.any() and np.isnan(xy_sky).all()
    out = pitch_to_image(Hm, [[0.0, -34.0 - 8.0]])       # 8 m beyond the far touch line
    assert not image_to_pitch(G, out, margin=5.0)[1].any()
    assert image_to_pitch(G, out, margin=10.0)[1].all()


def test_covariance_matches_monte_carlo_and_grows_with_distance():
    Hm, _, size = broadcast_camera(target=(0.0, 0.0), cam=(0.0, 74.0, -18.0))
    G = np.linalg.inv(Hm)
    G /= np.linalg.norm(G)
    near, far = pitch_to_image(Hm, [[0.0, 25.0], [0.0, -30.0]])
    cov = pitch_covariance(G, np.stack([near, far]), np.eye(2) * 4.0)
    assert cov.shape == (2, 2, 2) and np.trace(cov[1]) > 4 * np.trace(cov[0])
    rng = np.random.default_rng(0)
    samples = far + rng.normal(0.0, 2.0, (20000, 2))
    xy, ok = image_to_pitch(G, samples, margin=100.0)
    emp = np.cov(xy[ok].T)
    assert np.linalg.norm(emp - cov[1]) / np.linalg.norm(cov[1]) < 0.05


def test_propagate_follows_the_image_motion():
    Hm, _, size = broadcast_camera()
    G = np.linalg.inv(Hm)
    M = np.array([[1.01, 0.002, -12.0], [-0.001, 1.01, 3.0], [1e-6, 0.0, 1.0]])   # p_cur ~ M p_prev
    G_cur = propagate(G, M)
    p_prev = pitch_to_image(Hm, [[-40.0, 10.0]])
    p_cur = homogeneous(p_prev) @ M.T
    xy, _ = image_to_pitch(G_cur * np.sign(np.sum(G_cur * G)), p_cur[:, :2] / p_cur[:, 2:], margin=50.0)
    assert np.allclose(xy, [[-40.0, 10.0]], atol=1e-6)
