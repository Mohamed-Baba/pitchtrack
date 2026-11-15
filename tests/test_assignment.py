import itertools

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from pitchtrack.assignment import hungarian, linear_assignment

RNG = np.random.default_rng(3)


def brute_force(cost, max_cost):
    """Best value of sum(matched costs) + max_cost / 2 per unmatched row or column."""
    n, m = cost.shape
    best = np.inf
    for k in range(0, min(n, m) + 1):
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.permutations(range(m), k):
                c = cost[list(rows), list(cols)]
                if np.any(c > max_cost):
                    continue
                best = min(best, c.sum() + max_cost / 2 * (n + m - 2 * k))
    return best


def objective(cost, matches, max_cost):
    n, m = cost.shape
    return cost[matches[:, 0], matches[:, 1]].sum() + max_cost / 2 * (n + m - 2 * len(matches))


# ----------------------------------------------------------------- assignment with a gate
@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (3, 2), (4, 4), (3, 5)])
def test_optimal_against_brute_force(shape):
    for _ in range(20):
        cost = RNG.uniform(0, 1, shape)
        matches, ur, uc = linear_assignment(cost, 0.6)
        assert objective(cost, matches, 0.6) == pytest.approx(brute_force(cost, 0.6))
        assert np.all(cost[matches[:, 0], matches[:, 1]] <= 0.6)
        assert sorted(list(matches[:, 0]) + list(ur)) == list(range(shape[0]))
        assert sorted(list(matches[:, 1]) + list(uc)) == list(range(shape[1]))


def test_does_not_break_an_obvious_match():
    """Maximising the number of matches would pair A with 2 and B with 1."""
    cost = np.array([[0.0, 0.45], [0.45, 5.0]])
    matches, ur, uc = linear_assignment(cost, 0.5)
    assert matches.tolist() == [[0, 0]] and ur.tolist() == [1] and uc.tolist() == [1]


def test_empty():
    m, ur, uc = linear_assignment(np.zeros((0, 3)), 0.5)
    assert m.shape == (0, 2) and ur.size == 0 and uc.tolist() == [0, 1, 2]


# ----------------------------------------------------------------- Hungarian method
@pytest.mark.parametrize("shape", [(1, 1), (3, 3), (4, 7), (7, 4), (10, 10), (6, 9)])
def test_hungarian_matches_scipy(shape):
    for _ in range(10):
        cost = RNG.uniform(0, 10, shape)
        r, c = hungarian(cost)
        r_ref, c_ref = linear_sum_assignment(cost)
        assert cost[r, c].sum() == pytest.approx(cost[r_ref, c_ref].sum())
        assert len(set(r.tolist())) == len(r) == min(shape) and len(set(c.tolist())) == len(c)
        assert list(r) == sorted(r)


def test_hungarian_with_ties_and_negative_costs():
    cost = RNG.integers(-3, 3, (8, 8)).astype(float)
    r, c = hungarian(cost)
    r_ref, c_ref = linear_sum_assignment(cost)
    assert cost[r, c].sum() == cost[r_ref, c_ref].sum()
