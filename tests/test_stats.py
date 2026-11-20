"""Tests for xtft.evaluation.stats."""

import numpy as np
import pytest
from scipy import stats

from xtft.evaluation import diebold_mariano, friedman_nemenyi


def test_dm_matches_hand_computation_h1():
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 1.0, 50)
    b = rng.normal(1.2, 1.0, 50)
    d = a - b
    n = d.size
    dm = d.mean() / np.sqrt(d.var() / n)  # var with 1/T
    expected = dm * np.sqrt((n - 1) / n)  # HLN, h = 1
    res = diebold_mariano(a, b)
    assert res.statistic == pytest.approx(expected, rel=1e-12)
    p = 2 * stats.t(n - 1).sf(abs(expected))
    assert res.pvalue == pytest.approx(p, rel=1e-12)
    assert res.n == n


def test_dm_uses_autocovariances_for_overlap():
    rng = np.random.default_rng(2)
    e = rng.normal(size=301)
    d = e[1:] + 0.8 * e[:-1] + 0.1  # MA(1) differential
    zeros = np.zeros_like(d)
    h1 = diebold_mariano(d, zeros, horizon=1)
    h2 = diebold_mariano(d, zeros, horizon=2)
    # Positive lag-1 autocorrelation widens the variance.
    assert abs(h2.statistic) < abs(h1.statistic)


def test_dm_one_sided_directions():
    rng = np.random.default_rng(3)
    a = rng.gamma(2.0, size=300)
    b = a + rng.normal(0.2, 0.3, 300)
    assert diebold_mariano(a, b, alternative="less").pvalue < 0.01
    assert diebold_mariano(a, b, alternative="greater").pvalue > 0.99


def test_dm_drops_nan_pairs_and_handles_identical():
    a = np.array([1.0, np.nan, 2.0, 3.0, 4.0])
    b = np.array([1.5, 1.0, np.nan, 3.5, 4.5])
    assert diebold_mariano(a, b).n == 3
    res = diebold_mariano(np.ones(10), np.ones(10))
    assert res.statistic == 0.0 and res.pvalue == 1.0
    shifted = diebold_mariano(np.ones(10), np.ones(10) + 1)
    assert shifted.statistic == -np.inf and shifted.pvalue == 0.0


@pytest.mark.parametrize(
    ("a", "b", "kw"),
    [
        (np.ones(5), np.ones(4), {}),
        (np.ones(2), np.ones(2), {}),
        (np.ones(5), np.ones(5), {"horizon": 5}),
        (np.ones(5), np.ones(5), {"horizon": 0}),
        (np.ones(5), np.ones(5), {"alternative": "both"}),
    ],
)
def test_dm_validation(a, b, kw):
    with pytest.raises(ValueError):
        diebold_mariano(a, b, **kw)


def test_friedman_matches_scipy_without_ties():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, 4)) + np.array([0.0, 0.3, 0.6, 0.9])
    res = friedman_nemenyi(x)
    ref = stats.friedmanchisquare(*x.T)
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert res.pvalue == pytest.approx(ref.pvalue, rel=1e-10)
    assert res.average_ranks.sum() == pytest.approx(4 * 5 / 2)


def test_nemenyi_cd_matches_demsar_table():
    # Demsar (2006), Table 5(a): q_0.05 = 2.569 for k = 4.
    x = np.random.default_rng(5).normal(size=(10, 4))
    res = friedman_nemenyi(x, alpha=0.05)
    cd = 2.569 * np.sqrt(4 * 5 / (6 * 10))
    assert res.critical_difference == pytest.approx(cd, abs=2e-3)


def test_friedman_drops_incomplete_rows_and_validates():
    x = np.array([[1.0, 2.0], [np.nan, 1.0], [2.0, 1.0], [1, 3]])
    assert friedman_nemenyi(x).n_tasks == 3
    with pytest.raises(ValueError):
        friedman_nemenyi(np.ones((5, 1)))
    with pytest.raises(ValueError):
        friedman_nemenyi(np.ones((1, 3)))
    with pytest.raises(ValueError):
        friedman_nemenyi(np.ones((4, 3)), alpha=1.5)
