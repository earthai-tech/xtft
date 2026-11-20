"""Tests for simultaneous conformal bands."""

import math

import numpy as np
import pytest

from xtft.evaluation import SimultaneousConformalBand, path_coverage

LEVELS = [0.05, 0.1, 0.5, 0.9, 0.95]


def model(y, scale=1.0):
    """Per-step quantiles of N(0, scale^2) at LEVELS."""
    z = np.array([-1.645, -1.282, 0.0, 1.282, 1.645]) * scale
    return np.zeros(y.shape + (5,)) + z


@pytest.mark.parametrize("method", ["max", "bonferroni"])
def test_path_coverage_reaches_target_on_iid_paths(method):
    rng = np.random.default_rng(0)
    y_cal = rng.normal(size=(3000, 8))
    y_new = rng.normal(size=(3000, 8))
    band = SimultaneousConformalBand(LEVELS, 0.8, method=method)
    lo, hi = band.fit(y_cal, model(y_cal)).transform(model(y_new))
    cov = path_coverage(y_new, lo, hi)
    if method == "max":
        assert abs(cov - 0.8) < 0.03
    else:
        assert cov >= 0.78  # union bound: conservative


def test_max_band_is_narrower_than_bonferroni_for_dependent_steps():
    # A shared shock makes the steps strongly dependent: the union
    # bound wastes width, the max score does not.
    rng = np.random.default_rng(1)

    def paths(n):
        return rng.normal(size=(n, 1)) + 0.2 * rng.normal(size=(n, 12))

    y_cal, y_new = paths(3000), paths(3000)
    widths, covers = {}, {}
    for method in ("max", "bonferroni"):
        band = SimultaneousConformalBand(LEVELS, 0.9, method=method)
        lo, hi = band.fit(y_cal, model(y_cal)).transform(model(y_new))
        widths[method] = float(np.mean(hi - lo))
        covers[method] = path_coverage(y_new, lo, hi)
    assert widths["max"] < widths["bonferroni"]
    assert abs(covers["max"] - 0.9) < 0.03


def test_band_follows_the_model_uncertainty_shape():
    rng = np.random.default_rng(2)
    scale = np.linspace(0.5, 3.0, 6)  # error grows with the step
    y_cal = rng.normal(size=(2000, 6)) * scale
    y_new = rng.normal(size=(2000, 6)) * scale
    q_cal = model(y_cal) * scale[:, None]
    q_new = model(y_new) * scale[:, None]
    band = SimultaneousConformalBand(LEVELS, 0.8)
    lo, hi = band.fit(y_cal, q_cal).transform(q_new)
    width = (hi - lo).mean(axis=0)
    assert np.all(np.diff(width) > 0)
    assert abs(path_coverage(y_new, lo, hi) - 0.8) < 0.03


def test_band_contains_and_never_inverts():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(500, 4)) * 0.01  # model far too wide
    band = SimultaneousConformalBand(LEVELS, 0.8).fit(y, model(y))
    assert -1.0 <= band.margin_ < 0  # shrinks, clipped at -1
    lo, hi = band.transform(model(y))
    assert np.all(lo <= hi)


def test_too_few_windows_gives_unbounded_band():
    y = np.zeros((10, 24))
    band = SimultaneousConformalBand(LEVELS, 0.8, method="bonferroni")
    lo, hi = band.fit(y, model(y)).transform(model(y))
    assert np.isinf(lo).all() and np.isinf(hi).all()


def test_pair_selection_and_missing_targets():
    band = SimultaneousConformalBand(LEVELS, 0.8)
    assert band.pair_ == (1, 3)
    assert SimultaneousConformalBand(LEVELS, 0.85).pair_ == (0, 4)
    assert SimultaneousConformalBand(LEVELS, 0.9, base=0.8).pair_ == (
        1,
        3,
    )
    y = np.random.default_rng(4).normal(size=(200, 3))
    y[0] = np.nan
    y[1, 1] = np.nan
    band.fit(y, model(y))
    assert band.n_calibration_ == 199
    assert math.isnan(path_coverage(y[:1], y[:1], y[:1]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coverage": 1.0},
        {"coverage": 0.8, "method": "union"},
        {"coverage": 0.8, "base": 0.7},
    ],
)
def test_validation(kwargs):
    with pytest.raises(ValueError):
        SimultaneousConformalBand(LEVELS, **kwargs)


def test_shape_checks():
    y = np.zeros((50, 3))
    band = SimultaneousConformalBand(LEVELS)
    with pytest.raises(RuntimeError):
        band.transform(model(y))
    band.fit(y, model(y))
    with pytest.raises(ValueError, match="horizon"):
        band.transform(model(np.zeros((5, 4))))
    with pytest.raises(ValueError):
        path_coverage(y, y[:, :2], y)


def test_online_band_recovers_path_coverage_under_shift():
    rng = np.random.default_rng(5)
    h = 6
    y_cal = rng.normal(size=(800, h))
    scale = np.where(np.arange(3000) < 1500, 2.0, 0.7)[:, None]
    y = rng.normal(size=(3000, h)) * scale
    q_cal, q = model(y_cal), model(y)
    band = SimultaneousConformalBand(LEVELS, 0.8).fit(y_cal, q_cal)
    static = path_coverage(y, *band.transform(q))
    lo, hi = band.transform_online(
        y, q, times=np.arange(3000) * h, gamma=0.01, window=300
    )
    online = path_coverage(y, lo, hi)
    assert abs(static - 0.8) > 0.15
    assert abs(online - 0.8) < 0.04


def test_online_band_is_causal():
    rng = np.random.default_rng(6)
    y_cal = rng.normal(size=(300, 4))
    y = rng.normal(size=(200, 4))
    times = np.arange(200)  # overlapping paths, stride 1
    band = SimultaneousConformalBand(LEVELS, 0.8).fit(
        y_cal, model(y_cal)
    )
    lo, _ = band.transform_online(y, model(y), times, gamma=0.05)
    y2 = y.copy()
    y2[100] += 40.0  # targets 101..104, all observed by origin 104
    lo2, _ = band.transform_online(y2, model(y2), times, gamma=0.05)
    np.testing.assert_allclose(lo[:104], lo2[:104])
    assert not np.allclose(lo[104:], lo2[104:])


def test_online_band_validation():
    y = np.zeros((40, 2))
    band = SimultaneousConformalBand(LEVELS, method="bonferroni")
    band.fit(y, model(y))
    with pytest.raises(ValueError, match="max"):
        band.transform_online(y, model(y), np.arange(40))
    band = SimultaneousConformalBand(LEVELS)
    with pytest.raises(RuntimeError):
        band.transform_online(y, model(y), np.arange(40))
    band.fit(y, model(y))
    with pytest.raises(ValueError):
        band.transform_online(y, model(y), np.arange(40), gamma=-1)
    with pytest.raises(ValueError):
        band.transform_online(y, model(y), np.arange(39))
