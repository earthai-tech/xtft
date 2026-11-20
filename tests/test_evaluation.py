"""Tests for xtft.evaluation (metrics and conformal calibration)."""

import numpy as np
import pytest
from scipy import stats

from xtft.evaluation import (
    ConformalQuantileCalibrator,
    calibration_curve,
    conformal_margin,
    coverage,
    crps_from_quantiles,
    evaluate_quantile_forecast,
    interval_pairs,
    interval_score,
    mae,
    mean_width,
    pinball,
    rmse,
    seasonal_scale,
)

LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def gaussian_quantiles(n, mu=0.0, sigma=1.0, levels=LEVELS):
    q = stats.norm.ppf(levels, loc=mu, scale=sigma)
    return np.tile(q, (n, 1))


def test_point_metrics_skip_missing():
    y = np.array([1.0, np.nan, 3.0])
    f = np.array([2.0, 100.0, 1.0])
    assert mae(y, f) == pytest.approx(1.5)
    assert rmse(y, f) == pytest.approx(np.sqrt(2.5))


def test_all_missing_gives_nan():
    assert np.isnan(mae([np.nan], [1.0]))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mae(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        pinball(np.zeros(3), np.zeros((3, 2)), [0.5])


def test_crps_matches_gaussian_closed_form():
    # CRPS of N(0, 1) at y: y(2 Phi(y) - 1) + 2 phi(y) - 1/sqrt(pi)
    rng = np.random.default_rng(0)
    y = rng.normal(size=4000)
    levels = np.linspace(0.005, 0.995, 199)
    q = gaussian_quantiles(y.size, levels=levels)
    exact = np.mean(
        y * (2 * stats.norm.cdf(y) - 1)
        + 2 * stats.norm.pdf(y)
        - 1 / np.sqrt(np.pi)
    )
    approx = crps_from_quantiles(y, q, levels)
    assert approx == pytest.approx(exact, rel=0.01)


def test_calibrated_gaussian_has_nominal_coverage():
    rng = np.random.default_rng(1)
    y = rng.normal(size=20000)
    m = evaluate_quantile_forecast(
        y, gaussian_quantiles(y.size), LEVELS
    )
    assert m["coverage_90"] == pytest.approx(0.9, abs=0.01)
    assert m["coverage_80"] == pytest.approx(0.8, abs=0.01)
    assert m["calibration_error"] < 0.01
    curve = calibration_curve(y, gaussian_quantiles(y.size), LEVELS)
    assert np.allclose(curve, LEVELS, atol=0.015)


def test_interval_score_definition():
    y = np.array([1.0, -1.0, 5.0])
    lo, hi = np.zeros(3), np.full(3, 2.0)
    # widths 2; misses of 1 below and 3 above, alpha = 0.1
    expected = np.mean([2.0, 2.0 + 20 * 1.0, 2.0 + 20 * 3.0])
    assert interval_score(y, lo, hi, 0.1) == pytest.approx(expected)
    assert coverage(y, lo, hi) == pytest.approx(1 / 3)
    assert mean_width(y, lo, hi, normalize=False) == 2.0


def test_interval_score_is_minimised_by_true_quantiles():
    rng = np.random.default_rng(2)
    y = rng.normal(size=50000)
    true = stats.norm.ppf([0.1, 0.9])
    best = interval_score(
        y, np.full(y.size, true[0]), np.full(y.size, true[1]), 0.2
    )
    for shift in (-0.3, 0.3):
        worse = interval_score(
            y,
            np.full(y.size, true[0] + shift),
            np.full(y.size, true[1] - shift),
            0.2,
        )
        assert worse > best


def test_interval_pairs_and_tags():
    assert interval_pairs([0.1, 0.5, 0.9]) == [(0, 2, 0.8)]
    assert interval_pairs([0.2, 0.5, 0.7]) == []
    keys = evaluate_quantile_forecast(
        np.zeros(5), gaussian_quantiles(5), LEVELS
    )
    assert {"coverage_90", "winkler_10", "mae", "crps_q"} <= set(keys)


def test_seasonal_scale_errors():
    with pytest.raises(ValueError):
        seasonal_scale([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        seasonal_scale([1.0], season=1)


def test_conformal_margin_rank_and_small_samples():
    assert conformal_margin(np.arange(1, 100), 0.1) == 90.0
    with pytest.raises(ValueError, match="too few"):
        conformal_margin(np.arange(5), 0.1)


@pytest.mark.parametrize("width", [0.3, 3.0])
def test_cqr_restores_coverage(width):
    # Intervals too narrow (0.3) or too wide (3.0) for N(0, 1).
    rng = np.random.default_rng(3)
    levels = [0.1, 0.5, 0.9]
    y_cal, y_test = rng.normal(size=(2, 3000, 4))

    def fc(n):
        base = np.zeros((n, 4))
        return np.stack([base - width, base, base + width], axis=-1)

    cal = ConformalQuantileCalibrator(levels).fit(y_cal, fc(3000))
    q = cal.transform(fc(3000))
    cov = coverage(y_test, q[..., 0], q[..., 2])
    assert cov == pytest.approx(0.8, abs=0.02)
    assert cal.margins_[(0, 2)].shape == (4,)


def test_cqr_keeps_quantiles_sorted_and_input_untouched():
    rng = np.random.default_rng(4)
    levels = [0.1, 0.4, 0.5, 0.6, 0.9]
    y = rng.normal(size=(500, 3))
    q = np.tile(stats.norm.ppf(levels) * 3.0, (500, 3, 1))
    before = q.copy()
    out = ConformalQuantileCalibrator(levels).fit_transform(y, q)
    assert np.all(np.diff(out, axis=-1) >= 0)
    assert np.array_equal(q, before)


def test_cqr_pooled_and_multitarget():
    rng = np.random.default_rng(5)
    y = rng.normal(size=(800, 2, 3))
    q = np.tile([-1.0, 0.0, 1.0], (800, 2, 3, 1))
    cal = ConformalQuantileCalibrator(
        [0.1, 0.5, 0.9], per_horizon=False
    )
    out = cal.fit_transform(y, q)
    assert out.shape == q.shape
    assert cal.margins_[(0, 2)].shape == (1,)


def test_cqr_errors():
    with pytest.raises(ValueError, match="symmetric"):
        ConformalQuantileCalibrator([0.2, 0.5, 0.7])
    cal = ConformalQuantileCalibrator([0.1, 0.9])
    with pytest.raises(RuntimeError):
        cal.transform(np.zeros((2, 3, 2)))
    with pytest.raises(ValueError):
        cal.fit(np.zeros((20, 3)), np.zeros((20, 3, 3)))
    cal.fit(np.zeros((40, 3)), np.zeros((40, 3, 2)))
    with pytest.raises(ValueError, match="horizon"):
        cal.transform(np.zeros((2, 4, 2)))
