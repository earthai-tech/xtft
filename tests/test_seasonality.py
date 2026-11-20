"""Tests for xtft.data.dominant_periods."""

import numpy as np
import pytest

from xtft.data import dominant_periods


def test_finds_daily_period_with_gaps_and_trend():
    rng = np.random.default_rng(0)
    t = np.arange(24 * 120)
    y = 0.01 * t + np.sin(2 * np.pi * t / 24) + rng.normal(size=t.size)
    y[rng.random(t.size) < 0.2] = np.nan
    assert dominant_periods(y, max_period=48)[0] == 24


def test_pools_several_series_of_different_lengths():
    rng = np.random.default_rng(1)
    series = [
        0.5 * np.sin(2 * np.pi * np.arange(n) / 12)
        + rng.normal(size=n)
        for n in (900, 1500, 2000)
    ]
    assert dominant_periods(series, max_period=30)[0] == 12


def test_rarely_reports_a_period_for_noise():
    rng = np.random.default_rng(2)
    alarms = sum(
        bool(dominant_periods(rng.normal(size=2000).cumsum()))
        for _ in range(100)
    )
    assert alarms <= 10  # ~4 % expected, see Notes


def test_degenerate_inputs():
    assert dominant_periods(np.ones(500)) == []
    assert dominant_periods(np.array([1.0, np.nan, 2.0])) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_period": 1},
        {"top": 0},
        {"rel_power": 0.0},
        {"alpha": 1.0},
        {"max_period": 1},
    ],
)
def test_validation(kwargs):
    with pytest.raises(ValueError):
        dominant_periods(
            np.random.default_rng(0).normal(size=200), **kwargs
        )
