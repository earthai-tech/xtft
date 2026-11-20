"""Tests for AdaptiveConformalCalibrator."""

import numpy as np
import pytest

from xtft.evaluation import (
    AdaptiveConformalCalibrator,
    ConformalQuantileCalibrator,
)

LEVELS = [0.1, 0.5, 0.9]


def flat(y, width=1.28):
    z = np.zeros_like(y)
    return np.stack([z - width, z, z + width], axis=-1)


def covered(y, q):
    return (y >= q[..., 0]) & (y <= q[..., -1])


def test_no_feedback_reduces_to_static_cqr():
    # All forecasts share one origin, so no target is observed in
    # time to update anything: margins equal the static CQR ones.
    rng = np.random.default_rng(0)
    y_val = rng.normal(size=(200, 3))
    y_test = rng.normal(size=(50, 3))
    q_val, q_test = flat(y_val, 0.5), flat(y_test, 0.5)
    static = ConformalQuantileCalibrator(LEVELS).fit(y_val, q_val)
    aci = AdaptiveConformalCalibrator(LEVELS, gamma=0.05)
    out = aci.fit(y_val, q_val).transform(
        y_test, q_test, times=np.zeros(50, dtype=int)
    )
    np.testing.assert_allclose(out, static.transform(q_test))


def test_is_causal():
    # Changing an outcome may only affect forecasts issued after
    # its target time.
    rng = np.random.default_rng(1)
    h, n = 4, 120
    y_val = rng.normal(size=(100, h))
    y = rng.normal(size=(n, h))
    times = np.arange(n)
    aci = AdaptiveConformalCalibrator(LEVELS, gamma=0.05, window=50)
    aci.fit(y_val, flat(y_val, 0.8))
    base = aci.transform(y, flat(y, 0.8), times)
    k = 60
    y2 = y.copy()
    y2[k] += 25.0  # targets at times k+1 .. k+h
    moved = aci.transform(y2, flat(y2, 0.8), times)
    changed = np.flatnonzero(~np.isclose(base, moved).all(axis=(1, 2)))
    # Only forecast k itself (its own quantiles do not depend on
    # y, but its row differs because flat(y2) is computed from
    # y2 here) and origins after its first target may change.
    assert changed.size > 0
    assert set(changed.tolist()) <= {k} | set(range(k + 1, n))
    np.testing.assert_allclose(base[:k], moved[:k])


def test_recovers_long_run_coverage_under_shift():
    rng = np.random.default_rng(2)
    y_val = rng.normal(size=(300, 1))
    scale = np.where(np.arange(3000) < 1500, 2.5, 0.6)
    y = scale[:, None] * rng.normal(size=(3000, 1))
    q = flat(y)
    static = ConformalQuantileCalibrator(LEVELS).fit(
        y_val, flat(y_val)
    )
    aci = AdaptiveConformalCalibrator(LEVELS, gamma=0.01, window=200)
    adaptive = aci.fit(y_val, flat(y_val)).transform(
        y, q, np.arange(3000)
    )
    cov_static = covered(y, static.transform(q)).mean()
    cov_adaptive = covered(y, adaptive).mean()
    assert abs(cov_adaptive - 0.8) < 0.03
    assert abs(cov_static - 0.8) > 0.1
    # Both regimes are tracked, not just the average.
    first = covered(y[:1500], adaptive[:1500]).mean()
    second = covered(y[1500:], adaptive[1500:]).mean()
    assert abs(first - 0.8) < 0.06 and abs(second - 0.8) < 0.06


def test_alpha_falls_after_misses_and_rises_after_hits():
    y_val = np.linspace(-1, 1, 101)[:, None]
    aci = AdaptiveConformalCalibrator(LEVELS, gamma=0.1)
    aci.fit(y_val, flat(y_val, 0.0))
    # Each target exceeds every earlier score, so even the clipped
    # (largest-score) margin misses it.
    far = 50.0 * np.arange(1, 6, dtype=float)[:, None]
    aci.transform(far, flat(far, 0.0), np.arange(5))
    path = aci.alpha_path_[(0, 2)][:, 0]
    assert path[0] == pytest.approx(0.2)
    assert np.all(np.diff(path) < 0)
    near = np.zeros((5, 1))  # every target hit
    aci.transform(near, flat(near, 0.0), np.arange(5))
    assert np.all(np.diff(aci.alpha_path_[(0, 2)][:, 0]) > 0)


def test_pooled_nan_targets_and_ordering():
    rng = np.random.default_rng(3)
    y_val = rng.normal(size=(80, 3))
    y = rng.normal(size=(30, 3))
    y[5, 1] = np.nan
    aci = AdaptiveConformalCalibrator(
        [0.05, 0.1, 0.5, 0.9, 0.95], per_horizon=False, window=40
    )
    q = np.stack(
        [np.full_like(y, v) for v in (-1.6, -1.2, 0, 1.2, 1.6)], -1
    )
    q_val = np.stack(
        [np.full_like(y_val, v) for v in (-1.6, -1.2, 0, 1.2, 1.6)],
        -1,
    )
    out = aci.fit(y_val, q_val).transform(y, q, np.arange(30))
    assert out.shape == q.shape
    assert np.all(np.diff(out, axis=-1) >= 0)
    assert aci.alpha_path_[(0, 4)].shape == (30, 1)
    # Same margin on every step when pooled.
    m = aci.margins_[(1, 3)]
    np.testing.assert_allclose(m, m[:, :1].repeat(3, axis=1))


@pytest.mark.parametrize(
    "kwargs",
    [{"gamma": -0.1}, {"window": 0}, {"window": 2.5}],
)
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        AdaptiveConformalCalibrator(LEVELS, **kwargs)


def test_needs_a_pair_and_fit_first():
    with pytest.raises(ValueError):
        AdaptiveConformalCalibrator([0.5])
    aci = AdaptiveConformalCalibrator(LEVELS)
    y = np.zeros((4, 2))
    with pytest.raises(RuntimeError):
        aci.transform(y, flat(y), np.arange(4))


def test_shape_checks():
    y = np.zeros((6, 2))
    aci = AdaptiveConformalCalibrator(LEVELS).fit(y, flat(y))
    with pytest.raises(ValueError, match="times"):
        aci.transform(y, flat(y), np.arange(5))
    y3 = np.zeros((6, 3))
    with pytest.raises(ValueError, match="horizon"):
        aci.transform(y3, flat(y3), np.arange(6))
    with pytest.raises(ValueError, match="shape"):
        aci.transform(y, flat(y)[..., :2], np.arange(6))
