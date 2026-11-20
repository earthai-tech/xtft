"""Point and probabilistic forecast metrics (NumPy).

All functions take plain arrays, so forecasts from XTFT and from
any baseline are scored by the same code. Non-finite targets mark
missing observations and are excluded everywhere; the denominators
count observed entries only.

Shape convention: targets ``y`` have any shape ``(...)``; point
forecasts share it; quantile forecasts add a last axis of length
:math:`K`, ordered like `quantiles`.

References
----------
.. [1] Gneiting, T. and Raftery, A. E. (2007). Strictly proper
   scoring rules, prediction, and estimation. Journal of the
   American Statistical Association, 102(477), 359-378.
   https://doi.org/10.1198/016214506000001437
.. [2] Hyndman, R. J. and Koehler, A. B. (2006). Another look at
   measures of forecast accuracy. International Journal of
   Forecasting, 22(4), 679-688.
   https://doi.org/10.1016/j.ijforecast.2006.03.001
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "calibration_curve",
    "coverage",
    "crps_from_quantiles",
    "evaluate_quantile_forecast",
    "interval_pairs",
    "interval_score",
    "mae",
    "mase",
    "mean_width",
    "pinball",
    "rmse",
    "seasonal_scale",
]

logger = logging.getLogger(__name__)

_PAIR_TOL = 1e-9


def _levels(quantiles: Sequence[float]) -> NDArray[np.float64]:
    tau = np.asarray(quantiles, dtype=float)
    if tau.ndim != 1 or tau.size == 0:
        raise ValueError("quantiles must be a non-empty 1-D sequence.")
    if np.any((tau <= 0) | (tau >= 1)) or np.any(np.diff(tau) <= 0):
        raise ValueError(
            "quantiles must be strictly increasing and in (0, 1), "
            f"got {tau.tolist()}."
        )
    return tau


def _pair(y: ArrayLike, f: ArrayLike, extra_axis: bool):
    """Return float arrays and the observed-target mask."""
    y = np.asarray(y, dtype=float)
    f = np.asarray(f, dtype=float)
    expected = f.shape[:-1] if extra_axis else f.shape
    if y.shape != expected:
        raise ValueError(
            f"Target shape {y.shape} does not match forecast shape "
            f"{f.shape}."
        )
    return y, f, np.isfinite(y)


def _masked_mean(values: NDArray, mask: NDArray) -> float:
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    return float(values[mask].sum() / n)


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r"""Mean absolute error over observed targets.

    Parameters
    ----------
    y_true, y_pred : array_like
        Targets and point forecasts of the same shape.

    Returns
    -------
    float
        :math:`\frac{1}{|\Omega|}\sum_{i\in\Omega}|y_i - \hat y_i|`,
        NaN if no target is observed.

    Examples
    --------
    >>> from xtft.evaluation import mae
    >>> mae([1.0, 2.0, float("nan")], [0.0, 4.0, 9.0])
    1.5
    """
    y, f, m = _pair(y_true, y_pred, extra_axis=False)
    return _masked_mean(np.abs(y - f), m)


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r"""Root mean squared error over observed targets.

    Examples
    --------
    >>> from xtft.evaluation import rmse
    >>> rmse([0.0, 0.0], [3.0, 4.0]) == (12.5) ** 0.5
    True
    """
    y, f, m = _pair(y_true, y_pred, extra_axis=False)
    return float(np.sqrt(_masked_mean((y - f) ** 2, m)))


def seasonal_scale(y_train: ArrayLike, season: int = 1) -> float:
    r"""In-sample mean absolute seasonal difference (MASE scale).

    Parameters
    ----------
    y_train : array_like
        One training series, shape ``(n,)``; NaNs are skipped.
    season : int, default=1
        Seasonal period :math:`m \ge 1`.

    Returns
    -------
    float
        :math:`\frac{1}{n-m}\sum_{t=m}^{n-1}|y_t - y_{t-m}|`.

    Raises
    ------
    ValueError
        If the series is shorter than ``season + 1`` or the scale
        is zero or undefined.

    Notes
    -----
    This is the denominator of the mean absolute scaled error
    (MASE) [2]_ and of the mean scaled interval score (MSIS).

    Examples
    --------
    >>> from xtft.evaluation import seasonal_scale
    >>> seasonal_scale([1.0, 2.0, 4.0, 7.0])
    2.0
    """
    y = np.asarray(y_train, dtype=float).ravel()
    if season < 1 or y.size <= season:
        raise ValueError(
            f"Need more than season={season} points, got {y.size}."
        )
    diff = np.abs(y[season:] - y[:-season])
    scale = float(np.nanmean(diff)) if np.isfinite(diff).any() else 0.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            "Seasonal scale is zero or undefined (constant or "
            "all-missing training series)."
        )
    return scale


def mase(y_true: ArrayLike, y_pred: ArrayLike, scale: float) -> float:
    r"""Mean absolute scaled error, :math:`\mathrm{MAE}/\text{scale}`.

    Parameters
    ----------
    y_true, y_pred : array_like
        Targets and point forecasts.
    scale : float
        Positive scale, usually :func:`seasonal_scale` of the
        training part of the same series [2]_.

    Returns
    -------
    float

    Examples
    --------
    >>> from xtft.evaluation import mase
    >>> mase([1.0, 3.0], [2.0, 2.0], scale=2.0)
    0.5
    """
    if not scale > 0:
        raise ValueError(f"scale must be positive, got {scale}.")
    return mae(y_true, y_pred) / scale


def pinball(
    y_true: ArrayLike,
    y_quant: ArrayLike,
    quantiles: Sequence[float],
) -> float:
    r"""Mean pinball loss over observed targets and levels.

    Parameters
    ----------
    y_true : array_like
        Targets, shape ``(...)``.
    y_quant : array_like
        Quantile forecasts, shape ``(..., K)``.
    quantiles : sequence of float
        The :math:`K` levels, strictly increasing in ``(0, 1)``.

    Returns
    -------
    float
        :math:`\frac{1}{K|\Omega|}\sum_{i\in\Omega}\sum_k
        \rho_{\tau_k}(y_i - \hat q_{i,k})` with
        :math:`\rho_\tau(u) = \max(\tau u, (\tau-1)u)`.

    Examples
    --------
    >>> from xtft.evaluation import pinball
    >>> round(pinball([1.0], [[0.0, 1.0, 2.0]], [0.1, 0.5, 0.9]), 4)
    0.0667
    """
    tau = _levels(quantiles)
    y, q, m = _pair(y_true, y_quant, extra_axis=True)
    if q.shape[-1] != tau.size:
        raise ValueError(
            f"Forecast has {q.shape[-1]} quantiles, expected "
            f"{tau.size}."
        )
    u = y[..., None] - q
    loss = np.maximum(tau * u, (tau - 1) * u).mean(axis=-1)
    return _masked_mean(np.where(m, loss, 0.0), m)


def crps_from_quantiles(
    y_true: ArrayLike,
    y_quant: ArrayLike,
    quantiles: Sequence[float],
) -> float:
    r"""Continuous ranked probability score approximated by quantiles.

    Parameters
    ----------
    y_true, y_quant, quantiles
        As in :func:`pinball`.

    Returns
    -------
    float
        Twice the mean pinball loss.

    Notes
    -----
    The CRPS of a predictive distribution :math:`F` equals the
    integrated quantile score [1]_,

    .. math::

        \mathrm{CRPS}(F, y) = 2 \int_0^1
            \rho_\tau\bigl(y - F^{-1}(\tau)\bigr)\, d\tau,

    which the mean over :math:`K` levels approximates by a Riemann
    sum. The approximation is good only when the levels cover
    ``(0, 1)`` densely and roughly evenly (for example
    ``0.05, 0.10, ..., 0.95``); with three levels it is a coarse
    proxy and should be labelled as such.

    Examples
    --------
    >>> from xtft.evaluation import crps_from_quantiles
    >>> round(crps_from_quantiles([1.0], [[0.0, 1.0, 2.0]],
    ...                           [0.1, 0.5, 0.9]), 4)
    0.1333
    """
    return 2.0 * pinball(y_true, y_quant, quantiles)


def coverage(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike
) -> float:
    r"""Prediction interval coverage probability (PICP).

    Returns
    -------
    float
        Share of observed targets with
        :math:`\ell_i \le y_i \le u_i`.

    Examples
    --------
    >>> from xtft.evaluation import coverage
    >>> coverage([1.0, 5.0], [0.0, 0.0], [2.0, 2.0])
    0.5
    """
    y, lo, m = _pair(y_true, lower, extra_axis=False)
    _, hi, _ = _pair(y_true, upper, extra_axis=False)
    inside = (y >= lo) & (y <= hi)
    return _masked_mean(inside.astype(float), m)


def mean_width(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    normalize: bool = True,
) -> float:
    r"""Mean interval width, optionally normalised (PINAW).

    Parameters
    ----------
    y_true : array_like
        Targets; used for the mask and the normalising range.
    lower, upper : array_like
        Interval bounds.
    normalize : bool, default=True
        Divide by the range :math:`\max y - \min y` of the observed
        targets (PINAW); ``False`` gives the width in target units.

    Returns
    -------
    float

    Examples
    --------
    >>> from xtft.evaluation import mean_width
    >>> mean_width([0.0, 4.0], [0.0, 1.0], [2.0, 3.0])
    0.5
    """
    y, lo, m = _pair(y_true, lower, extra_axis=False)
    _, hi, _ = _pair(y_true, upper, extra_axis=False)
    width = _masked_mean(hi - lo, m)
    if not normalize:
        return width
    span = float(np.ptp(y[m])) if m.any() else float("nan")
    return width / span if span > 0 else float("nan")


def interval_score(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    alpha: float,
) -> float:
    r"""Mean interval (Winkler) score of central intervals.

    Parameters
    ----------
    y_true, lower, upper : array_like
        Targets and the bounds of the :math:`(1-\alpha)` interval.
    alpha : float
        Miscoverage level in ``(0, 1)``; an 80% interval has
        ``alpha = 0.2``.

    Returns
    -------
    float
        Lower is better.

    Notes
    -----
    Following [1]_ (Section 6.2),

    .. math::

        S_\alpha(\ell, u; y) = (u - \ell)
            + \frac{2}{\alpha}(\ell - y)\,\mathbb{1}\{y < \ell\}
            + \frac{2}{\alpha}(y - u)\,\mathbb{1}\{y > u\},

    a proper score that rewards narrow intervals and penalises
    misses in proportion to their distance. Dividing by
    :func:`seasonal_scale` gives the MSIS.

    Examples
    --------
    >>> from xtft.evaluation import interval_score
    >>> interval_score([1.0, 4.0], [0.0, 0.0], [2.0, 2.0], 0.2)
    12.0
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    y, lo, m = _pair(y_true, lower, extra_axis=False)
    _, hi, _ = _pair(y_true, upper, extra_axis=False)
    y0 = np.where(m, y, 0.0)
    score = (
        (hi - lo)
        + (2.0 / alpha) * np.clip(lo - y0, 0.0, None)
        + (2.0 / alpha) * np.clip(y0 - hi, 0.0, None)
    )
    return _masked_mean(np.where(m, score, 0.0), m)


def calibration_curve(
    y_true: ArrayLike,
    y_quant: ArrayLike,
    quantiles: Sequence[float],
) -> NDArray[np.float64]:
    r"""Observed frequency of :math:`y \le \hat q_\tau` per level.

    Returns
    -------
    numpy.ndarray
        Shape ``(K,)``. A calibrated forecast gives values close to
        `quantiles` (the reliability diagram plots one against the
        other).

    Examples
    --------
    >>> import numpy as np
    >>> from xtft.evaluation import calibration_curve
    >>> y = np.arange(10.0)
    >>> q = np.tile([2.0, 7.0], (10, 1))
    >>> calibration_curve(y, q, [0.25, 0.75])
    array([0.3, 0.8])
    """
    tau = _levels(quantiles)
    y, q, m = _pair(y_true, y_quant, extra_axis=True)
    if q.shape[-1] != tau.size:
        raise ValueError(
            f"Forecast has {q.shape[-1]} quantiles, expected "
            f"{tau.size}."
        )
    below = (y[..., None] <= q)[m]
    if below.size == 0:
        return np.full(tau.size, np.nan)
    return below.mean(axis=0)


def interval_pairs(
    quantiles: Sequence[float],
) -> list[tuple[int, int, float]]:
    r"""Find symmetric level pairs that form central intervals.

    Parameters
    ----------
    quantiles : sequence of float
        Strictly increasing levels in ``(0, 1)``.

    Returns
    -------
    list of (int, int, float)
        ``(i_lower, i_upper, nominal)`` with
        :math:`\tau_{i_u} = 1 - \tau_{i_\ell}` and nominal coverage
        :math:`\tau_{i_u} - \tau_{i_\ell}`, widest first.

    Examples
    --------
    >>> from xtft.evaluation import interval_pairs
    >>> interval_pairs([0.05, 0.1, 0.5, 0.9, 0.95])
    [(0, 4, 0.9), (1, 3, 0.8)]
    """
    tau = _levels(quantiles)
    pairs = []
    for i, lo in enumerate(tau):
        if lo >= 0.5:
            break
        match = np.nonzero(np.abs(tau - (1.0 - lo)) < _PAIR_TOL)[0]
        if match.size:
            j = int(match[0])
            pairs.append((i, j, round(float(tau[j] - lo), 10)))
    return pairs


def evaluate_quantile_forecast(
    y_true: ArrayLike,
    y_quant: ArrayLike,
    quantiles: Sequence[float],
    scale: float | None = None,
) -> dict[str, float]:
    r"""Compute the standard metric set for a quantile forecast.

    Parameters
    ----------
    y_true : array_like
        Targets, shape ``(...)``.
    y_quant : array_like
        Quantile forecasts, shape ``(..., K)``.
    quantiles : sequence of float
        The :math:`K` levels.
    scale : float or None, optional
        Seasonal scale for MASE and MSIS (see
        :func:`seasonal_scale`); omitted metrics are skipped.

    Returns
    -------
    dict of str to float
        ``pinball``, ``crps_q`` (quantile CRPS proxy) and
        ``calibration_error`` (mean :math:`|\hat F(\tau) - \tau|`);
        ``mae``, ``rmse`` (and ``mase``) on the median when 0.5 is a
        level; and, for every symmetric pair with nominal coverage
        :math:`c` (in percent), ``coverage_c``, ``pinaw_c``,
        ``width_c``, ``winkler_c`` (and ``msis_c``).

    Examples
    --------
    >>> import numpy as np
    >>> from xtft.evaluation import evaluate_quantile_forecast
    >>> rng = np.random.default_rng(0)
    >>> y = rng.normal(size=2000)
    >>> q = np.tile([-1.2816, 0.0, 1.2816], (2000, 1))
    >>> m = evaluate_quantile_forecast(y, q, [0.1, 0.5, 0.9])
    >>> round(m["coverage_80"], 2)
    0.8
    """
    tau = _levels(quantiles)
    q = np.asarray(y_quant, dtype=float)
    out = {
        "pinball": pinball(y_true, q, tau),
        "crps_q": crps_from_quantiles(y_true, q, tau),
        "calibration_error": float(
            np.nanmean(np.abs(calibration_curve(y_true, q, tau) - tau))
        ),
    }
    median = np.nonzero(np.abs(tau - 0.5) < _PAIR_TOL)[0]
    if median.size:
        point = q[..., int(median[0])]
        out["mae"] = mae(y_true, point)
        out["rmse"] = rmse(y_true, point)
        if scale is not None:
            out["mase"] = mase(y_true, point, scale)
    for i, j, nominal in interval_pairs(tau):
        tag = f"{round(100 * nominal):d}"
        lo, hi = q[..., i], q[..., j]
        alpha = 1.0 - nominal
        out[f"coverage_{tag}"] = coverage(y_true, lo, hi)
        out[f"pinaw_{tag}"] = mean_width(y_true, lo, hi)
        out[f"width_{tag}"] = mean_width(y_true, lo, hi, False)
        win = interval_score(y_true, lo, hi, alpha)
        out[f"winkler_{tag}"] = win
        if scale is not None:
            out[f"msis_{tag}"] = win / scale
    return out
