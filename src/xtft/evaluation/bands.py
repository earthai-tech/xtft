r"""Simultaneous prediction bands over the whole forecast horizon.

A per-step interval with coverage :math:`1 - \alpha` covers each
horizon step with that probability, but the whole path
:math:`(y_1, \dots, y_H)` lies inside all :math:`H` intervals far
less often. Questions such as "will the price exceed a cap at any
hour tomorrow?" or "will subsidence exceed a threshold within five
years?" need a band that contains the whole path with probability
:math:`1 - \alpha`. Two split-conformal constructions are provided:

* ``"max"``: a normalised max-score band in the spirit of the
  modulation functions of Diquigiovanni et al. [2]_. Each
  calibration window is scored by its worst step, measured in units
  of the model's own half-width at that step, and one margin is
  applied to every step, so the band keeps the shape of the
  model's uncertainty;
* ``"bonferroni"``: per-step CQR at level :math:`\alpha / H`, as in
  Stankeviciute et al. [1]_; valid but conservative when the steps
  are dependent.

References
----------
.. [1] Stankeviciute, K., Alaa, A. M. and van der Schaar, M.
   (2021). Conformal time-series forecasting. Advances in Neural
   Information Processing Systems 34 (NeurIPS 2021).
.. [2] Diquigiovanni, J., Fontana, M. and Vantini, S. (2022).
   Conformal prediction bands for multivariate functional data.
   Journal of Multivariate Analysis, 189, 104879.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conformal import conformal_margin
from .metrics import _levels, interval_pairs

__all__ = ["SimultaneousConformalBand", "path_coverage"]

logger = logging.getLogger(__name__)

# Floor of the half-width used to normalise scores (degenerate
# intervals), in the units of the forecasts.
_WIDTH_FLOOR = 1e-8


def _margin_or_inf(scores: NDArray, alpha: float) -> float:
    """Conformal margin, or +inf when too few scores support it."""
    try:
        return conformal_margin(scores, alpha)
    except ValueError:
        logger.warning(
            "%d scores are too few for miscoverage %.4g: the band is "
            "unbounded.",
            int(np.isfinite(scores).sum()),
            alpha,
        )
        return math.inf


def path_coverage(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike
) -> float:
    r"""Fraction of forecast paths entirely inside their band.

    Parameters
    ----------
    y_true, lower, upper : array_like
        Shape ``(N, H)``: targets and band limits per window and
        step. Steps with a missing target are ignored; a window
        with no observed target is skipped.

    Returns
    -------
    float
        :math:`N^{-1} \sum_i \mathbb{1}\{\ell_{i,h} \le y_{i,h} \le
        u_{i,h} \;\forall h\}` over windows with an observed
        target; ``nan`` if there is none.

    Raises
    ------
    ValueError
        If the shapes differ or are not 2-D.

    Examples
    --------
    >>> import numpy as np
    >>> from xtft.evaluation import path_coverage
    >>> y = np.array([[0.0, 1.0], [0.0, 5.0]])
    >>> path_coverage(y, y - 1, y + 1), path_coverage(y, -1 + 0 * y,
    ...                                               2 + 0 * y)
    (1.0, 0.5)
    """
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if y.ndim != 2 or y.shape != lo.shape or y.shape != hi.shape:
        raise ValueError(
            "y_true, lower and upper must share a 2-D shape, got "
            f"{y.shape}, {lo.shape}, {hi.shape}."
        )
    observed = np.isfinite(y)
    inside = (y >= lo) & (y <= hi) | ~observed
    windows = observed.any(axis=1)
    if not windows.any():
        return math.nan
    return float(inside.all(axis=1)[windows].mean())


class SimultaneousConformalBand:
    r"""Split-conformal band covering whole forecast paths.

    Parameters
    ----------
    quantiles : sequence of float
        Levels of the last forecast axis, strictly increasing in
        ``(0, 1)``.
    coverage : float, default=0.8
        Target path coverage :math:`1 - \alpha`, in ``(0, 1)``.
    method : {"max", "bonferroni"}, default="max"
        Band construction (see Notes).
    base : float or None, default=None
        Nominal coverage of the quantile pair :math:`(\tau,
        1 - \tau)` that gives the band its shape (``"max"``); the
        default uses the pair whose coverage equals `coverage`, or
        else the widest pair. Ignored by ``"bonferroni"``, which
        always uses that same pair.

    Attributes
    ----------
    pair_ : tuple of int
        Indices of the quantile pair used.
    margin_ : float or numpy.ndarray
        Fitted margin: a scalar multiple of the half-width for
        ``"max"``, one additive margin per step, shape ``(H,)``, for
        ``"bonferroni"``.
    n_calibration_ : int
        Number of calibration windows with an observed target.

    Raises
    ------
    ValueError
        If `coverage` is outside ``(0, 1)``, `method` is unknown or
        no quantile pair matches `base`.

    See Also
    --------
    path_coverage : Empirical path coverage of a band.
    ConformalQuantileCalibrator : Per-step (marginal) calibration.

    Notes
    -----
    Let :math:`\hat\ell_{i,h}, \hat u_{i,h}` be the base quantile
    pair of window :math:`i` at step :math:`h` and
    :math:`w_{i,h} = \max((\hat u_{i,h} - \hat\ell_{i,h})/2,
    \epsilon)` its half-width. The ``"max"`` method scores each
    calibration window by

    .. math::

        S_i = \max_{h:\, y_{i,h} \text{ observed}}
            \frac{\max(\hat\ell_{i,h} - y_{i,h},\;
                       y_{i,h} - \hat u_{i,h})}{w_{i,h}},

    takes :math:`\hat Q` as the
    :math:`\lceil (n+1)(1-\alpha) \rceil`-th smallest of the
    :math:`n` scores (:func:`conformal_margin`), and returns

    .. math::

        [\hat\ell_{h} - \hat Q w_{h},\; \hat u_{h} + \hat Q w_{h}],
        \quad h = 1, \dots, H,

    with :math:`\hat Q` clipped at :math:`-1` so the band never
    inverts. If the calibration and test windows are exchangeable,
    a new path lies entirely in the band with probability at least
    :math:`1 - \alpha` [2]_. The ``"bonferroni"`` method applies
    per-step CQR margins at miscoverage :math:`\alpha / H`, so the
    union bound gives the same guarantee [1]_; it ignores the
    dependence between steps and is usually wider. Overlapping
    rolling windows are not exchangeable, so the guarantee is
    approximate in a backtest; report :func:`path_coverage` on
    held-out data.

    References
    ----------
    .. [1] Stankeviciute, K., Alaa, A. M. and van der Schaar, M.
       (2021). Conformal time-series forecasting. Advances in Neural
       Information Processing Systems 34 (NeurIPS 2021).
    .. [2] Diquigiovanni, J., Fontana, M. and Vantini, S. (2022).
       Conformal prediction bands for multivariate functional data.
       Journal of Multivariate Analysis, 189, 104879.

    Examples
    --------
    >>> import numpy as np
    >>> from xtft.evaluation import (SimultaneousConformalBand,
    ...                              path_coverage)
    >>> rng = np.random.default_rng(0)
    >>> levels = [0.1, 0.5, 0.9]
    >>> def model(y):  # calibrated per step, too narrow for paths
    ...     z = np.zeros_like(y)
    ...     return np.stack([z - 1.28, z, z + 1.28], axis=-1)
    >>> y_cal = rng.normal(size=(2000, 12))
    >>> y_new = rng.normal(size=(2000, 12))
    >>> q = model(y_new)
    >>> round(path_coverage(y_new, q[..., 0], q[..., 2]), 2) < 0.1
    True
    >>> band = SimultaneousConformalBand(levels, coverage=0.8)
    >>> lo, hi = band.fit(y_cal, model(y_cal)).transform(q)
    >>> abs(path_coverage(y_new, lo, hi) - 0.8) < 0.03
    True
    """

    def __init__(
        self,
        quantiles: Sequence[float],
        coverage: float = 0.8,
        *,
        method: str = "max",
        base: float | None = None,
    ) -> None:
        if not 0.0 < coverage < 1.0:
            raise ValueError(
                f"coverage must be in (0, 1), got {coverage}."
            )
        if method not in ("max", "bonferroni"):
            raise ValueError(f"Unknown method {method!r}.")
        self.quantiles = tuple(_levels(quantiles).tolist())
        self.coverage = float(coverage)
        self.method = method
        pairs = interval_pairs(self.quantiles)
        if not pairs:
            raise ValueError(
                "No symmetric quantile pair (tau, 1 - tau) in "
                f"{self.quantiles}."
            )
        target = coverage if base is None else base
        match = [p for p in pairs if math.isclose(p[2], target)]
        if match:
            chosen = match[0]
        elif base is None:
            chosen = max(pairs, key=lambda p: p[2])
        else:
            raise ValueError(
                f"No quantile pair with coverage {base}; available "
                f"{[round(p[2], 6) for p in pairs]}."
            )
        self.pair_ = (chosen[0], chosen[1])

    def _check(self, y_quant: NDArray, y_true: NDArray | None = None):
        if y_quant.ndim != 3 or y_quant.shape[-1] != len(
            self.quantiles
        ):
            raise ValueError(
                "Forecasts must have shape (N, H, K) with "
                f"K={len(self.quantiles)}, got {y_quant.shape}."
            )
        if y_true is not None and y_true.shape != y_quant.shape[:-1]:
            raise ValueError(
                f"Target shape {y_true.shape} does not match "
                f"forecast shape {y_quant.shape}."
            )

    def _limits(self, q: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        lo, hi = q[..., self.pair_[0]], q[..., self.pair_[1]]
        width = np.maximum((hi - lo) / 2.0, _WIDTH_FLOOR)
        return lo, hi, width

    def fit(
        self, y_true: ArrayLike, y_quant: ArrayLike
    ) -> SimultaneousConformalBand:
        """Estimate the band margin on calibration windows.

        Parameters
        ----------
        y_true : array_like
            Calibration targets, shape ``(N, H)``; NaN steps are
            ignored.
        y_quant : array_like
            Calibration forecasts, shape ``(N, H, K)``.

        Returns
        -------
        SimultaneousConformalBand
            The fitted band.

        Raises
        ------
        ValueError
            If the shapes disagree.

        Warns
        -----
        A logged warning, and an unbounded band (infinite margin),
        when there are fewer than about :math:`1/lpha` windows
        (``"max"``) or :math:`H/lpha` windows per step
        (``"bonferroni"``).
        """
        y = np.asarray(y_true, dtype=float)
        q = np.asarray(y_quant, dtype=float)
        self._check(q, y)
        lo, hi, width = self._limits(q)
        alpha = 1.0 - self.coverage
        score = np.maximum(lo - y, y - hi)
        windows = np.isfinite(y).any(axis=1)
        self.n_calibration_ = int(windows.sum())
        if self.method == "max":
            scaled = np.where(np.isfinite(y), score / width, -np.inf)
            path = scaled.max(axis=1)[windows]
            self._path_scores = path.tolist()
            self.margin_ = max(_margin_or_inf(path, alpha), -1.0)
        else:
            per_step = alpha / y.shape[1]
            self.margin_ = np.array(
                [
                    _margin_or_inf(score[:, h], per_step)
                    for h in range(y.shape[1])
                ]
            )
        self.horizon_ = y.shape[1]
        logger.debug("band margin %s", np.round(self.margin_, 4))
        return self

    def transform(
        self, y_quant: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Band limits for new forecasts.

        Parameters
        ----------
        y_quant : array_like
            Forecasts of shape ``(N, H, K)`` with the `H` of
            :meth:`fit`.

        Returns
        -------
        lower, upper : numpy.ndarray
            Band limits, shape ``(N, H)`` each, with
            ``lower <= upper``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If the shape does not match the fitted horizon.
        """
        if not hasattr(self, "margin_"):
            raise RuntimeError("Call fit() before transform().")
        q = np.asarray(y_quant, dtype=float)
        self._check(q)
        if q.shape[1] != self.horizon_:
            raise ValueError(
                f"Fitted for horizon {self.horizon_}, "
                f"got {q.shape[1]}."
            )
        lo, hi, width = self._limits(q)
        if self.method == "max":
            pad = self.margin_ * width
        else:
            pad = np.broadcast_to(self.margin_, lo.shape)
        lower, upper = lo - pad, hi + pad
        # A negative Bonferroni margin can cross a narrow interval;
        # collapse crossed limits to their midpoint. Infinite
        # (unbounded) limits never cross and are left as they are.
        crossed = lower > upper
        if crossed.any():
            mid = 0.5 * (lower[crossed] + upper[crossed])
            lower[crossed] = mid
            upper[crossed] = mid
        return lower, upper

    def transform_online(
        self,
        y_true: ArrayLike,
        y_quant: ArrayLike,
        times: ArrayLike,
        *,
        gamma: float = 0.005,
        window: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Shift-adaptive band, updated causally along the origins.

        The ``"max"`` band with its margin re-estimated at every
        origin from the path scores of windows already fully
        observed, with the adaptive conformal update of
        :class:`AdaptiveConformalCalibrator` applied to whole
        paths: a path that leaves its band lowers the working level
        (wider bands), a covered path raises it.

        Parameters
        ----------
        y_true : array_like
            Targets, shape ``(N, H)``; each path is used only once
            all its steps are past (time ``times[n] + H``).
        y_quant : array_like
            Forecasts, shape ``(N, H, K)``.
        times : array_like
            Integer origin of each forecast, shape ``(N,)``.
        gamma : float, default=0.005
            Step size of the level update, ``>= 0``.
        window : int or None, default=None
            Keep only the most recent `window` path scores.

        Returns
        -------
        lower, upper : numpy.ndarray
            Band limits, shape ``(N, H)`` each.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If the method is not ``"max"``, `gamma` is negative,
            `window` is not positive, or the shapes disagree.

        Notes
        -----
        With :math:`S_n` the normalised path score of forecast
        :math:`n` (see the class Notes) and :math:`\hat Q_n` the
        margin it was issued with, observing the full path updates
        :math:`\alpha \leftarrow \alpha + \gamma(\alpha^{*} -
        \mathbb{1}\{S_n > \hat Q_n\})`; the margin at an origin is
        the :math:`\lceil (n+1)(1-\alpha) \rceil`-th smallest of
        the :math:`n` buffered scores, rank clipped to
        :math:`[1, n]` and margin clipped at :math:`-1`. The ACI
        long-run guarantee [1]_ then applies to whole-path
        coverage, provided the level stays in :math:`(0, 1)`.

        References
        ----------
        .. [1] Gibbs, I. and Candes, E. (2021). Adaptive conformal
           inference under distribution shift. Advances in Neural
           Information Processing Systems 34 (NeurIPS 2021).
        """
        if not hasattr(self, "margin_"):
            raise RuntimeError("Call fit() before transform_online().")
        if self.method != "max":
            raise ValueError("transform_online needs method='max'.")
        if not gamma >= 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}.")
        if window is not None and window < 1:
            raise ValueError(f"window must be positive, got {window}.")
        y = np.asarray(y_true, dtype=float)
        q = np.asarray(y_quant, dtype=float)
        self._check(q, y)
        t = np.asarray(times)
        n, horizon = y.shape
        if t.shape != (n,) or horizon != self.horizon_:
            raise ValueError(
                f"Expected times of shape ({n},) and horizon "
                f"{self.horizon_}."
            )
        lo, hi, width = self._limits(q)
        scaled = np.where(
            np.isfinite(y), np.maximum(lo - y, y - hi) / width, -np.inf
        )
        score = scaled.max(axis=1)
        known = np.isfinite(y).any(axis=1)
        done = t + horizon  # time at which the path is fully seen
        order = np.argsort(done, kind="stable")
        alpha0 = 1.0 - self.coverage
        alpha = alpha0
        buffer = list(self._path_scores)
        issued = np.full(n, np.nan)
        ptr = 0
        for now in np.unique(t):
            while ptr < n and done[order[ptr]] <= now:
                k = order[ptr]
                if known[k]:
                    err = float(score[k] > issued[k])
                    alpha += gamma * (alpha0 - err)
                    buffer.append(score[k])
                ptr += 1
            recent = buffer[-window:] if window else buffer
            e = np.sort(np.asarray(recent))
            rank = math.ceil((e.size + 1) * (1.0 - alpha))
            margin = e[min(max(rank, 1), e.size) - 1]
            issued[t == now] = max(margin, -1.0)
        self.online_margins_ = issued
        pad = issued[:, None] * width
        return lo - pad, hi + pad
