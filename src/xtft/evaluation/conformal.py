r"""Conformal calibration of quantile forecasts.

Quantile regressors, XTFT included, are fitted by minimising the
pinball loss, which gives no finite-sample guarantee that an
interval :math:`[\hat q_{\tau}, \hat q_{1-\tau}]` covers
:math:`1 - 2\tau` of new observations. Conformalized quantile
regression (CQR) [1]_ corrects each interval with a margin
estimated on held-out data, restoring the nominal coverage under
exchangeability (:class:`ConformalQuantileCalibrator`).

Forecast errors are not exchangeable when the series drifts: a
margin estimated on the validation period can be far too wide or
too narrow on the test period. :class:`AdaptiveConformalCalibrator`
re-estimates the margins causally as outcomes arrive, with the
adaptive conformal inference update of Gibbs and Candes [3]_ and an
optional sliding window of recent scores.

References
----------
.. [1] Romano, Y., Patterson, E. and Candes, E. (2019). Conformalized
   quantile regression. Advances in Neural Information Processing
   Systems 32 (NeurIPS 2019).
.. [2] Chernozhukov, V., Fernandez-Val, I. and Galichon, A. (2010).
   Quantile and probability curves without crossing.
   Econometrica, 78(3), 1093-1125.
   https://doi.org/10.3982/ECTA7880
.. [3] Gibbs, I. and Candes, E. (2021). Adaptive conformal inference
   under distribution shift. Advances in Neural Information
   Processing Systems 34 (NeurIPS 2021).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .metrics import _levels, interval_pairs

__all__ = [
    "AdaptiveConformalCalibrator",
    "ConformalQuantileCalibrator",
    "conformal_margin",
]

logger = logging.getLogger(__name__)


def conformal_margin(scores: ArrayLike, alpha: float) -> float:
    r"""Finite-sample conformal quantile of non-conformity scores.

    Parameters
    ----------
    scores : array_like
        Calibration scores :math:`E_1, \dots, E_n`; non-finite
        values are dropped.
    alpha : float
        Target miscoverage in ``(0, 1)``.

    Returns
    -------
    float
        The :math:`\lceil (n+1)(1-\alpha) \rceil`-th smallest
        score.

    Raises
    ------
    ValueError
        If `alpha` is outside ``(0, 1)`` or there are fewer than
        :math:`\lceil 1/\alpha \rceil - 1` finite scores, in which
        case the conformal margin is infinite.

    Notes
    -----
    If the calibration and test scores are exchangeable, then
    :math:`P(E_{n+1} \le \hat Q) \ge 1 - \alpha` with
    :math:`\hat Q` the value returned here [1]_.

    References
    ----------
    .. [1] Romano, Y., Patterson, E. and Candes, E. (2019).
       Conformalized quantile regression. Advances in Neural
       Information Processing Systems 32 (NeurIPS 2019).

    Examples
    --------
    >>> from xtft.evaluation import conformal_margin
    >>> conformal_margin(range(1, 20), alpha=0.1)
    18.0
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    e = np.asarray(scores, dtype=float).ravel()
    e = np.sort(e[np.isfinite(e)])
    n = e.size
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        raise ValueError(
            f"{n} calibration scores are too few for alpha={alpha}: "
            f"need at least {math.ceil(1.0 / alpha) - 1}."
        )
    return float(e[rank - 1])


class ConformalQuantileCalibrator:
    r"""Conformalized quantile regression for multi-horizon forecasts.

    Parameters
    ----------
    quantiles : sequence of float
        Levels of the last forecast axis, strictly increasing in
        ``(0, 1)``. Symmetric pairs :math:`(\tau, 1 - \tau)` are
        calibrated; other levels (the median, unpaired levels) are
        left unchanged.
    per_horizon : bool, default=True
        Estimate one margin per forecast step (axis 1). Errors grow
        with the horizon, so a pooled margin over-covers short
        horizons and under-covers long ones.

    Attributes
    ----------
    margins_ : dict of (int, int) to numpy.ndarray
        Fitted margins per interval ``(i_lower, i_upper)``, shape
        ``(H,)`` when `per_horizon`, else ``(1,)``.
    n_calibration_ : int
        Number of finite calibration targets used.

    See Also
    --------
    conformal_margin : The finite-sample quantile used here.
    xtft.evaluation.evaluate_quantile_forecast : Coverage checks.

    Notes
    -----
    For an interval with levels :math:`(\tau_\ell, \tau_u)` and
    nominal coverage :math:`1 - \alpha = \tau_u - \tau_\ell`, the
    CQR score of calibration point :math:`i` is [1]_

    .. math::

        E_i = \max\bigl(\hat q_{\tau_\ell}(x_i) - y_i,\;
                        y_i - \hat q_{\tau_u}(x_i)\bigr),

    and the calibrated interval is
    :math:`[\hat q_{\tau_\ell} - \hat Q,\; \hat q_{\tau_u} + \hat Q]`
    with :math:`\hat Q` from :func:`conformal_margin`. A negative
    :math:`\hat Q` narrows an over-wide interval. After all
    intervals are adjusted, the levels are sorted along the
    quantile axis (monotone rearrangement [2]_), which keeps the
    quantiles ordered without changing any interval's coverage.

    The guarantee needs exchangeable calibration and test errors.
    Forecast errors of a time series are serially dependent and
    may drift, so the guarantee is approximate: calibrate on the
    most recent validation window before the test period, and
    report the empirical test coverage.

    References
    ----------
    .. [1] Romano, Y., Patterson, E. and Candes, E. (2019).
       Conformalized quantile regression. Advances in Neural
       Information Processing Systems 32 (NeurIPS 2019).
    .. [2] Chernozhukov, V., Fernandez-Val, I. and Galichon, A. (2010).
       Quantile and probability curves without crossing.
       Econometrica, 78(3), 1093-1125.
       https://doi.org/10.3982/ECTA7880

    Examples
    --------
    >>> import numpy as np
    >>> from xtft.evaluation import ConformalQuantileCalibrator
    >>> rng = np.random.default_rng(0)
    >>> y = rng.normal(size=(500, 4))
    >>> narrow = np.stack([-0.5 * np.ones_like(y), 0 * y,
    ...                    0.5 * np.ones_like(y)], axis=-1)
    >>> cal = ConformalQuantileCalibrator([0.1, 0.5, 0.9])
    >>> cal = cal.fit(y, narrow)
    >>> q = cal.transform(narrow)
    >>> inside = (y >= q[..., 0]) & (y <= q[..., 2])
    >>> bool(inside.mean() >= 0.8)
    True
    """

    def __init__(
        self,
        quantiles: Sequence[float],
        per_horizon: bool = True,
    ) -> None:
        self.quantiles = tuple(_levels(quantiles).tolist())
        self.per_horizon = per_horizon
        self.pairs = interval_pairs(self.quantiles)
        if not self.pairs:
            raise ValueError(
                "No symmetric quantile pair (tau, 1 - tau) to "
                f"calibrate in {self.quantiles}."
            )

    def _check(self, y_quant: NDArray, y_true: NDArray | None = None):
        if y_quant.ndim < 2 or y_quant.shape[-1] != len(
            self.quantiles
        ):
            raise ValueError(
                "Forecasts must have shape (N, H, ..., K) with "
                f"K={len(self.quantiles)}, got {y_quant.shape}."
            )
        if y_true is not None and y_true.shape != y_quant.shape[:-1]:
            raise ValueError(
                f"Target shape {y_true.shape} does not match "
                f"forecast shape {y_quant.shape}."
            )

    def fit(
        self, y_true: ArrayLike, y_quant: ArrayLike
    ) -> ConformalQuantileCalibrator:
        """Estimate the margins on a calibration set.

        Parameters
        ----------
        y_true : array_like
            Calibration targets, shape ``(N, H, ...)``; NaNs are
            ignored.
        y_quant : array_like
            Calibration forecasts, shape ``(N, H, ..., K)``.

        Returns
        -------
        ConformalQuantileCalibrator
            The fitted calibrator.

        Raises
        ------
        ValueError
            If shapes disagree or there are too few calibration
            targets for an interval (see :func:`conformal_margin`).
        """
        y = np.asarray(y_true, dtype=float)
        q = np.asarray(y_quant, dtype=float)
        self._check(q, y)
        self.n_calibration_ = int(np.isfinite(y).sum())
        self.margins_ = {}
        for i, j, nominal in self.pairs:
            alpha = 1.0 - nominal
            scores = np.maximum(q[..., i] - y, y - q[..., j])
            if self.per_horizon:
                margin = np.array(
                    [
                        conformal_margin(scores[:, h], alpha)
                        for h in range(scores.shape[1])
                    ]
                )
            else:
                margin = np.array([conformal_margin(scores, alpha)])
            self.margins_[(i, j)] = margin
            logger.debug(
                "CQR %.0f%% interval: margins %s",
                100 * nominal,
                np.round(margin, 4),
            )
        return self

    def transform(self, y_quant: ArrayLike) -> NDArray[np.float64]:
        """Apply the fitted margins to new forecasts.

        Parameters
        ----------
        y_quant : array_like
            Forecasts of shape ``(N, H, ..., K)`` with the same `H`
            as in :meth:`fit` when `per_horizon`.

        Returns
        -------
        numpy.ndarray
            Calibrated, sorted quantiles of the same shape; the
            input is not modified.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If the shape does not match the fitted horizon.
        """
        if not hasattr(self, "margins_"):
            raise RuntimeError("Call fit() before transform().")
        q = np.array(y_quant, dtype=float, copy=True)
        self._check(q)
        extra = q.ndim - 3  # target axes between H and K
        for (i, j), margin in self.margins_.items():
            if self.per_horizon and margin.size != q.shape[1]:
                raise ValueError(
                    f"Fitted for horizon {margin.size}, got "
                    f"{q.shape[1]}."
                )
            m = margin.reshape((1, -1) + (1,) * max(extra, 0))
            q[..., i] -= m
            q[..., j] += m
        return np.sort(q, axis=-1)

    def fit_transform(
        self, y_true: ArrayLike, y_quant: ArrayLike
    ) -> NDArray[np.float64]:
        """Fit on the given data and return calibrated forecasts."""
        return self.fit(y_true, y_quant).transform(y_quant)


def _rank_select(scores: NDArray, alpha: float) -> float:
    """Conformal quantile with the rank clipped to ``[1, n]``."""
    n = scores.size
    rank = math.ceil((n + 1) * (1.0 - alpha))
    rank = min(max(rank, 1), n)
    return float(np.partition(scores, rank - 1)[rank - 1])


class AdaptiveConformalCalibrator:
    r"""Causal, shift-adaptive conformal calibration of quantiles.

    Starts from the CQR scores of a calibration (validation)
    period, then walks through the forecast origins in time order.
    At each origin the interval margins are computed only from the
    scores of targets already observed; once a target is observed,
    its score joins the score buffer and the working miscoverage
    level is updated with the adaptive conformal inference (ACI)
    rule [1]_. Margins therefore follow a change in the error
    distribution instead of keeping the validation-period value.

    Parameters
    ----------
    quantiles : sequence of float
        Levels of the last forecast axis, strictly increasing in
        ``(0, 1)``. Symmetric pairs :math:`(\tau, 1 - \tau)` are
        calibrated; other levels are left unchanged.
    gamma : float, default=0.005
        ACI step size :math:`\gamma \ge 0`; 0 keeps the nominal
        level and adapts through the score window only.
    window : int or None, default=None
        Keep only the most recent `window` scores per horizon step
        (validation scores first, then observed test scores);
        ``None`` keeps all of them.
    per_horizon : bool, default=True
        One score buffer and one level per forecast step; if
        ``False``, all steps share them.

    Attributes
    ----------
    origin_times_ : numpy.ndarray
        Sorted distinct origin times seen by :meth:`transform`.
    alpha_path_ : dict of (int, int) to numpy.ndarray
        Working miscoverage level used at each origin, shape
        ``(n_origins, G)`` with ``G = H`` if `per_horizon` else 1.
    margins_ : dict of (int, int) to numpy.ndarray
        Margin applied to each forecast, shape ``(N, H)``.

    Raises
    ------
    ValueError
        If `gamma` is negative, `window` is not a positive integer,
        or there is no symmetric quantile pair.

    See Also
    --------
    ConformalQuantileCalibrator : Static split-conformal CQR.
    conformal_margin : Finite-sample conformal quantile.

    Notes
    -----
    For an interval with levels :math:`(\tau_\ell, \tau_u)` and
    target miscoverage :math:`\alpha^{*} = 1 - (\tau_u -
    \tau_\ell)`, the CQR score of forecast :math:`n` at step
    :math:`h` is :math:`E_{n,h} = \max(\hat q_{\tau_\ell} - y,\;
    y - \hat q_{\tau_u})`. Forecast :math:`n` is issued at origin
    :math:`t_n` and its step :math:`h` (0-based) targets time
    :math:`t_n + h + 1`. At an origin :math:`t`, with
    :math:`\mathcal{E}_h(t)` the buffered scores of step :math:`h`
    whose targets are at or before :math:`t` (calibration scores
    included) and :math:`n_h` their number, the margin
    :math:`\hat Q` is the
    :math:`\lceil (n_h + 1)(1 - \alpha_{h}) \rceil`-th smallest
    score in :math:`\mathcal{E}_h(t)`, and the interval is
    :math:`[\hat q_{\tau_\ell} - \hat Q,\; \hat q_{\tau_u} +
    \hat Q]`. When the target of :math:`(n, h)` is observed, with
    :math:`\mathrm{err} = \mathbb{1}\{E_{n,h} > \hat Q_{n,h}\}`
    for the margin :math:`\hat Q_{n,h}` it was issued with,

    .. math::

        \alpha_{h} \leftarrow \alpha_{h} + \gamma\,
            (\alpha^{*} - \mathrm{err}),

    so misses lower the level (wider intervals) and hits raise it.
    Over :math:`T` updates, ACI bounds the gap between the
    long-run miscoverage and :math:`\alpha^{*}` by
    :math:`(\max(\alpha_1, 1 - \alpha_1) + \gamma)/(\gamma T)`
    whatever the data [1]_. Here the rank is clipped to
    :math:`[1, n_h]`: a level :math:`\alpha_h \le 0` gives the
    largest buffered score instead of an infinite interval, and
    :math:`\alpha_h \ge 1` the smallest instead of an empty one,
    so the bound holds only while the level stays in
    :math:`(0, 1)`. All windows sharing an origin are calibrated
    with the same margins; outcomes become available exactly at
    their target time (no reporting delay). After the margins are
    applied the levels are sorted (monotone rearrangement), as in
    :class:`ConformalQuantileCalibrator`.

    The update costs :math:`O(1)` per observed target; computing
    the margins costs :math:`O(\sum_h n_h)` per origin and
    interval.

    References
    ----------
    .. [1] Gibbs, I. and Candes, E. (2021). Adaptive conformal
       inference under distribution shift. Advances in Neural
       Information Processing Systems 34 (NeurIPS 2021).

    Examples
    --------
    Validation errors have unit scale, test errors triple: the
    static margin under-covers, the adaptive one recovers.

    >>> import numpy as np
    >>> from xtft.evaluation import (AdaptiveConformalCalibrator,
    ...                              ConformalQuantileCalibrator)
    >>> rng = np.random.default_rng(0)
    >>> levels = [0.1, 0.5, 0.9]
    >>> def flat(y):  # a forecaster unaware of the shift
    ...     z = np.zeros_like(y)
    ...     return np.stack([z - 1.28, z, z + 1.28], axis=-1)
    >>> y_val = rng.normal(size=(300, 1))
    >>> y_test = 3.0 * rng.normal(size=(600, 1))
    >>> static = ConformalQuantileCalibrator(levels).fit(
    ...     y_val, flat(y_val)).transform(flat(y_test))
    >>> aci = AdaptiveConformalCalibrator(levels, gamma=0.02,
    ...                                   window=100)
    >>> adaptive = aci.fit(y_val, flat(y_val)).transform(
    ...     y_test, flat(y_test), times=np.arange(600))
    >>> def cover(q):
    ...     inside = (y_test >= q[..., 0]) & (y_test <= q[..., 2])
    ...     return float(inside.mean())
    >>> cover(static) < 0.5, cover(adaptive) > 0.75
    (True, True)
    """

    def __init__(
        self,
        quantiles: Sequence[float],
        *,
        gamma: float = 0.005,
        window: int | None = None,
        per_horizon: bool = True,
    ) -> None:
        if not gamma >= 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}.")
        if window is not None and (
            not isinstance(window, (int, np.integer)) or window < 1
        ):
            raise ValueError(
                "window must be a positive int or None, got "
                f"{window!r}."
            )
        self.quantiles = tuple(_levels(quantiles).tolist())
        self.gamma = float(gamma)
        self.window = None if window is None else int(window)
        self.per_horizon = per_horizon
        self.pairs = interval_pairs(self.quantiles)
        if not self.pairs:
            raise ValueError(
                "No symmetric quantile pair (tau, 1 - tau) to "
                f"calibrate in {self.quantiles}."
            )

    def _check(self, y: NDArray, q: NDArray) -> None:
        if q.ndim != 3 or q.shape[-1] != len(self.quantiles):
            raise ValueError(
                "Forecasts must have shape (N, H, K) with "
                f"K={len(self.quantiles)}, got {q.shape}."
            )
        if y.shape != q.shape[:-1]:
            raise ValueError(
                f"Target shape {y.shape} does not match forecast "
                f"shape {q.shape}."
            )

    def fit(
        self, y_true: ArrayLike, y_quant: ArrayLike
    ) -> AdaptiveConformalCalibrator:
        """Seed the score buffers with a calibration period.

        Parameters
        ----------
        y_true : array_like
            Calibration targets, shape ``(N, H)``; NaNs ignored.
        y_quant : array_like
            Calibration forecasts, shape ``(N, H, K)``.

        Returns
        -------
        AdaptiveConformalCalibrator
            The calibrator, ready for :meth:`transform`.

        Raises
        ------
        ValueError
            If the shapes disagree or a horizon step has no finite
            calibration score.
        """
        y = np.asarray(y_true, dtype=float)
        q = np.asarray(y_quant, dtype=float)
        self._check(y, q)
        self.horizon_ = q.shape[1]
        self._seed = {}
        for i, j, _ in self.pairs:
            scores = np.maximum(q[..., i] - y, y - q[..., j])
            if not self.per_horizon:
                scores = scores.reshape(-1, 1)
            seeds = []
            for g in range(scores.shape[1]):
                col = scores[:, g]
                col = col[np.isfinite(col)]
                if col.size == 0:
                    raise ValueError(
                        f"No finite calibration score at step {g}."
                    )
                seeds.append(col.tolist())
            self._seed[(i, j)] = seeds
        return self

    def transform(
        self,
        y_true: ArrayLike,
        y_quant: ArrayLike,
        times: ArrayLike,
    ) -> NDArray[np.float64]:
        """Calibrate forecasts causally along their origins.

        Parameters
        ----------
        y_true : array_like
            Targets of the forecasts, shape ``(N, H)``. Each value
            is used only after its target time, to update the
            margins of later origins; NaNs give no update.
        y_quant : array_like
            Forecasts, shape ``(N, H, K)``, with the `H` of
            :meth:`fit`.
        times : array_like
            Integer origin time of each forecast, shape ``(N,)``,
            in steps of the series; step ``h`` (0-based) of
            forecast ``n`` targets ``times[n] + h + 1``.

        Returns
        -------
        numpy.ndarray
            Calibrated, sorted quantiles, shape ``(N, H, K)``; the
            inputs are not modified.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If the shapes disagree with each other or with
            :meth:`fit`.
        """
        if not hasattr(self, "_seed"):
            raise RuntimeError("Call fit() before transform().")
        y = np.asarray(y_true, dtype=float)
        q = np.array(y_quant, dtype=float, copy=True)
        self._check(y, q)
        t = np.asarray(times)
        n, horizon = y.shape
        if t.shape != (n,):
            raise ValueError(f"times must have shape ({n},).")
        if horizon != self.horizon_:
            raise ValueError(
                f"Fitted for horizon {self.horizon_}, got {horizon}."
            )
        groups = horizon if self.per_horizon else 1
        group_of = (
            np.arange(horizon)
            if self.per_horizon
            else np.zeros(horizon, dtype=int)
        )

        # Observation events, in order of target time.
        rows, steps = np.nonzero(np.isfinite(y))
        target = t[rows] + steps + 1
        order = np.argsort(target, kind="stable")
        rows, steps, target = rows[order], steps[order], target[order]

        alpha0 = {(i, j): 1.0 - nom for i, j, nom in self.pairs}
        alpha = {p: np.full(groups, a) for p, a in alpha0.items()}
        buffers = {
            p: [list(s) for s in seeds]
            for p, seeds in self._seed.items()
        }
        scores = {
            (i, j): np.maximum(q[..., i] - y, y - q[..., j])
            for i, j, _ in self.pairs
        }
        issued = {p: np.full((n, horizon), np.nan) for p in alpha}
        origins = np.unique(t)
        path = {p: np.empty((origins.size, groups)) for p in alpha}

        ptr = 0
        for k, now in enumerate(origins):
            # Targets observed by `now` were forecast at earlier
            # origins, so their issued margins already exist.
            while ptr < target.size and target[ptr] <= now:
                r, h = rows[ptr], steps[ptr]
                g = group_of[h]
                for p in alpha:
                    s = scores[p][r, h]
                    err = float(s > issued[p][r, h])
                    alpha[p][g] += self.gamma * (alpha0[p] - err)
                    buffers[p][g].append(s)
                ptr += 1
            here = np.flatnonzero(t == now)
            for p in alpha:
                margin = np.empty(groups)
                for g in range(groups):
                    buf = buffers[p][g]
                    if self.window is not None:
                        buf = buf[-self.window :]
                    margin[g] = _rank_select(
                        np.asarray(buf), alpha[p][g]
                    )
                issued[p][here] = margin[group_of]
                path[p][k] = alpha[p]

        for (i, j), m in issued.items():
            q[..., i] -= m
            q[..., j] += m
        self.origin_times_ = origins
        self.alpha_path_ = path
        self.margins_ = issued
        return np.sort(q, axis=-1)
