r"""Split-conformal calibration of quantile forecasts (CQR).

Quantile regressors, XTFT included, are fitted by minimising the
pinball loss, which gives no finite-sample guarantee that an
interval :math:`[\\hat q_{\tau}, \\hat q_{1-\tau}]` covers
:math:`1 - 2\tau` of new observations. Conformalized quantile
regression (CQR) [1]_ corrects each interval with a margin
estimated on held-out data, restoring the nominal coverage under
exchangeability.

References
----------
.. [1] Romano, Y., Patterson, E. and Candes, E. (2019). Conformalized
   quantile regression. Advances in Neural Information Processing
   Systems 32 (NeurIPS 2019).
.. [2] Chernozhukov, V., Fernandez-Val, I. and Galichon, A. (2010).
   Quantile and probability curves without crossing.
   Econometrica, 78(3), 1093-1125.
   https://doi.org/10.3982/ECTA7880
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .metrics import _levels, interval_pairs

__all__ = ["ConformalQuantileCalibrator", "conformal_margin"]

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
