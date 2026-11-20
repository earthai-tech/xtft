r"""Significance tests for comparing forecasters.

Two levels of comparison are covered:

* two models on one task: the Diebold-Mariano test on per-window
  losses, with the small-sample correction of Harvey, Leybourne and
  Newbold (:func:`diebold_mariano`);
* many models over many tasks: average ranks, the Friedman test and
  the Nemenyi critical difference used in critical-difference
  diagrams (:func:`friedman_nemenyi`).
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = [
    "DieboldMarianoResult",
    "FriedmanNemenyiResult",
    "diebold_mariano",
    "friedman_nemenyi",
]


class DieboldMarianoResult(NamedTuple):
    r"""Result of :func:`diebold_mariano`.

    Attributes
    ----------
    statistic : float
        Corrected DM statistic; negative when the first model has
        the lower mean loss.
    pvalue : float
        p-value from Student's t with :math:`T - 1` degrees of
        freedom.
    mean_difference : float
        Mean loss differential :math:`\bar d`.
    n : int
        Number :math:`T` of loss pairs used.
    """

    statistic: float
    pvalue: float
    mean_difference: float
    n: int


class FriedmanNemenyiResult(NamedTuple):
    r"""Result of :func:`friedman_nemenyi`.

    Attributes
    ----------
    average_ranks : numpy.ndarray
        Shape ``(k,)``, mean rank of each model (1 = best).
    statistic : float
        Friedman :math:`\chi^2_F` statistic.
    pvalue : float
        p-value of :math:`\chi^2_F` against :math:`\chi^2_{k-1}`.
    critical_difference : float
        Nemenyi critical difference of the average ranks at the
        requested level.
    n_tasks : int
        Number :math:`N` of tasks (rows) used.
    """

    average_ranks: NDArray[np.float64]
    statistic: float
    pvalue: float
    critical_difference: float
    n_tasks: int


def diebold_mariano(
    loss_a: ArrayLike,
    loss_b: ArrayLike,
    *,
    horizon: int = 1,
    alternative: str = "two-sided",
) -> DieboldMarianoResult:
    r"""Test equal predictive accuracy of two forecasters.

    Parameters
    ----------
    loss_a, loss_b : array_like
        Shape ``(T,)``, losses of models A and B on the same
        forecast targets, in time order (e.g. the CRPS of each
        rolling window, averaged over series sharing an origin).
        Pairs with a NaN in either array are dropped.
    horizon : int, default=1
        Overlap :math:`h` of consecutive forecasts: autocovariances
        up to lag :math:`h - 1` enter the variance. Use the
        horizon in steps divided by the origin stride, rounded up
        (1 for non-overlapping windows).
    alternative : {"two-sided", "less", "greater"}, default="two-sided"
        ``"less"`` tests whether A has the lower expected loss.

    Returns
    -------
    DieboldMarianoResult
        Statistic, p-value, mean differential and sample size.

    Raises
    ------
    ValueError
        If the shapes differ, fewer than 3 valid pairs remain,
        `horizon` is not a positive integer smaller than the
        sample, or `alternative` is unknown.

    See Also
    --------
    friedman_nemenyi : Comparison of many models over many tasks.

    Notes
    -----
    With :math:`d_t = L^{A}_t - L^{B}_t`, :math:`t = 1, \dots, T`,
    mean :math:`\bar d` and sample autocovariances
    :math:`\hat\gamma_k = T^{-1} \sum_{t=k+1}^{T}
    (d_t - \bar d)(d_{t-k} - \bar d)`, the statistic of Diebold
    and Mariano [1]_ is

    .. math::

        \mathrm{DM} = \bar d \Big/ \sqrt{T^{-1}\bigl(
            \hat\gamma_0 + 2 \textstyle\sum_{k=1}^{h-1}
            \hat\gamma_k\bigr)},

    and the corrected statistic of Harvey et al. [2]_ is

    .. math::

        \mathrm{DM}^{*} = \mathrm{DM} \sqrt{
            \frac{T + 1 - 2h + h(h-1)/T}{T}},

    compared with Student's t on :math:`T - 1` degrees of freedom.
    If the long-run variance estimate is not positive (possible
    for :math:`h > 1`), :math:`\hat\gamma_0` alone is used and a
    warning is logged. When every :math:`d_t` is equal the
    variance is zero; the statistic is then 0 for
    :math:`\bar d = 0` and :math:`\pm\infty` otherwise.

    The test assumes covariance-stationary loss differentials. For
    a panel, average the losses of all series per origin first, so
    that :math:`t` indexes time.

    References
    ----------
    .. [1] Diebold, F. X. and Mariano, R. S. (1995). Comparing
       predictive accuracy. *Journal of Business & Economic
       Statistics*, 13(3), 253-263.
    .. [2] Harvey, D., Leybourne, S. and Newbold, P. (1997).
       Testing the equality of prediction mean squared errors.
       *International Journal of Forecasting*, 13(2), 281-291.

    Examples
    --------
    >>> import numpy as np
    >>> from xtft.evaluation import diebold_mariano
    >>> rng = np.random.default_rng(0)
    >>> base = rng.gamma(2.0, size=200)
    >>> res = diebold_mariano(base, base + 0.3, alternative="less")
    >>> res.statistic < 0, res.pvalue < 0.01
    (True, True)
    >>> diebold_mariano(base, base).pvalue
    1.0
    """
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(
            "loss_a and loss_b must be 1-D with the same shape, "
            f"got {a.shape} and {b.shape}."
        )
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(f"Unknown alternative {alternative!r}.")
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    n = d.size
    if n < 3:
        raise ValueError(f"Need >= 3 valid loss pairs, got {n}.")
    if not isinstance(horizon, (int, np.integer)) or not (
        1 <= horizon < n
    ):
        raise ValueError(
            f"horizon must be an int in [1, {n - 1}], got {horizon!r}."
        )
    d_bar = float(d.mean())
    centred = d - d_bar
    gamma = [float(centred @ centred) / n]
    for k in range(1, horizon):
        gamma.append(float(centred[k:] @ centred[:-k]) / n)
    long_run = gamma[0] + 2.0 * sum(gamma[1:])
    if long_run <= 0.0 and gamma[0] > 0.0:
        logger.warning(
            "Non-positive long-run variance; using lag-0 variance."
        )
        long_run = gamma[0]
    if long_run <= 0.0:
        stat = 0.0 if d_bar == 0.0 else np.copysign(np.inf, d_bar)
    else:
        dm = d_bar / np.sqrt(long_run / n)
        h = horizon
        stat = float(
            dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        )
    t = stats.t(df=n - 1)
    if alternative == "two-sided":
        p = 2.0 * t.sf(abs(stat))
    elif alternative == "less":
        p = t.cdf(stat)
    else:
        p = t.sf(stat)
    return DieboldMarianoResult(
        float(stat), float(min(p, 1.0)), d_bar, int(n)
    )


def friedman_nemenyi(
    scores: ArrayLike, *, alpha: float = 0.05
) -> FriedmanNemenyiResult:
    r"""Rank models over tasks: Friedman test, Nemenyi CD.

    Parameters
    ----------
    scores : array_like
        Shape ``(N, k)``: score of each of :math:`k \ge 2` models
        (columns) on :math:`N \ge 2` tasks (rows), lower is better
        (e.g. CRPS per dataset and horizon). Rows with a NaN are
        dropped.
    alpha : float, default=0.05
        Level of the Nemenyi critical difference, in ``(0, 1)``.

    Returns
    -------
    FriedmanNemenyiResult
        Average ranks, Friedman statistic and p-value, critical
        difference and the number of tasks used.

    Raises
    ------
    ValueError
        If `scores` is not 2-D, has fewer than 2 models or fewer
        than 2 complete rows, or `alpha` is outside ``(0, 1)``.

    See Also
    --------
    diebold_mariano : Pairwise test on one task.

    Notes
    -----
    Following Demšar [1]_, models are ranked within each task (ties
    get average ranks) and :math:`R_j` is the mean rank of model
    :math:`j`. The Friedman statistic [2]_ is

    .. math::

        \chi^2_F = \frac{12 N}{k(k+1)} \Bigl(\sum_{j=1}^{k} R_j^2
            - \frac{k (k+1)^2}{4}\Bigr),

    referred to :math:`\chi^2_{k-1}` (no tie correction). Two
    models differ significantly under the Nemenyi test when their
    mean ranks differ by more than

    .. math::

        \mathrm{CD} = q_\alpha \sqrt{\frac{k(k+1)}{6N}},

    where :math:`q_\alpha` is the upper-:math:`\alpha` quantile of
    the studentized range of :math:`k` means with infinite degrees
    of freedom, divided by :math:`\sqrt 2`.

    References
    ----------
    .. [1] Demšar, J. (2006). Statistical comparisons of
       classifiers over multiple data sets. *Journal of Machine
       Learning Research*, 7, 1-30.
    .. [2] Friedman, M. (1937). The use of ranks to avoid the
       assumption of normality implicit in the analysis of
       variance. *Journal of the American Statistical
       Association*, 32(200), 675-701.

    Examples
    --------
    >>> import numpy as np
    >>> from xtft.evaluation import friedman_nemenyi
    >>> scores = np.array([[0.10, 0.20, 0.30],
    ...                    [0.12, 0.25, 0.20],
    ...                    [0.11, 0.21, 0.31],
    ...                    [0.09, 0.20, 0.29]])
    >>> res = friedman_nemenyi(scores)
    >>> res.average_ranks.tolist()
    [1.0, 2.25, 2.75]
    >>> round(res.critical_difference, 3)
    1.657
    """
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError(
            f"scores must have shape (N, k >= 2), got {x.shape}."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    complete = np.isfinite(x).all(axis=1)
    if (~complete).any():
        logger.info("Dropped %d incomplete rows.", (~complete).sum())
    x = x[complete]
    n, k = x.shape
    if n < 2:
        raise ValueError(f"Need >= 2 complete rows, got {n}.")
    ranks = stats.rankdata(x, axis=1).mean(axis=0)
    chi2 = (
        12.0
        * n
        / (k * (k + 1))
        * (float(np.sum(ranks**2)) - k * (k + 1) ** 2 / 4.0)
    )
    pvalue = float(stats.chi2.sf(chi2, k - 1))
    q = stats.studentized_range.ppf(1.0 - alpha, k, np.inf)
    cd = float(q / np.sqrt(2.0) * np.sqrt(k * (k + 1) / (6.0 * n)))
    return FriedmanNemenyiResult(ranks, float(chi2), pvalue, cd, n)
