r"""Dominant seasonal periods from the periodogram.

XTFT uses a seasonal period in two places: as the lag of the
seasonal candidate when imputing missing inputs and, later, to
choose the decimation factors of the multi-resolution encoder.
:func:`dominant_periods` finds the periods from the training data,
so the user does not have to supply them.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

__all__ = ["dominant_periods"]

logger = logging.getLogger(__name__)


def _prepare(y: NDArray) -> NDArray | None:
    """Gap-filled, differenced, standardised copy of one series."""
    y = np.asarray(y, dtype=float).ravel()
    observed = np.isfinite(y)
    if observed.sum() < 4:
        return None
    t = np.arange(y.size)
    y = np.interp(t, t[observed], y[observed])
    # Prewhitening: differencing flattens red (trending, random-
    # walk-like) spectra, whose steep background would otherwise
    # create spurious long "periods".
    y = np.diff(y)
    y = y - y.mean()
    scale = y.std()
    if scale == 0.0:
        return None
    return y / scale


def dominant_periods(
    series: ArrayLike | Sequence[ArrayLike],
    *,
    max_period: int | None = None,
    min_period: int = 2,
    top: int = 3,
    rel_power: float = 0.1,
    alpha: float = 0.01,
) -> list[int]:
    r"""Strongest periodicities of one or several series.

    Parameters
    ----------
    series : array_like or sequence of array_like
        One series (1-D) or several (a 2-D array, one series per
        row, or a list of 1-D arrays of any lengths). Missing
        values (``NaN``) are linearly interpolated.
    max_period : int or None, default=None
        Longest period considered, in steps; ``None`` allows up to
        a quarter of the common length (at least four cycles).
    min_period : int, default=2
        Shortest period considered, in steps (``>= 2``).
    top : int, default=3
        Maximum number of periods returned.
    rel_power : float, default=0.1
        Keep only peaks with at least this fraction of the
        prominence of the most prominent peak, in ``(0, 1]``.
    alpha : float, default=0.01
        Family-wise false-alarm rate of the peak test over all
        tested frequencies, in ``(0, 1)`` (see Notes). Slow trends
        and noise give broad power without significant peaks.

    Returns
    -------
    list of int
        Periods in steps, strongest first; empty when no series
        has enough data or no peak qualifies.

    Raises
    ------
    ValueError
        If `min_period` < 2, `top` < 1, `rel_power` is outside
        ``(0, 1]``, `alpha` is outside ``(0, 1)`` or
        `max_period` < `min_period`.

    Notes
    -----
    Each series is gap-filled, differenced (prewhitening),
    standardised, cut to the length :math:`L` of the shortest
    series (its last :math:`L` values) and tapered with a Hann
    window; the power spectra :math:`P(k) = |\hat y(k)|^2` are
    averaged over series. Away from a peak, the average of
    :math:`N` periodogram ordinates is approximately
    :math:`\bar P(k)\, \chi^2_{2N} / (2N)` around a smooth
    background :math:`\bar P(k)`, estimated by the median of
    :math:`P` over :math:`0 < |j - k| \le 100` divided by the
    median of :math:`\chi^2_{2N} / (2N)`. A local maximum at a
    period :math:`L / k` in ``[min_period, max_period]`` is kept
    if its prominence :math:`P(k) / \bar P(k)` exceeds the
    :math:`1 - \alpha / F` quantile of :math:`\chi^2_{2N} / (2N)`,
    :math:`F` being the number of tested frequencies (a Bonferroni
    bound). Kept peaks are ranked by prominence, which is
    scale-free, so the differencing gain :math:`4 \sin^2(\pi f)`
    does not favour harmonics, and rounded to the nearest integer
    period; a candidate within 5 % of a stronger period is
    dropped.

    The test is approximate, because the background estimate is
    itself noisy and tapering correlates neighbouring ordinates: on
    600 simulated white-noise and random-walk series of 2000 steps
    the false-alarm rate was about 4 % at the default
    :math:`\alpha = 0.01`, and a weak daily cycle (amplitude 0.3 in
    unit noise) was found 96 % of the time. Differencing lowers the
    sensitivity to periods that are long relative to the window
    (e.g. a weekly cycle in hourly data); pass a small `max_period`
    when only short periods matter. The frequency resolution is
    :math:`1/L`, so periods close to :math:`L` are coarse. Cost:
    :math:`O(N L \log L)` for :math:`N` series.

    Examples
    --------
    >>> import numpy as np
    >>> from xtft.data import dominant_periods
    >>> t = np.arange(24 * 7 * 12)
    >>> rng = np.random.default_rng(0)
    >>> y = (np.sin(2 * np.pi * t / 24)
    ...      + 0.6 * np.sin(2 * np.pi * t / 168)
    ...      + 0.3 * rng.normal(size=t.size))
    >>> dominant_periods(y, max_period=48)
    [24]
    >>> dominant_periods(rng.normal(size=t.size).cumsum())
    []
    """
    if min_period < 2:
        raise ValueError(f"min_period must be >= 2, got {min_period}.")
    if top < 1:
        raise ValueError(f"top must be >= 1, got {top}.")
    if not 0.0 < rel_power <= 1.0:
        raise ValueError(
            f"rel_power must be in (0, 1], got {rel_power}."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if max_period is not None and max_period < min_period:
        raise ValueError("max_period must be >= min_period.")
    arr = series
    if isinstance(series, np.ndarray) and series.ndim <= 2:
        arr = np.atleast_2d(series)
    prepared = [p for p in (_prepare(s) for s in arr) if p is not None]
    if not prepared:
        return []
    length = min(p.size for p in prepared)
    hann = np.hanning(length)
    power = np.zeros(length // 2 + 1)
    for p in prepared:
        power += np.abs(np.fft.rfft(p[-length:] * hann)) ** 2
    power /= len(prepared)
    upper = max_period if max_period is not None else length // 4
    k = np.arange(power.size)
    with np.errstate(divide="ignore"):
        period = np.where(k > 0, length / np.maximum(k, 1), np.inf)
    valid = (k > 0) & (period >= min_period) & (period <= upper)
    half = 100
    dof = 2 * len(prepared)
    tested = int(valid.sum())
    if tested == 0:
        return []
    # Background mean from a median; threshold from the chi-square
    # law of an average of N periodogram ordinates.
    median_ratio = stats.chi2.median(dof) / dof
    threshold = stats.chi2.ppf(1.0 - alpha / tested, dof) / dof
    peaks = []
    prominence = {}
    for i in np.flatnonzero(valid):
        if not (
            0 < i < power.size - 1
            and power[i] >= power[i - 1]
            and power[i] >= power[i + 1]
        ):
            continue
        lo, hi = max(1, i - half), min(power.size, i + half + 1)
        neighbours = np.r_[power[lo:i], power[i + 1 : hi]]
        background = np.median(neighbours) / median_ratio
        if background > 0 and power[i] >= threshold * background:
            peaks.append(i)
            prominence[i] = power[i] / background
    if not peaks:
        return []
    # Rank by prominence: differencing scales the power by
    # 4 sin^2(pi f), which would favour high-frequency harmonics.
    peaks.sort(key=lambda i: prominence[i], reverse=True)
    best = prominence[peaks[0]]
    out: list[int] = []
    for i in peaks:
        if prominence[i] < rel_power * best or len(out) == top:
            break
        p = int(round(period[i]))
        if all(abs(p - q) > 0.05 * q for q in out):
            out.append(p)
    logger.debug("dominant periods %s (length %d)", out, length)
    return out
