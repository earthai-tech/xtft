"""Panel (multi-series) data, chronological splits and windows.

The benchmark protocol rests on three rules implemented here:

1. **Chronological splits.** Each series is cut at two global
   timestamps into train, validation and test periods; nothing is
   shuffled across time.
2. **Shared rolling origins.** Forecast origins are spaced by a
   fixed stride and aligned to the *end* of the data, which is how
   ``neuralforecast`` and ``statsforecast`` place their
   cross-validation cutoffs, so every model forecasts from the same
   origins. An origin belongs to the validation set when its whole
   target window lies in the validation period, to the test set
   when it lies in the test period; windows that straddle the
   boundary are dropped.
3. **Train-only statistics.** Scaling statistics are computed on
   the train period of each series.

Windows are produced lazily (:class:`WindowDataset`), so long,
wide panels never materialise every window in memory.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "PanelData",
    "PanelSpec",
    "WindowDataset",
    "add_calendar_features",
    "rolling_origins",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PanelSpec:
    r"""Column roles of a long-format panel.

    Parameters
    ----------
    id_col : str, default="unique_id"
        Series identifier.
    time_col : str, default="ds"
        Timestamp (datetime or integer step).
    target : tuple of str, default=("y",)
        Target column(s), :math:`O \ge 1`.
    static : tuple of str, default=()
        Columns constant within a series (numeric).
    past : tuple of str, default=()
        Observed covariates, known only up to the forecast origin.
        Past targets are always added to the past inputs.
    known : tuple of str, default=()
        Covariates known over the forecast horizon (calendar
        features, planned values).

    Examples
    --------
    >>> from xtft.data import PanelSpec
    >>> PanelSpec(known=("hour_sin",)).target
    ('y',)
    """

    id_col: str = "unique_id"
    time_col: str = "ds"
    target: tuple[str, ...] = ("y",)
    static: tuple[str, ...] = ()
    past: tuple[str, ...] = ()
    known: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("target", "static", "past", "known"):
            value = getattr(self, name)
            if isinstance(value, str):
                object.__setattr__(self, name, (value,))
        if not self.target:
            raise ValueError("At least one target column is required.")
        cols = [*self.target, *self.static, *self.past, *self.known]
        if len(set(cols)) != len(cols):
            raise ValueError(f"Column roles overlap: {cols}.")


_CALENDAR = {
    "hour": (lambda t: t.hour, 24),
    "dayofweek": (lambda t: t.day_of_week, 7),
    "dayofmonth": (lambda t: t.day - 1, 31),
    "dayofyear": (lambda t: t.day_of_year - 1, 366),
    "month": (lambda t: t.month - 1, 12),
    "minute": (lambda t: t.minute, 60),
}


def add_calendar_features(
    df: pd.DataFrame,
    time_col: str = "ds",
    features: Sequence[str] = ("hour", "dayofweek", "month"),
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    r"""Add cyclical calendar encodings as known covariates.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel with a datetime column `time_col`.
    time_col : str, default="ds"
        Timestamp column.
    features : sequence of str
        Any of ``"hour"``, ``"dayofweek"``, ``"dayofmonth"``,
        ``"dayofyear"``, ``"month"``, ``"minute"``.

    Returns
    -------
    frame : pandas.DataFrame
        Copy of `df` with columns ``<f>_sin`` and ``<f>_cos``.
    names : tuple of str
        The added column names.

    Raises
    ------
    ValueError
        For an unknown feature name.

    Notes
    -----
    A feature :math:`c \in \{0, \dots, P-1\}` with period
    :math:`P` is encoded as
    :math:`(\sin 2\pi c/P, \cos 2\pi c/P)`, so that its last and
    first values are neighbours.

    Examples
    --------
    >>> import pandas as pd
    >>> from xtft.data import add_calendar_features
    >>> df = pd.DataFrame({"ds": pd.date_range("2024", periods=3,
    ...                                       freq="h")})
    >>> out, names = add_calendar_features(df, features=["hour"])
    >>> names
    ('hour_sin', 'hour_cos')
    """
    out = df.copy()
    t = pd.DatetimeIndex(pd.to_datetime(out[time_col]))
    names = []
    for f in features:
        if f not in _CALENDAR:
            raise ValueError(
                f"Unknown calendar feature {f!r}; choose from "
                f"{sorted(_CALENDAR)}."
            )
        getter, period = _CALENDAR[f]
        angle = 2 * np.pi * np.asarray(getter(t), dtype=float) / period
        out[f"{f}_sin"] = np.sin(angle)
        out[f"{f}_cos"] = np.cos(angle)
        names += [f"{f}_sin", f"{f}_cos"]
    return out, tuple(names)


def rolling_origins(
    n_steps: int, first: int, horizon: int, stride: int
) -> NDArray[np.int64]:
    r"""Forecast origins aligned to the end of a series.

    Parameters
    ----------
    n_steps : int
        Series length :math:`L`.
    first : int
        Smallest admissible origin index.
    horizon : int
        Forecast horizon :math:`H \ge 1`.
    stride : int
        Spacing :math:`r \ge 1` between origins.

    Returns
    -------
    numpy.ndarray
        Increasing origins :math:`t_0 = L - 1 - H - k r \ge`
        `first`, :math:`k = 0, 1, \dots`; the origin is the index of
        the last observed step, and its targets are
        :math:`t_0 + 1, \dots, t_0 + H`.

    Examples
    --------
    >>> from xtft.data import rolling_origins
    >>> rolling_origins(20, first=10, horizon=3, stride=2).tolist()
    [10, 12, 14, 16]
    """
    if horizon < 1 or stride < 1:
        raise ValueError("horizon and stride must be >= 1.")
    last = n_steps - 1 - horizon
    if last < first:
        return np.zeros(0, dtype=np.int64)
    k = (last - first) // stride
    return last - stride * np.arange(k, -1, -1, dtype=np.int64)


class PanelData:
    """Per-series arrays of a regular panel, with train scaling.

    Use :meth:`from_frame`. Each series must be regularly sampled
    (equal steps) and sorted; gaps must be filled upstream with
    NaN rows, which the loss and metrics then skip for targets.

    Attributes
    ----------
    ids : list
        Series identifiers, in order.
    times : list of numpy.ndarray
        Timestamps per series.
    target, past, known : list of numpy.ndarray
        Float32 arrays per series, shapes ``(L_i, O)``,
        ``(L_i, D)``, ``(L_i, F)``, in scaled units after
        :meth:`fit_scaling`.
    static : numpy.ndarray
        Shape ``(n_series, S)``.
    """

    def __init__(self, spec: PanelSpec) -> None:
        self.spec = spec
        self.ids: list = []
        self.times: list[NDArray] = []
        self.target: list[NDArray[np.float32]] = []
        self.past: list[NDArray[np.float32]] = []
        self.known: list[NDArray[np.float32]] = []
        self.static = np.zeros((0, len(spec.static)), np.float32)
        self.loc: NDArray[np.float64] | None = None
        self.scale: NDArray[np.float64] | None = None

    @classmethod
    def from_frame(
        cls, df: pd.DataFrame, spec: PanelSpec
    ) -> PanelData:
        """Build the panel from a long-format frame.

        Parameters
        ----------
        df : pandas.DataFrame
            One row per (series, timestamp).
        spec : PanelSpec
            Column roles.

        Returns
        -------
        PanelData

        Raises
        ------
        KeyError
            If a column named in `spec` is missing.
        ValueError
            If a series has duplicated timestamps, irregular steps
            or non-constant static values.
        """
        cols = [
            spec.id_col,
            spec.time_col,
            *spec.target,
            *spec.static,
            *spec.past,
            *spec.known,
        ]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}.")
        panel = cls(spec)
        statics = []
        frame = df.sort_values([spec.id_col, spec.time_col])
        for sid, g in frame.groupby(spec.id_col, sort=False):
            t = g[spec.time_col].to_numpy()
            if len(t) > 1:
                steps = np.diff(t)
                if np.any(steps == steps[0] * 0):
                    raise ValueError(
                        f"Series {sid!r}: duplicated time."
                    )
                if np.any(steps != steps[0]):
                    raise ValueError(
                        f"Series {sid!r}: irregular time steps; fill "
                        "gaps with NaN rows first."
                    )
            panel.ids.append(sid)
            panel.times.append(t)
            # Own, writable copies: pandas copy-on-write may hand out
            # read-only views, and scaling works in place.
            for store, names in (
                (panel.target, spec.target),
                (panel.past, spec.past),
                (panel.known, spec.known),
            ):
                store.append(
                    np.array(
                        g[list(names)].to_numpy(np.float32), copy=True
                    )
                )
            if spec.static:
                s = g[list(spec.static)]
                if (s.nunique(dropna=False) > 1).any():
                    raise ValueError(
                        f"Series {sid!r}: static columns vary in time."
                    )
                statics.append(s.iloc[0].to_numpy(np.float32))
        n = len(panel.ids)
        panel.static = (
            np.stack(statics)
            if statics
            else np.zeros((n, 0), np.float32)
        )
        logger.info(
            "Panel: %d series, lengths %d-%d",
            n,
            min(map(len, panel.times)),
            max(map(len, panel.times)),
        )
        return panel

    @property
    def n_series(self) -> int:
        """Number of series."""
        return len(self.ids)

    def split_index(self, cutoff) -> NDArray[np.int64]:
        """Per-series index of the first step at or after `cutoff`."""
        return np.array(
            [
                int(np.searchsorted(t, cutoff, side="left"))
                for t in self.times
            ],
            dtype=np.int64,
        )

    def fit_scaling(self, train_end: NDArray[np.int64]) -> None:
        r"""Standardise targets and covariates with train statistics.

        Parameters
        ----------
        train_end : numpy.ndarray
            Per-series exclusive end of the train period (see
            :meth:`split_index`).

        Notes
        -----
        Each series :math:`i` and column :math:`c` is transformed as
        :math:`(x - \mu_{ic}) / \sigma_{ic}` with mean and standard
        deviation over the train period (NaNs skipped;
        :math:`\sigma < 10^{-8}` replaced by 1). Static columns are
        standardised across series. The target statistics are kept
        in :attr:`loc` and :attr:`scale` for
        :meth:`inverse_target`. Calling it twice raises, to avoid
        double scaling.

        Raises
        ------
        RuntimeError
            If the panel is already scaled.
        ValueError
            If a train period is empty.
        """
        if self.loc is not None:
            raise RuntimeError("Panel is already scaled.")
        if np.any(train_end < 2):
            raise ValueError("Every series needs >= 2 train steps.")
        locs, scales = [], []
        for i in range(self.n_series):
            stats = []
            for arr in (self.target[i], self.past[i], self.known[i]):
                if arr.shape[1] == 0:
                    stats.append(None)
                    continue
                part = arr[: train_end[i]].astype(np.float64)
                mu = np.nan_to_num(np.nanmean(part, axis=0))
                sd = np.nan_to_num(np.nanstd(part, axis=0), nan=1.0)
                sd = np.where(sd < 1e-8, 1.0, sd)
                arr -= mu.astype(np.float32)
                arr /= sd.astype(np.float32)
                stats.append((mu, sd))
            locs.append(stats[0][0])
            scales.append(stats[0][1])
        self.loc = np.stack(locs)
        self.scale = np.stack(scales)
        if self.static.shape[1]:
            mu = np.nanmean(self.static, axis=0)
            sd = np.nanstd(self.static, axis=0)
            self.static = (
                (self.static - mu) / np.where(sd < 1e-8, 1.0, sd)
            ).astype(np.float32)

    def inverse_target(
        self, values: NDArray, series_index: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """Map scaled target values back to original units.

        Parameters
        ----------
        values : numpy.ndarray
            Shape ``(N, H, O)`` or ``(N, H, O, K)``.
        series_index : numpy.ndarray
            Shape ``(N,)``, series of each row.

        Returns
        -------
        numpy.ndarray
            Same shape, original units.
        """
        if self.loc is None:
            return np.asarray(values, dtype=float)
        loc = self.loc[series_index][:, None, :]
        scale = self.scale[series_index][:, None, :]
        values = np.asarray(values, dtype=float)
        if values.ndim == 4:
            loc, scale = loc[..., None], scale[..., None]
        return values * scale + loc

    def origins(
        self,
        first: NDArray[np.int64],
        end: NDArray[np.int64] | None,
        input_size: int,
        horizon: int,
        stride: int,
    ) -> NDArray[np.int64]:
        r"""Origins whose targets fall in ``[first + 1, end)``.

        Parameters
        ----------
        first : numpy.ndarray
            Per-series smallest origin.
        end : numpy.ndarray or None
            Per-series exclusive end of the target period; ``None``
            means the series end.
        input_size : int
            History length :math:`T`; origins need
            :math:`t_0 + 1 \ge T`.
        horizon, stride : int
            See :func:`rolling_origins`.

        Returns
        -------
        numpy.ndarray
            Shape ``(N, 2)``: series index and origin, ordered by
            series then time.
        """
        rows = []
        for i, t in enumerate(self.times):
            n = len(t) if end is None else int(end[i])
            lo = max(int(first[i]), input_size - 1)
            o = rolling_origins(n, lo, horizon, stride)
            rows.append(np.column_stack([np.full(o.size, i), o]))
        if not rows:
            return np.zeros((0, 2), np.int64)
        return np.concatenate(rows).astype(np.int64)


class WindowDataset:
    r"""Lazy forecasting windows over a :class:`PanelData`.

    Parameters
    ----------
    panel : PanelData
        Source panel (already scaled if scaling is wanted).
    index : numpy.ndarray
        Shape ``(N, 2)``: series index and origin per window, from
        :meth:`PanelData.origins`.
    input_size : int
        History length :math:`T`.
    horizon : int
        Forecast horizon :math:`H`.
    fill_value : float or None, default=0.0
        Value replacing missing past and known inputs; ``None``
        keeps them as ``NaN`` for a model that handles missing
        values itself (XTFT with ``missing=True``).

    Notes
    -----
    Item ``n`` with series :math:`i` and origin :math:`t_0` is a
    dict of float32 arrays: ``static`` ``(S,)``; ``past``
    ``(T, O + D)`` holding targets and observed covariates at
    :math:`t_0 - T + 1, \dots, t_0`; ``future`` ``(T + H, F)``
    holding known covariates up to :math:`t_0 + H`; ``y``
    ``(H, O)``. Missing past and known inputs are replaced by
    `fill_value` (0 is the train mean after scaling) unless it is
    ``None``; missing future targets always stay NaN for the loss
    to skip.
    The item works with ``torch.utils.data.DataLoader`` and its
    default collate function.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from xtft.data import PanelData, PanelSpec, WindowDataset
    >>> df = pd.DataFrame({"unique_id": "a", "ds": np.arange(30),
    ...                    "y": np.arange(30.0)})
    >>> panel = PanelData.from_frame(df, PanelSpec())
    >>> idx = panel.origins(np.array([0]), None, 8, 4, 5)
    >>> ds = WindowDataset(panel, idx, input_size=8, horizon=4)
    >>> item = ds[len(ds) - 1]
    >>> item["past"][:, 0].tolist()[-1], item["y"][:, 0].tolist()
    (25.0, [26.0, 27.0, 28.0, 29.0])
    """

    def __init__(
        self,
        panel: PanelData,
        index: NDArray[np.int64],
        input_size: int,
        horizon: int,
        fill_value: float | None = 0.0,
    ) -> None:
        if input_size < 1 or horizon < 1:
            raise ValueError("input_size and horizon must be >= 1.")
        self.fill_value = fill_value
        self.panel = panel
        self.index = np.asarray(index, dtype=np.int64).reshape(-1, 2)
        self.input_size = input_size
        self.horizon = horizon
        for i, t0 in self.index:
            if t0 + 1 < input_size or t0 + horizon >= len(
                panel.times[i]
            ):
                raise ValueError(
                    f"Origin {t0} of series {i} leaves the window "
                    "outside the series."
                )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, n: int) -> dict[str, NDArray[np.float32]]:
        i, t0 = (int(v) for v in self.index[n])
        p = self.panel
        lo, hi = t0 + 1 - self.input_size, t0 + 1
        past = np.concatenate(
            [p.target[i][lo:hi], p.past[i][lo:hi]], axis=1
        )
        future = p.known[i][lo : hi + self.horizon]
        if self.fill_value is not None:
            past = np.nan_to_num(past, nan=self.fill_value)
            future = np.nan_to_num(future, nan=self.fill_value)
        return {
            "static": p.static[i],
            "past": past,
            "future": future,
            "y": p.target[i][hi : hi + self.horizon],
        }
