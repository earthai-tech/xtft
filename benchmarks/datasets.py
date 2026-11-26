"""Benchmark dataset registry.

Each loader returns a :class:`BenchmarkData` with a long-format
frame (``unique_id``, ``ds``, ``y`` plus covariates), the column
roles, and the validation and test lengths in steps. Data are
cached under ``XTFT_DATA`` (default: ``data/raw`` at the
repository root, see ``data/README.md``); on the HPC point it to
``$XTFT_SCRATCH/data``.

Long-horizon datasets come from the ``datasetsforecast`` package
(Nixtla), which distributes the ETT, ECL, Exchange, Traffic, ILI
and Weather series used by Informer, Autoformer and PatchTST, with
the conventional validation/test lengths. Two errors in its
metadata (``datasetsforecast`` 1.1.0) are corrected here: the hourly
ETTh1/ETTh2 inherit the 15-minute split sizes (11520 steps), so the
12/4/4-month split of Zhou et al. (2021) is used instead (2880
steps each), and ECL is labelled 15-minute although it is hourly,
so the frequency is inferred from the time stamps. Those series are
already
standardised by the distributor; metrics on them are therefore in
standardised units, as in the literature.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]

__all__ = [
    "BenchmarkData",
    "available",
    "forward_fill",
    "inject_missing",
    "load",
]

# Seasonal period per sampling frequency (for seasonal naive,
# ETS and the MASE/MSIS scale).
_SEASON = {
    "15min": 96,
    "h": 24,
    "H": 24,
    "10min": 144,
    "D": 7,
    "W": 52,
    "M": 12,
}

_LONG_HORIZON = {
    "ETTh1": "ETTh1",
    "ETTh2": "ETTh2",
    "ETTm1": "ETTm1",
    "ETTm2": "ETTm2",
    "ECL": "ECL",
    "Exchange": "Exchange",
    "Traffic": "TrafficL",
    "ILI": "ILI",
    "Weather": "Weather",
}

# (val, test) lengths replacing wrong datasetsforecast metadata:
# 12/4/4 months of hourly data for ETTh (Zhou et al., 2021).
_SPLIT_OVERRIDE = {
    "ETTh1": (2880, 2880),
    "ETTh2": (2880, 2880),
}


@dataclass
class BenchmarkData:
    """A loaded benchmark dataset.

    Attributes
    ----------
    name : str
        Registry name.
    df : pandas.DataFrame
        Long format: ``unique_id``, ``ds``, ``y`` and covariates.
    freq : str
        pandas frequency alias.
    season : int
        Seasonal period in steps.
    val_size, test_size : int
        Lengths, in steps, of the validation and test periods at
        the end of every series.
    horizons : tuple of int
        Horizons conventionally reported for this dataset.
    static, past, known : tuple of str
        Covariate column roles (empty for univariate benchmarks).
    val_stride : int or None
        Origin spacing in the validation period when the protocol
        fixes it (day-ahead markets: one forecast per day); ``None``
        lets ``run.py`` choose.
    """

    name: str
    df: pd.DataFrame
    freq: str
    season: int
    val_size: int
    test_size: int
    horizons: tuple[int, ...]
    static: tuple[str, ...] = ()
    past: tuple[str, ...] = ()
    known: tuple[str, ...] = ()
    val_stride: int | None = None


# Day-ahead electricity markets of the open-access benchmark of
# Lago et al. (2021), Zenodo record 4624805 (licence listed as
# "other-open"); hourly price with two day-ahead covariates.
_EPF_MARKETS = ("NP", "PJM", "BE", "FR", "DE")
_EPF_URL = "https://zenodo.org/records/4624805/files/{}.csv"
_HOURS_PER_YEAR = 24 * 364  # protocol year: 52 weeks


def _slug(name: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "_" for c in name)
    return "_".join(part for part in out.split("_") if part)


def _load_epf(name: str) -> BenchmarkData:
    """Day-ahead price of one market with its covariates.

    Protocol of the benchmark: the test period is the last two
    52-week years (17472 hours) and a forecast of the 24 hours of
    the next day is issued once per day. The year before the test
    period is used for validation here. Known covariates: the two
    day-ahead forecasts of the market (load, and wind or
    generation) plus hour, day-of-week and month encodings.
    """
    from xtft.data import add_calendar_features

    market = name.split("-", 1)[1]
    path = _data_dir() / "epf" / f"{market}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(_EPF_URL.format(market)).to_csv(path, index=False)
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    time_col, price_col, *exog = raw.columns
    exog_names = tuple(_slug(c) for c in exog)
    df = pd.DataFrame(
        {
            "unique_id": market,
            "ds": pd.to_datetime(raw[time_col]),
            "y": raw[price_col].astype(float),
        }
    )
    for src, dst in zip(exog, exog_names, strict=True):
        df[dst] = raw[src].astype(float)
    df, calendar = add_calendar_features(
        df, features=("hour", "dayofweek", "month")
    )
    return BenchmarkData(
        name=name,
        df=df,
        freq="h",
        season=168,
        val_size=_HOURS_PER_YEAR,
        test_size=2 * _HOURS_PER_YEAR,
        horizons=(24,),
        known=exog_names + tuple(calendar),
        val_stride=24,
    )


def _data_dir() -> Path:
    # Repository data/raw (a junction to E:\_DATA\xtft\raw on the
    # development laptop) unless XTFT_DATA overrides it.
    default = _REPO_ROOT / "data" / "raw"
    path = Path(os.environ.get("XTFT_DATA", default))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _infer_freq(df: pd.DataFrame, fallback: str) -> str:
    """Infer the sampling frequency from the first series."""
    first = df["unique_id"].iloc[0]
    ds = df.loc[df["unique_id"] == first, "ds"].iloc[:50]
    freq = pd.infer_freq(ds)
    if freq is None:
        return fallback
    # pandas returns e.g. "15min", "h", "D", "W-SUN", "10min".
    freq = freq.split("-")[0]
    if freq != fallback:
        logger.info("freq %s inferred (metadata: %s)", freq, fallback)
    return freq


def _load_long_horizon(name: str) -> BenchmarkData:
    try:
        from datasetsforecast.long_horizon import (
            LongHorizon,
            LongHorizonInfo,
        )
    except ImportError as exc:
        raise ImportError(
            "Long-horizon datasets need the datasetsforecast package."
        ) from exc
    group = _LONG_HORIZON[name]
    y_df, *_ = LongHorizon.load(
        directory=str(_data_dir()), group=group
    )
    info = LongHorizonInfo[group]
    y_df = y_df[["unique_id", "ds", "y"]].copy()
    y_df["ds"] = pd.to_datetime(y_df["ds"])
    y_df = y_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    freq = _infer_freq(y_df, fallback=str(info.freq))
    val_size, test_size = _SPLIT_OVERRIDE.get(
        name, (int(info.val_size), int(info.test_size))
    )
    return BenchmarkData(
        name=name,
        df=y_df,
        freq=freq,
        season=_SEASON.get(freq, 1),
        val_size=val_size,
        test_size=test_size,
        horizons=tuple(int(h) for h in info.horizons),
    )


def _load_synthetic(name: str) -> BenchmarkData:
    """Deterministic seasonal panel for smoke tests (no download).

    Ten hourly series: daily and weekly sinusoids with random phase
    and level, plus heteroscedastic noise whose scale drifts upward
    over time, so uncalibrated intervals under-cover the test
    period.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    n_series, length = 10, 2400
    t = np.arange(length)
    frames = []
    for i in range(n_series):
        phase = rng.uniform(0, 2 * np.pi)
        noise = 0.2 * (1 + t / length) * rng.standard_normal(length)
        y = (
            rng.normal(0, 2)
            + np.sin(2 * np.pi * t / 24 + phase)
            + 0.5 * np.sin(2 * np.pi * t / 168 + 2 * phase)
            + noise
        )
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": f"s{i}",
                    "ds": pd.date_range(
                        "2020-01-01", periods=length, freq="h"
                    ),
                    "y": y,
                }
            )
        )
    return BenchmarkData(
        name=name,
        df=pd.concat(frames, ignore_index=True),
        freq="h",
        season=24,
        val_size=240,
        test_size=480,
        horizons=(24, 48),
    )


_M5_EVENT_TYPES = ("Sporting", "Cultural", "National", "Religious")


def _m5_store_dept_frame(directory: Path) -> pd.DataFrame:
    """Aggregate M5 to store x department (level 9) with covariates.

    Cached as parquet after the first call (the item-level data
    have about 48 million rows).
    """
    cache = directory / "m5" / "m5_store_dept.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from datasetsforecast.m5 import M5

    y, x, s = M5.load(str(directory))
    s = s.set_index("unique_id")
    y["store_id"] = y["unique_id"].map(s["store_id"]).astype(str)
    y["dept_id"] = y["unique_id"].map(s["dept_id"]).astype(str)
    sales = (
        y.groupby(["store_id", "dept_id", "ds"], observed=True)["y"]
        .sum()
        .reset_index()
    )
    # Relative price level: each item's price over its own mean
    # price, averaged over the items of a store and department.
    price = x[["unique_id", "ds", "sell_price"]].copy()
    price["rel"] = price["sell_price"] / price.groupby("unique_id")[
        "sell_price"
    ].transform("mean")
    price["store_id"] = (
        price["unique_id"].map(s["store_id"]).astype(str)
    )
    price["dept_id"] = price["unique_id"].map(s["dept_id"]).astype(str)
    price = (
        price.groupby(["store_id", "dept_id", "ds"], observed=True)[
            "rel"
        ]
        .mean()
        .rename("price_index")
        .reset_index()
    )
    # Calendar covariates are the same for every item: take them
    # from one item per state.
    first = x.drop_duplicates("ds")[
        ["ds", "event_type_1", "event_type_2"]
    ].copy()
    events = pd.DataFrame({"ds": first["ds"]})
    for kind in _M5_EVENT_TYPES:
        hit = (first["event_type_1"].astype(str) == kind) | (
            first["event_type_2"].astype(str) == kind
        )
        events[f"event_{kind.lower()}"] = hit.astype(float).to_numpy()
    snap = x.drop_duplicates("ds")[
        ["ds", "snap_CA", "snap_TX", "snap_WI"]
    ]
    # Full daily grid: a store and department with no item rows on
    # a day (before its first listed item) sold nothing that day,
    # and every series then shares the same calendar.
    keys = sales[["store_id", "dept_id"]].drop_duplicates()
    grid = keys.merge(
        pd.DataFrame({"ds": np.sort(y["ds"].unique())}), how="cross"
    )
    sales = grid.merge(
        sales, on=["store_id", "dept_id", "ds"], how="left"
    ).fillna({"y": 0.0})
    out = sales.merge(
        price, on=["store_id", "dept_id", "ds"], how="left"
    )
    out = out.merge(events, on="ds", how="left").merge(snap, on="ds")
    state = out["store_id"].str[:2]
    out["snap"] = np.select(
        [state == "CA", state == "TX", state == "WI"],
        [out["snap_CA"], out["snap_TX"], out["snap_WI"]],
    ).astype(float)
    out = out.drop(columns=["snap_CA", "snap_TX", "snap_WI"])
    out["price_index"] = out["price_index"].fillna(1.0)
    out["unique_id"] = out["store_id"] + "_" + out["dept_id"]
    out = out.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def _load_m5(name: str) -> BenchmarkData:
    """M5 at the store x department level (70 daily series).

    Level 9 of the M5 competition (Makridakis et al., 2022): unit
    sales summed over the items of each store and department, from
    2011-01-29 to 2016-06-19 (1969 days, the evaluation period
    included). Known covariates: SNAP day of the store's state,
    event indicators by type, a relative price index and calendar
    encodings; static covariates: one-hot store, department and
    state. Test: the last 364 days (13 origins of 28 days);
    validation: the 364 days before. Data: the M5 files as
    mirrored by ``datasetsforecast``; the original data are
    distributed under the Kaggle competition rules.
    """
    from xtft.data import add_calendar_features

    raw = _m5_store_dept_frame(_data_dir())
    df = raw[
        ["unique_id", "ds", "y", "snap", "price_index"]
        + [f"event_{k.lower()}" for k in _M5_EVENT_TYPES]
    ].copy()
    df, calendar = add_calendar_features(
        df, features=("dayofweek", "dayofmonth", "month")
    )
    static = []
    for col, prefix in (
        ("store_id", "store"),
        ("dept_id", "dept"),
        ("store_id", "state"),
    ):
        values = raw[col] if prefix != "state" else raw[col].str[:2]
        for v in sorted(values.unique()):
            name_ = f"{prefix}_{v}"
            df[name_] = (values == v).astype(float).to_numpy()
            static.append(name_)
    known = (
        ("snap", "price_index")
        + tuple(f"event_{k.lower()}" for k in _M5_EVENT_TYPES)
        + tuple(calendar)
    )
    return BenchmarkData(
        name=name,
        df=df,
        freq="D",
        season=7,
        val_size=364,
        test_size=364,
        horizons=(28,),
        static=tuple(static),
        known=known,
    )


# CAMELS-US catchments from Caravan (Kratzert et al., 2023; Zenodo
# record 10968468, CC-BY-4.0): daily streamflow (mm/day) with
# ERA5-Land meteorology and static catchment attributes.
_CAMELS_START, _CAMELS_END = "1981-10-01", "2014-09-30"
_CAMELS_MET = (
    "total_precipitation_sum",
    "potential_evaporation_sum",
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "snow_depth_water_equivalent_mean",
    "surface_net_solar_radiation_mean",
    "volumetric_soil_water_layer_1_mean",
)
# Weather known in advance in the oracle scenario.
_CAMELS_FUTURE = (
    "total_precipitation_sum",
    "potential_evaporation_sum",
    "temperature_2m_mean",
)
_CAMELS_STATIC = (
    "p_mean",
    "pet_mean",
    "aridity",
    "frac_snow",
    "moisture_index",
    "seasonality",
    "high_prec_freq",
    "high_prec_dur",
    "low_prec_freq",
    "low_prec_dur",
    "log_area",
    "ele_mt_sav",
    "slp_dg_sav",
    "for_pc_sse",
    "cly_pc_sav",
    "snd_pc_sav",
)
# Basins kept: at least this share of observed streamflow days.
_CAMELS_MIN_OBSERVED = 0.9


def _caravan_root() -> Path:
    """Folder holding Caravan's ``attributes`` and ``timeseries``."""
    base = Path(
        os.environ.get("XTFT_CARAVAN", _data_dir() / "caravan")
    )
    hits = sorted(base.glob("**/attributes/camels"))
    if not hits:
        raise FileNotFoundError(
            f"Caravan CAMELS-US files not found under {base}; set "
            "XTFT_CARAVAN to the extracted Caravan folder."
        )
    return hits[0].parent.parent


def camels_attributes() -> pd.DataFrame:
    """Return static attributes and gauge locations of the basins.

    Returns
    -------
    pandas.DataFrame
        One row per basin, indexed by ``gauge_id``, with the
        Caravan climate indices, HydroATLAS attributes, ``area``
        (km^2), ``log_area``, ``gauge_lat`` and ``gauge_lon``.
    """
    root = _caravan_root() / "attributes" / "camels"
    tables = [
        pd.read_csv(root / f"attributes_{part}_camels.csv").set_index(
            "gauge_id"
        )
        for part in ("caravan", "hydroatlas", "other")
    ]
    attrs = pd.concat(tables, axis=1)
    log_area = np.log(attrs["area"].astype(float)).rename("log_area")
    return pd.concat([attrs, log_area], axis=1)


def _camels_frame() -> pd.DataFrame:
    """Long frame of all basins, cached as parquet on first use."""
    cache = _data_dir() / "camels_us.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    root = _caravan_root() / "timeseries" / "csv" / "camels"
    cols = ["date", *_CAMELS_MET, "streamflow"]
    frames = []
    for path in sorted(root.glob("camels_*.csv")):
        d = pd.read_csv(path, usecols=cols, parse_dates=["date"])
        period = (d["date"] >= _CAMELS_START) & (
            d["date"] <= _CAMELS_END
        )
        d = d[period]
        d.insert(0, "unique_id", path.stem)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"date": "ds", "streamflow": "y"})
    df.to_parquet(cache, index=False)
    return df


def _load_camels(name: str) -> BenchmarkData:
    """Daily streamflow of the CAMELS-US basins (flood forecasting).

    Daily streamflow (mm/day) of the CAMELS-US catchments in
    Caravan, water years 1982-2014 (1981-10-01 to 2014-09-30),
    for basins with at least 90% observed days. Past covariates:
    eight ERA5-Land meteorological series; static covariates:
    climate indices and catchment attributes, standardised across
    basins; known covariates: day-of-year encodings and, for
    ``CAMELS-US-Weather`` only, the observed precipitation,
    potential evaporation and temperature of the forecast days (an
    oracle-weather scenario that bounds what a perfect weather
    forecast could add). Missing streamflow stays ``NaN``.
    Test: water years 2010-2014 (1826 days); validation: water
    years 2006-2009 (1461 days).
    """
    from xtft.data import add_calendar_features

    raw = _camels_frame()
    share = raw.groupby("unique_id")["y"].apply(
        lambda s: s.notna().mean()
    )
    keep = share.index[share >= _CAMELS_MIN_OBSERVED]
    df = raw[raw["unique_id"].isin(keep)].copy()
    df, calendar = add_calendar_features(df, features=("dayofyear",))

    attrs = camels_attributes().loc[keep]
    static = [c for c in _CAMELS_STATIC if c in attrs.columns]
    z = attrs[static].astype(float)
    z = (z - z.mean()) / z.std(ddof=0).replace(0.0, 1.0)
    df = df.join(z, on="unique_id")

    known = tuple(calendar)
    if name == "CAMELS-US-Weather":
        for col in _CAMELS_FUTURE:
            df[f"{col}_future"] = df[col]
        known += tuple(f"{c}_future" for c in _CAMELS_FUTURE)
    return BenchmarkData(
        name=name,
        df=df,
        freq="D",
        season=365,
        val_size=1461,
        test_size=1826,
        horizons=(7,),
        static=tuple(static),
        past=_CAMELS_MET,
        known=known,
        val_stride=7,
    )


def available() -> list[str]:
    """Names accepted by :func:`load`."""
    epf = [f"EPF-{m}" for m in _EPF_MARKETS]
    return sorted(
        [
            *_LONG_HORIZON,
            *epf,
            "M5-StoreDept",
            "Synthetic",
            "CAMELS-US",
            "CAMELS-US-Weather",
        ]
    )


def load(name: str) -> BenchmarkData:
    """Load a benchmark dataset by registry name.

    Parameters
    ----------
    name : str
        One of :func:`available`.

    Returns
    -------
    BenchmarkData

    Raises
    ------
    KeyError
        For an unknown name.
    """
    if name in _LONG_HORIZON:
        data = _load_long_horizon(name)
    elif name.startswith("EPF-") and name[4:] in _EPF_MARKETS:
        data = _load_epf(name)
    elif name == "M5-StoreDept":
        data = _load_m5(name)
    elif name == "Synthetic":
        data = _load_synthetic(name)
    elif name in ("CAMELS-US", "CAMELS-US-Weather"):
        data = _load_camels(name)
    else:
        raise KeyError(f"Unknown dataset {name!r}; see {available()}.")
    logger.info(
        "%s: %d series, %d rows, freq %s, val %d, test %d",
        name,
        data.df["unique_id"].nunique(),
        len(data.df),
        data.freq,
        data.val_size,
        data.test_size,
    )
    return data


def inject_missing(
    df: pd.DataFrame,
    rate: float,
    *,
    mode: str = "random",
    block: int = 24,
    seed: int = 0,
    column: str = "y",
) -> pd.DataFrame:
    """Return a copy of `df` with values of `column` set to NaN.

    ``mode="random"`` drops each value independently with
    probability `rate`; ``mode="blocks"`` drops contiguous gaps of
    geometric length (mean `block` steps) until a fraction `rate`
    of each series is missing, as in sensor outages. The pattern
    depends only on `seed` and the series order, so every model of
    a benchmark cell sees the same gaps.
    """
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"rate must be in [0, 1), got {rate}.")
    if mode not in ("random", "blocks"):
        raise ValueError(f"Unknown mode {mode!r}.")
    out = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    values = out[column].to_numpy(dtype=float, copy=True)
    for _, rows in out.groupby("unique_id", sort=True).indices.items():
        n = rows.size
        if mode == "random":
            drop = rng.random(n) < rate
        else:
            drop = np.zeros(n, dtype=bool)
            while drop.mean() < rate:
                start = rng.integers(n)
                length = rng.geometric(1.0 / block)
                drop[start : start + length] = True
        values[rows[drop]] = np.nan
    out[column] = values
    return out


def forward_fill(df: pd.DataFrame, column: str = "y") -> pd.DataFrame:
    """Impute gaps by the last observation (leading gaps: next one).

    The usual preprocessing for models that cannot take missing
    values; causal except for gaps at the very start of a series.
    """
    out = df.copy()
    grouped = out.groupby("unique_id", sort=False)[column]
    out[column] = grouped.ffill()
    out[column] = out.groupby("unique_id", sort=False)[column].bfill()
    return out
