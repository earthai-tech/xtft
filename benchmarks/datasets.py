"""Benchmark dataset registry.

Each loader returns a :class:`BenchmarkData` with a long-format
frame (``unique_id``, ``ds``, ``y`` plus covariates), the column
roles, and the validation and test lengths in steps. Data are
cached under ``XTFT_DATA`` (default: ``data/raw`` at the
repository root, see ``data/README.md``); on the HPC point it to
``/SCRATCH/kouakloMI/xtft/data``.

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

import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]

__all__ = ["BenchmarkData", "available", "load"]

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


def available() -> list[str]:
    """Names accepted by :func:`load`."""
    return sorted([*_LONG_HORIZON, "Synthetic"])


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
    elif name == "Synthetic":
        data = _load_synthetic(name)
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
