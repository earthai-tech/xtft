"""Model adapters with one interface for the benchmark runner.

Every adapter is fitted on the train period, may use the
validation period for early stopping, and forecasts explicit
(series, origin) pairs. Forecasts are returned as quantiles of
shape ``(N, H, K)`` in the *panel scaled units* (train statistics
of each series), so all models are scored on the same scale.

Adapters
--------
``xtft``                 PyTorch XTFT (this package)
``nf:<Model>``           neuralforecast model, e.g. ``nf:TFT``,
                         ``nf:PatchTST``, ``nf:NHITS``, ``nf:DLinear``
``sf:<Model>``           statsforecast model, ``sf:SeasonalNaive``,
                         ``sf:AutoETS``, ``sf:AutoTheta``
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from xtft.data import PanelData, WindowDataset

logger = logging.getLogger(__name__)

__all__ = ["RunContext", "make_model"]


@dataclass
class RunContext:
    """Everything an adapter needs, shared by all models of a run."""

    data: object  # BenchmarkData
    panel: PanelData
    horizon: int
    input_size: int
    quantiles: tuple[float, ...]
    val_start: np.ndarray
    test_start: np.ndarray
    train_index: np.ndarray
    val_index: np.ndarray
    seed: int = 0
    max_epochs: int = 100
    max_steps: int = 2000
    patience: int = 10
    batch_size: int = 64
    steps_per_epoch: int | None = 200
    model_kw: dict = field(default_factory=dict)

    @property
    def levels(self) -> list[int]:
        """Central interval levels (percent) of the quantile set."""
        q = np.asarray(self.quantiles)
        lo = q[q < 0.5]
        return sorted(
            {int(round(100 * (1 - 2 * t))) for t in lo}, reverse=True
        )


def _to_scaled(ctx: RunContext, q: np.ndarray, idx: np.ndarray):
    """Original units -> panel scaled units, q of shape (N, H, K)."""
    if ctx.panel.loc is None:
        return q
    loc = ctx.panel.loc[idx[:, 0], 0][:, None, None]
    scale = ctx.panel.scale[idx[:, 0], 0][:, None, None]
    return (q - loc) / scale


def _quantile_columns(frame: pd.DataFrame, model: str, quantiles):
    """Return the quantile columns of `model` in `quantiles` order.

    Accepted names: ``M``, ``M-median``, ``M-lo-L``, ``M-hi-L``.
    """
    cols = {}
    for c in frame.columns:
        if not str(c).startswith(model):
            continue
        rest = str(c)[len(model) :]
        if rest in ("", "-median"):
            cols.setdefault(0.5, c)
        elif rest.startswith("-lo-"):
            cols[round((1 - float(rest[4:]) / 100) / 2, 6)] = c
        elif rest.startswith("-hi-"):
            cols[round((1 + float(rest[4:]) / 100) / 2, 6)] = c
    try:
        return [cols[round(q, 6)] for q in quantiles]
    except KeyError as exc:
        raise KeyError(
            f"{model}: missing quantile column {exc}; got "
            f"{sorted(cols)}."
        ) from exc


class _HistoryLoop:
    """Forecast origins by re-feeding the history up to each origin.

    Origins are grouped by their distance from the series end, so
    one call covers every series sharing that cutoff.
    """

    def __init__(self, ctx: RunContext) -> None:
        df = ctx.data.df.sort_values(["unique_id", "ds"])
        df = df.reset_index(drop=True)
        df["_pos"] = df.groupby("unique_id").cumcount()
        self.df = df
        self.pos = {sid: i for i, sid in enumerate(ctx.panel.ids)}
        self.ids = ctx.panel.ids

    def run(self, ctx: RunContext, index: np.ndarray, call):
        h, k = ctx.horizon, len(ctx.quantiles)
        out = np.full((len(index), h, k), np.nan)
        by_origin: dict[int, list[int]] = {}
        for row, (_, t0) in enumerate(index):
            by_origin.setdefault(int(t0), []).append(row)
        for t0, rows in sorted(by_origin.items()):
            series = [self.ids[int(index[r, 0])] for r in rows]
            hist = self.df[
                self.df["unique_id"].isin(series)
                & (self.df["_pos"] <= t0)
            ]
            fc = call(hist.drop(columns="_pos"))
            fc = fc.reset_index()
            fc = fc.sort_values(["unique_id", "ds"])
            for r, sid in zip(rows, series, strict=True):
                vals = fc[fc["unique_id"] == sid]
                if len(vals) != h:
                    raise RuntimeError(
                        f"{sid}: expected {h} forecast rows, got "
                        f"{len(vals)} at origin {t0}."
                    )
                out[r] = vals[self.cols].to_numpy(float)
        return _to_scaled(ctx, out, index)


class XTFTAdapter:
    """PyTorch XTFT trained with :func:`xtft.nn.torch.trainer.fit`."""

    def __init__(self, ctx: RunContext) -> None:
        import torch

        from xtft.nn.torch.models import XTFT

        torch.manual_seed(ctx.seed)
        kw = {
            "hidden_units": 64,
            "num_heads": 4,
            "scales": (1, 2, 4),
            "dropout": 0.1,
            # Long-horizon series drift out of the training range
            # (ETTh1's test level); the neuralforecast baselines
            # normalise every window too (scaler_type="standard"),
            # so this keeps the comparison even.
            "instance_norm": True,
        }
        kw.update(ctx.model_kw)
        kw["scales"] = tuple(kw["scales"])
        self.lr = float(kw.pop("lr", 1e-3))
        p = ctx.panel
        self.model = XTFT(
            p.static.shape[1],
            p.target[0].shape[1] + p.past[0].shape[1],
            p.known[0].shape[1],
            ctx.horizon,
            output_dim=p.target[0].shape[1],
            quantiles=ctx.quantiles,
            **kw,
        )
        self.kw = kw
        self.history: dict = {}

    def fit(self, ctx: RunContext) -> None:
        from xtft.nn.torch.trainer import TrainConfig, fit

        train = WindowDataset(
            ctx.panel, ctx.train_index, ctx.input_size, ctx.horizon
        )
        val = WindowDataset(
            ctx.panel, ctx.val_index, ctx.input_size, ctx.horizon
        )
        cfg = TrainConfig(
            max_epochs=ctx.max_epochs,
            batch_size=ctx.batch_size,
            patience=ctx.patience,
            seed=ctx.seed,
            steps_per_epoch=ctx.steps_per_epoch,
            lr=self.lr,
        )
        self.history = fit(self.model, train, val, ctx.quantiles, cfg)

    def forecast(self, ctx: RunContext, index: np.ndarray):
        from xtft.nn.torch.trainer import predict

        ds = WindowDataset(
            ctx.panel, index, ctx.input_size, ctx.horizon
        )
        q = predict(self.model, ds)
        return q[:, :, 0, :]  # single target

    def info(self) -> dict:
        return {
            "params": self.model.num_parameters(),
            "history": self.history,
            "model_kw": self.kw,
        }


class NeuralForecastAdapter(_HistoryLoop):
    """neuralforecast model with a multi-quantile loss."""

    def __init__(self, ctx: RunContext, name: str) -> None:
        super().__init__(ctx)
        import neuralforecast.models as nfm
        from neuralforecast.losses.pytorch import MQLoss

        cls = getattr(nfm, name)
        kw = {
            "h": ctx.horizon,
            "input_size": ctx.input_size,
            "loss": MQLoss(level=ctx.levels),
            "max_steps": ctx.max_steps,
            "early_stop_patience_steps": ctx.patience,
            "val_check_steps": 50,
            "batch_size": min(ctx.batch_size, 32),
            "random_seed": ctx.seed,
            "scaler_type": "standard",
            # Lightning trainer options: no progress bars in the
            # SLURM logs, no lightning_logs/ on the home disk.
            "enable_progress_bar": False,
            "logger": False,
        }
        kw.update(ctx.model_kw)
        self.name = name
        self.model = cls(**kw)
        self.cols: list[str] = []
        self.kw = {k: v for k, v in kw.items() if k != "loss"}

    def fit(self, ctx: RunContext) -> None:
        from neuralforecast import NeuralForecast

        # Train on the train period; the validation period drives
        # early stopping (val_size), the test period is withheld.
        train_val = self.df[
            self.df["_pos"]
            < self.df["unique_id"].map(
                dict(zip(ctx.panel.ids, ctx.test_start, strict=True))
            )
        ].drop(columns="_pos")
        self.nf = NeuralForecast(
            models=[self.model], freq=ctx.data.freq
        )
        val_size = int(np.min(ctx.test_start - ctx.val_start))
        start = time.perf_counter()
        self.nf.fit(df=train_val, val_size=val_size)
        self.fit_seconds = time.perf_counter() - start
        probe = self.nf.predict(df=train_val).reset_index()
        self.cols = _quantile_columns(probe, self.name, ctx.quantiles)

    def forecast(self, ctx: RunContext, index: np.ndarray):
        return self.run(ctx, index, lambda h: self.nf.predict(df=h))

    def info(self) -> dict:
        n = sum(p.numel() for p in self.model.parameters())
        return {
            "params": int(n),
            "model_kw": self.kw,
            "fit_seconds": self.fit_seconds,
        }


class StatsForecastAdapter(_HistoryLoop):
    """statsforecast model refitted at each origin (local model)."""

    def __init__(self, ctx: RunContext, name: str) -> None:
        super().__init__(ctx)
        import statsforecast.models as sfm

        season = ctx.data.season
        kw = dict(ctx.model_kw)
        self.max_history = int(kw.pop("max_history", 20 * season))
        if name in ("AutoETS", "AutoTheta", "SeasonalNaive"):
            # ETS state space is impractical beyond daily seasons
            # at sub-hourly sampling; cap and record it.
            kw.setdefault(
                "season_length",
                season if name == "SeasonalNaive" else min(season, 24),
            )
        self.name = name
        self.model = getattr(sfm, name)(**kw)
        self.kw = kw
        self.cols = _quantile_columns(
            pd.DataFrame(
                columns=[name, f"{name}-lo-{ctx.levels[0]}"]
                + [
                    f"{name}-{s}-{lv}"
                    for lv in ctx.levels
                    for s in ("lo", "hi")
                ]
            ),
            name,
            ctx.quantiles,
        )

    def fit(self, ctx: RunContext) -> None:
        """Local models are fitted at every origin in `forecast`."""

    def forecast(self, ctx: RunContext, index: np.ndarray):
        from statsforecast import StatsForecast

        sf = StatsForecast(
            models=[self.model], freq=ctx.data.freq, n_jobs=1
        )

        def call(hist: pd.DataFrame):
            hist = hist.groupby("unique_id").tail(self.max_history)
            return sf.forecast(
                df=hist, h=ctx.horizon, level=ctx.levels
            )

        return self.run(ctx, index, call)

    def info(self) -> dict:
        return {
            "params": 0,
            "model_kw": self.kw,
            "max_history": self.max_history,
        }


def make_model(spec: str, ctx: RunContext):
    """Build an adapter: ``xtft``, ``nf:<Model>`` or ``sf:<Model>``."""
    if spec == "xtft":
        return XTFTAdapter(ctx)
    family, _, name = spec.partition(":")
    if family == "nf" and name:
        return NeuralForecastAdapter(ctx, name)
    if family == "sf" and name:
        return StatsForecastAdapter(ctx, name)
    raise ValueError(
        f"Unknown model {spec!r}: use 'xtft', 'nf:<Model>' or "
        "'sf:<Model>'."
    )
