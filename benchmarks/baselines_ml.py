"""Machine-learning and EPF-specific baselines.

``lgb``   LightGBM, one global gradient-boosting model per quantile,
          direct multi-horizon (the forecast step is a feature).
``lear``  LEAR, the LASSO-estimated autoregressive model of the
          open EPF benchmark (Lago et al., 2021), re-implemented
          from the paper and recalibrated every day.

Both adapters follow the interface of ``models.py``: ``fit``,
``forecast`` returning quantiles ``(N, H, K)`` in the panel's scaled
units, and ``info``.

References
----------
Lago, J., Marcjasz, G., De Schutter, B. and Weron, R. (2021).
Forecasting day-ahead electricity prices: A review of
state-of-the-art algorithms, best practices and an open-access
benchmark. Applied Energy, 293, 116983.

Uniejewski, B., Weron, R. and Ziel, F. (2018). Variance stabilizing
transformations for electricity spot price forecasting. IEEE
Transactions on Power Systems, 33(2), 2219-2229.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def _last_finite(window: np.ndarray) -> float:
    """Most recent finite value of a 1-D window (0 if none)."""
    finite = np.flatnonzero(np.isfinite(window))
    return float(window[finite[-1]]) if finite.size else 0.0


class LightGBMAdapter:
    """Global LightGBM quantile regressors on lag features.

    For a window of series ``i`` with origin ``t0`` and step ``h``
    the target is ``y[t0 + h] - y_last`` (``y_last`` the most recent
    observed value), so level shifts move the anchor rather than
    the trees' leaves. Features: lags relative to ``y_last``
    (1-24 and multiples of the season within the input window),
    the level ``y_last``, window mean (relative) and standard
    deviation, known covariates at ``t0 + h``, static covariates,
    the series index (categorical) and ``h``. One model per
    quantile level, early-stopped on the validation windows;
    quantiles are sorted after prediction. All values are in the
    panel's scaled units, and NaN inputs are handled by LightGBM.

    Model options (``--model-kw``): any native LightGBM parameter,
    plus ``num_boost_round`` (default 2000, early-stopped),
    ``max_rows`` (training rows, default 400000) and
    ``max_val_rows`` (default 100000).
    """

    def __init__(self, ctx) -> None:
        # Native LightGBM parameter names (the scikit-learn wrapper
        # of LightGBM 3.x is incompatible with scikit-learn >= 1.6).
        kw = {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "feature_fraction": 0.8,
            "verbose": -1,
        }
        kw.update(ctx.model_kw)
        self.rounds = int(kw.pop("num_boost_round", 2000))
        self.max_rows = int(kw.pop("max_rows", 400_000))
        self.max_val_rows = int(kw.pop("max_val_rows", 100_000))
        self.kw = kw
        t = ctx.input_size
        season = ctx.data.season
        lags = set(range(2, min(t, 24) + 1))
        if season <= t:
            lags |= {season * k for k in range(1, t // season + 1)}
        lags.add(t)
        self.lags = np.array(sorted(lags))  # lag l: value at t0+1-l
        self.models: list = []

    def _features(self, ctx, index: np.ndarray):
        """Feature matrix and anchors for every (window, step)."""
        p, h, t = ctx.panel, ctx.horizon, ctx.input_size
        n = len(index)
        rows = []
        anchors = np.empty(n)
        for k, (i, t0) in enumerate(index):
            y = p.target[i][:, 0]
            window = y[t0 + 1 - t : t0 + 1]
            last = _last_finite(window)
            anchors[k] = last
            lagged = y[t0 + 1 - self.lags] - last
            base = np.concatenate(
                [
                    [
                        last,
                        np.nanmean(window) - last,
                        np.nanstd(window),
                    ],
                    lagged,
                    p.static[i],
                    [i],
                ]
            )
            known = p.known[i][t0 + 1 : t0 + 1 + h]
            steps = np.arange(1, h + 1)[:, None]
            rows.append(
                np.hstack([np.tile(base, (h, 1)), known, steps])
            )
        x = np.vstack(rows).astype(np.float32) if rows else None
        return x, anchors

    def _targets(self, ctx, index, anchors) -> np.ndarray:
        p, h = ctx.panel, ctx.horizon
        out = np.empty((len(index), h))
        for k, (i, t0) in enumerate(index):
            out[k] = p.target[i][t0 + 1 : t0 + 1 + h, 0] - anchors[k]
        return out.ravel()

    def _sample(self, index, cap, rng):
        """At most ``cap`` rows: a seeded subset of the windows."""
        n_win = max(1, cap // self._h)
        if len(index) <= n_win:
            return index
        return index[
            np.sort(rng.choice(len(index), n_win, replace=False))
        ]

    def fit(self, ctx) -> None:
        """Train one early-stopped model per quantile level."""
        import lightgbm as lgb

        self._h = ctx.horizon
        rng = np.random.default_rng(ctx.seed)
        train = self._sample(ctx.train_index, self.max_rows, rng)
        val = self._sample(ctx.val_index, self.max_val_rows, rng)
        x, a = self._features(ctx, train)
        y = self._targets(ctx, train, a)
        xv, av = self._features(ctx, val)
        yv = self._targets(ctx, val, av)
        keep, keep_v = np.isfinite(y), np.isfinite(yv)
        x, y, xv, yv = x[keep], y[keep], xv[keep_v], yv[keep_v]
        cat = [3 + len(self.lags) + ctx.panel.static.shape[1]]
        start = time.perf_counter()
        self.models, self.trees = [], []
        train_set = lgb.Dataset(
            x, y, categorical_feature=cat, free_raw_data=False
        )
        val_set = lgb.Dataset(
            xv, yv, categorical_feature=cat, reference=train_set
        )
        for tau in ctx.quantiles:
            params = {
                "objective": "quantile",
                "alpha": tau,
                "metric": "quantile",
                "seed": ctx.seed,
                **self.kw,
            }
            model = lgb.train(
                params,
                train_set,
                num_boost_round=self.rounds,
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            self.models.append(model)
            self.trees.append(int(model.best_iteration or 0))
        self.fit_seconds = time.perf_counter() - start
        logger.info(
            "LightGBM: %d rows, %d features, trees %s",
            len(y),
            x.shape[1],
            self.trees,
        )

    def forecast(self, ctx, index: np.ndarray) -> np.ndarray:
        """Sorted quantiles ``(N, H, K)`` in scaled units."""
        x, anchors = self._features(ctx, index)
        h = ctx.horizon
        q = np.stack(
            [
                m.predict(x, num_iteration=m.best_iteration).reshape(
                    len(index), h
                )
                for m in self.models
            ],
            axis=-1,
        )
        q = q + anchors[:, None, None]
        return np.sort(q, axis=-1)

    def info(self) -> dict:
        """Run metadata for the result file."""
        return {
            "params": 0,
            "trees": self.trees,
            "lags": self.lags.tolist(),
            "model_kw": self.kw,
            "num_boost_round": self.rounds,
            "max_rows": self.max_rows,
            "fit_seconds": self.fit_seconds,
        }


class LEARAdapter:
    """LEAR for hourly day-ahead markets, recalibrated every day.

    For a forecast day ``D`` and hour ``h`` the model is a LASSO
    regression of the price ``P[D, h]`` on prices of days D-1, D-2,
    D-3, D-7 (all 24 hours), each exogenous series on days D, D-1
    and D-7 (all 24 hours) and seven weekday dummies (247 features
    with two exogenous series), as in Lago et al. (2021). Inputs
    and target are scaled by their median and median absolute
    deviation over the calibration window and passed through
    ``asinh`` (variance stabilisation, Uniejewski et al., 2018);
    the penalty is chosen by AIC along the LARS path, then the
    LASSO is refitted at that penalty. The model is recalibrated
    for every forecast day on the preceding ``calibration_window``
    days (default 728); ``recalibrate_every`` (days) changes the
    schedule, 0 calibrating once at the first day of each forecast
    period (to compare with models trained once).

    LEAR gives point forecasts. Quantiles are the point plus
    ``z_tau * s_h``, with ``s_h`` the in-sample residual standard
    deviation of hour ``h`` in original units, a Gaussian spread
    that the conformal layer (CQR, ACI) then recalibrates; compare
    LEAR on the calibrated splits.

    Requires an hourly series starting at 00:00 with whole days,
    horizon 24 and origins at 23:00 (the EPF protocol).
    """

    def __init__(self, ctx) -> None:
        if ctx.horizon != 24 or ctx.data.freq != "h":
            raise ValueError("LEAR needs hourly data and horizon 24.")
        kw = dict(ctx.model_kw)
        self.window = int(kw.pop("calibration_window", 728))
        # 1: daily (benchmark protocol); k: every k days; 0: once at
        # the first day of each forecast period (no recalibration).
        self.every = int(kw.pop("recalibrate_every", 1))
        if self.every < 0:
            raise ValueError("recalibrate_every must be >= 0.")
        self.exog = [
            c
            for c in ctx.data.known
            if not c.endswith(("_sin", "_cos"))
        ]
        df = ctx.data.df.sort_values(["unique_id", "ds"])
        self.days = {}
        for sid, g in df.groupby("unique_id", sort=False):
            if len(g) % 24 or g["ds"].iloc[0].hour != 0:
                raise ValueError(f"{sid}: not whole days from 00:00.")
            n = len(g) // 24
            self.days[sid] = {
                "price": g["y"].to_numpy(float).reshape(n, 24),
                "exog": [
                    g[c].to_numpy(float).reshape(n, 24)
                    for c in self.exog
                ],
                "weekday": g["ds"].dt.day_of_week.to_numpy()[::24],
            }
        self.ids = list(ctx.panel.ids)
        self.fits = 0
        self.seconds = 0.0

    def _design(self, d: dict, days: np.ndarray) -> np.ndarray:
        p = d["price"]
        parts = [p[days - lag] for lag in (1, 2, 3, 7)]
        for x in d["exog"]:
            parts += [x[days - lag] for lag in (0, 1, 7)]
        wd = np.zeros((days.size, 7))
        wd[np.arange(days.size), d["weekday"][days]] = 1.0
        parts.append(wd)
        return np.hstack(parts)

    @staticmethod
    def _scaler(a: np.ndarray):
        med = np.median(a, axis=0)
        mad = stats.median_abs_deviation(a, axis=0, scale="normal")
        mad = np.where(mad > 1e-12, mad, 1.0)
        return med, mad

    def _calibrate(self, d: dict, day: int) -> dict:
        """Fit the 24 hourly LASSO models on days before `day`."""
        from sklearn.linear_model import Lasso, LassoLarsIC

        first = max(7, day - self.window)
        train = np.arange(first, day)
        x_raw = self._design(d, train)
        dummies = slice(x_raw.shape[1] - 7, None)
        xm, xs = self._scaler(x_raw)
        xm[dummies], xs[dummies] = 0.0, 1.0
        x = np.arcsinh((x_raw - xm) / xs)
        x[:, dummies] = x_raw[:, dummies]
        hours, spread = [], np.empty(24)
        for hour in range(24):
            target = d["price"][train, hour]
            ym, ys = self._scaler(target[:, None])
            z = np.arcsinh((target - ym[0]) / ys[0])
            alpha = LassoLarsIC(criterion="aic").fit(x, z).alpha_
            lasso = Lasso(alpha=max(alpha, 1e-6), max_iter=2500)
            lasso.fit(x, z)
            fitted = np.sinh(lasso.predict(x)) * ys[0] + ym[0]
            spread[hour] = np.std(target - fitted)
            hours.append((lasso, ym[0], ys[0]))
            self.fits += 1
        return {
            "xm": xm,
            "xs": xs,
            "dummies": dummies,
            "hours": hours,
            "spread": spread,
        }

    def _predict(self, state: dict, d: dict, day: int) -> np.ndarray:
        """Point forecast of the 24 hours of `day`."""
        x_raw = self._design(d, np.array([day]))
        x = np.arcsinh((x_raw - state["xm"]) / state["xs"])
        x[:, state["dummies"]] = x_raw[:, state["dummies"]]
        return np.array(
            [
                np.sinh(lasso.predict(x)[0]) * ys + ym
                for lasso, ym, ys in state["hours"]
            ]
        )

    def _predict_day(self, d: dict, day: int):
        """Calibrate on the days before `day` and forecast it."""
        state = self._calibrate(d, day)
        return self._predict(state, d, day), state["spread"]

    def fit(self, ctx) -> None:
        """Recalibrated for every forecast day in `forecast`."""

    def forecast(self, ctx, index: np.ndarray) -> np.ndarray:
        """Recalibrate and forecast each origin's next day."""
        from models import _to_scaled

        z = stats.norm.ppf(np.asarray(ctx.quantiles))
        out = np.empty((len(index), 24, z.size))
        start = time.perf_counter()
        if np.any((index[:, 1] + 1) % 24):
            raise ValueError("LEAR origins must be at 23:00.")
        # Walk each series' forecast days in time order and
        # recalibrate on the schedule (daily, every k days, or once
        # at the first day of this forecast period).
        order = np.lexsort((index[:, 1], index[:, 0]))
        state, first_day, current = None, None, None
        for k in order:
            i, t0 = index[k]
            day = (int(t0) + 1) // 24
            if current != i:
                current, state, first_day = i, None, day
            d = self.days[self.ids[int(i)]]
            due = state is None or (
                self.every > 0 and (day - first_day) % self.every == 0
            )
            if due:
                state = self._calibrate(d, day)
            point = self._predict(state, d, day)
            out[k] = point[:, None] + state["spread"][:, None] * z
        self.seconds += time.perf_counter() - start
        return _to_scaled(ctx, out, index)

    def info(self) -> dict:
        """Run metadata for the result file."""
        return {
            "params": 0,
            "calibration_window": self.window,
            "recalibrate_every": self.every,
            "exogenous": self.exog,
            "lasso_fits": self.fits,
            "seconds": self.seconds,
        }
