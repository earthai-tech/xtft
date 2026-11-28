"""
Forecast a panel of series
==========================

Train XTFT on a small panel of daily series and draw a quantile
fan for one of them. The panel has the three kinds of input XTFT
fuses: a static attribute per series, an observed covariate known
only up to the forecast origin, and calendar features known in
advance.
"""

# %%
# A synthetic panel
# -----------------
# Eight series of 500 days. Each has its own level (the static
# attribute), a weekly cycle, and a response to an observed
# covariate ``temp`` that is not known in the future.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from xtft.data import (
    PanelData,
    PanelSpec,
    WindowDataset,
    add_calendar_features,
)
from xtft.evaluation import evaluate_quantile_forecast
from xtft.nn import XTFT, TrainConfig, fit, predict

rng = np.random.default_rng(0)
n_series, n_days = 8, 500
days = pd.date_range("2022-01-01", periods=n_days, freq="D")
frames = []
for i in range(n_series):
    level = rng.uniform(5.0, 15.0)
    temp = np.sin(2 * np.pi * np.arange(n_days) / 90.0)
    temp = temp + 0.3 * rng.normal(size=n_days)
    weekly = np.sin(2 * np.pi * days.day_of_week / 7.0)
    y = level + 2.0 * weekly + 1.5 * temp
    y = y + 0.5 * rng.normal(size=n_days)
    frames.append(
        pd.DataFrame(
            {
                "unique_id": f"s{i}",
                "ds": days,
                "y": y,
                "temp": temp,
                "level": level,
            }
        )
    )
df = pd.concat(frames, ignore_index=True)
df, calendar = add_calendar_features(df, features=["dayofweek"])
df.head()

# %%
# Column roles, splits and scaling
# --------------------------------
# :class:`~xtft.data.PanelSpec` names the role of each column.
# The last 100 days are the test period and the 60 days before
# them the validation period. Scaling statistics come from the
# training period only.
spec = PanelSpec(static=("level",), past=("temp",), known=calendar)
panel = PanelData.from_frame(df, spec)
val_start = panel.split_index(days[-160])
test_start = panel.split_index(days[-100])
panel.fit_scaling(val_start)

T, H = 56, 14  # history and horizon, in days
first = np.zeros(panel.n_series, dtype=np.int64)
train_idx = panel.origins(first, val_start, T, H, stride=1)
val_idx = panel.origins(val_start - 1, test_start, T, H, stride=1)
test_idx = panel.origins(test_start - 1, None, T, H, stride=H)
train_ds = WindowDataset(panel, train_idx, T, H)
val_ds = WindowDataset(panel, val_idx, T, H)
test_ds = WindowDataset(panel, test_idx, T, H)
print(len(train_ds), len(val_ds), len(test_ds))

# %%
# Model and training
# ------------------
# The past inputs of a window hold the target first, then the
# observed covariates; the known inputs cover history and horizon.
quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
torch.manual_seed(0)
model = XTFT(
    static_input_dim=1,
    dynamic_input_dim=2,  # y and temp
    future_input_dim=len(calendar),
    forecast_horizon=H,
    quantiles=quantiles,
    hidden_units=16,
    num_heads=2,
    instance_norm=True,
)
history = fit(
    model,
    train_ds,
    val_ds,
    quantiles,
    TrainConfig(max_epochs=8, batch_size=64, patience=3, seed=0),
)
print("best epoch", history["best_epoch"])

# %%
# Forecast the test period
# ------------------------
# :func:`~xtft.nn.torch.trainer.predict` returns quantiles in the
# scaled units of the panel, shape ``(N, H, O, K)``;
# :meth:`~xtft.data.PanelData.inverse_target` maps them back.
q = predict(model, test_ds)
y = np.stack([test_ds[n]["y"] for n in range(len(test_ds))])
series = test_idx[:, 0]
q = panel.inverse_target(q, series)[:, :, 0]
y = panel.inverse_target(y, series)[:, :, 0]
metrics = evaluate_quantile_forecast(y, q, quantiles)
print({k: round(metrics[k], 3) for k in ("crps_q", "coverage_80")})

# %%
# A quantile fan for one series
# -----------------------------
rows = np.flatnonzero(series == 0)
t = np.concatenate(
    [np.arange(o + 1, o + 1 + H) for o in test_idx[rows, 1]]
)
fan = q[rows].reshape(-1, len(quantiles))
obs = y[rows].ravel()
fig, ax = plt.subplots(figsize=(7, 3))
for lo, hi, alpha in ((0, 6, 0.2), (1, 5, 0.3), (2, 4, 0.4)):
    ax.fill_between(
        days[t],
        fan[:, lo],
        fan[:, hi],
        color="C0",
        alpha=alpha,
        lw=0,
    )
ax.plot(days[t], fan[:, 3], color="C0", lw=1, label="median")
ax.plot(days[t], obs, color="k", lw=0.8, label="observed")
ax.set_ylabel("y")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
