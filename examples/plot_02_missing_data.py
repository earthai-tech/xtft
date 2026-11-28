"""
Missing inputs and level shift
==============================

Real panels have gaps, and their level can drift outside the
range seen in training. XTFT takes gaps as ``NaN`` and handles
them inside the model (``missing=True``), and reversible instance
normalisation (``instance_norm=True``) standardises every window
by its own history. This example trains XTFT with and without the
instance normalisation on series that have 15% missing values and
whose level rises in the test period.
"""

# %%
# Series with gaps and a level shift
# ----------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from xtft.data import PanelData, PanelSpec, WindowDataset
from xtft.evaluation import evaluate_quantile_forecast
from xtft.nn import XTFT, TrainConfig, fit, predict

rng = np.random.default_rng(2)
n_series, n_steps = 6, 600
t = np.arange(n_steps)
frames = []
for i in range(n_series):
    x = np.sin(2 * np.pi * t / 24.0 + rng.uniform(0, 6))
    level = np.where(t >= 500, 4.0, 0.0)  # shift in the test period
    y = level + 2.0 * x + 0.3 * rng.normal(size=n_steps)
    y[rng.random(n_steps) < 0.15] = np.nan
    x[rng.random(n_steps) < 0.15] = np.nan
    frames.append(
        pd.DataFrame({"unique_id": i, "ds": t, "y": y, "x": x})
    )
df = pd.concat(frames, ignore_index=True)
print(
    f"missing share: {df[['y', 'x']].isna().mean().round(3).to_dict()}"
)

# %%
# Windows that keep the gaps
# --------------------------
# ``fill_value=None`` passes missing inputs to the model as
# ``NaN`` instead of filling them with the training mean.
spec = PanelSpec(past=("x",))
panel = PanelData.from_frame(df, spec)
val_start = panel.split_index(440)
test_start = panel.split_index(500)
panel.fit_scaling(val_start)
T, H = 48, 12
first = np.zeros(panel.n_series, dtype=np.int64)


def make(idx):
    """Windows that keep missing inputs as NaN."""
    return WindowDataset(panel, idx, T, H, fill_value=None)


train_ds = make(panel.origins(first, val_start, T, H, 1))
val_ds = make(panel.origins(val_start - 1, test_start, T, H, 1))
test_idx = panel.origins(test_start - 1, None, T, H, H)
test_ds = make(test_idx)

# %%
# Train with and without instance normalisation
# ---------------------------------------------
quantiles = [0.1, 0.5, 0.9]
cfg = TrainConfig(max_epochs=8, batch_size=64, patience=3, seed=0)
forecasts = {}
for norm in (False, True):
    torch.manual_seed(0)
    model = XTFT(
        0,
        2,
        0,
        H,
        quantiles=quantiles,
        hidden_units=16,
        num_heads=2,
        missing=True,
        instance_norm=norm,
    )
    fit(model, train_ds, val_ds, quantiles, cfg)
    forecasts[norm] = predict(model, test_ds)[:, :, 0]

y = np.stack([test_ds[n]["y"][:, 0] for n in range(len(test_ds))])
for norm, q in forecasts.items():
    m = evaluate_quantile_forecast(y, q, quantiles)
    print(
        f"instance_norm={norm!s:5s} CRPS {m['crps_q']:.3f} "
        f"80% coverage {m['coverage_80']:.2f}"
    )

# %%
# The level shift
# ---------------
# Without instance normalisation the forecast stays near the
# training level; with it, the forecast follows the new level from
# the observed history of each window.
rows = np.flatnonzero(test_idx[:, 0] == 0)
steps = np.concatenate(
    [np.arange(o + 1, o + 1 + H) for o in test_idx[rows, 1]]
)
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(steps, y[rows].ravel(), "k.", ms=3, label="observed")
for norm, color in ((False, "C3"), (True, "C0")):
    q = forecasts[norm][rows].reshape(-1, 3)
    ax.fill_between(steps, q[:, 0], q[:, 2], color=color, alpha=0.25)
    ax.plot(
        steps,
        q[:, 1],
        color=color,
        lw=1,
        label=f"instance_norm={norm}",
    )
ax.set_xlabel("step")
ax.set_ylabel("y (scaled)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
