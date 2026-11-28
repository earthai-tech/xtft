"""
Compare forecasters
===================

Is one forecaster better than another, or did it win by chance?
:mod:`xtft.evaluation` scores quantile forecasts with the same
code for every model, tests two models against each other with
the Diebold-Mariano test, and ranks many models over many tasks
with the Friedman test and the Nemenyi critical difference.
"""

# %%
# Score two forecasters
# ---------------------
# Two forecasters of the same target: model A has the right
# spread, model B a slightly biased median.
import matplotlib.pyplot as plt
import numpy as np

from xtft.evaluation import (
    crps_from_quantiles,
    diebold_mariano,
    evaluate_quantile_forecast,
    friedman_nemenyi,
)

rng = np.random.default_rng(4)
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
z = np.array([-1.2816, -0.6745, 0.0, 0.6745, 1.2816])
n, H = 400, 7
mean = rng.normal(size=(n, H))
y = mean + rng.normal(size=(n, H))
q_a = mean[..., None] + z
q_b = mean[..., None] + 0.3 + z

for name, q in (("A", q_a), ("B", q_b)):
    m = evaluate_quantile_forecast(y, q, quantiles)
    print(
        f"{name}: CRPS {m['crps_q']:.3f}  MAE {m['mae']:.3f}  "
        f"80% coverage {m['coverage_80']:.2f}"
    )

# %%
# Diebold-Mariano test
# --------------------
# The test needs one loss per forecast origin, in time order. The
# windows here do not overlap, so ``horizon=1``; overlapping
# windows need the overlap (horizon divided by origin stride).
crps_a = [
    crps_from_quantiles(y[i], q_a[i], quantiles) for i in range(n)
]
crps_b = [
    crps_from_quantiles(y[i], q_b[i], quantiles) for i in range(n)
]
res = diebold_mariano(crps_a, crps_b)
print(
    f"DM statistic {res.statistic:.2f}, p-value {res.pvalue:.4f}, "
    f"mean difference {res.mean_difference:.4f}"
)

# %%
# Many models over many tasks
# ---------------------------
# Rows are tasks (data sets and horizons), columns models; lower
# scores are better. Models whose average ranks differ by less
# than the critical difference are not significantly different.
models = ["XTFT", "TFT", "LightGBM", "Naive"]
base = rng.uniform(0.5, 1.5, size=(12, 1))
scores = base * np.array([1.00, 1.04, 1.06, 1.40])
scores = scores * rng.lognormal(0.0, 0.03, size=scores.shape)
fn = friedman_nemenyi(scores)
cd = fn.critical_difference
print(
    f"Friedman p-value {fn.pvalue:.2g}, critical difference {cd:.2f}"
)

order = np.argsort(fn.average_ranks)
fig, ax = plt.subplots(figsize=(6, 1.8))
ax.scatter(fn.average_ranks[order], np.zeros(len(models)), color="C0")
for k, i in enumerate(order):
    ax.annotate(
        models[i],
        (fn.average_ranks[i], 0),
        xytext=(0, 10 if k % 2 else -16),
        textcoords="offset points",
        ha="center",
    )
best = fn.average_ranks[order[0]]
ax.plot([best, best + fn.critical_difference], [0.6, 0.6], "k-")
ax.text(best, 0.75, "critical difference", fontsize=8)
ax.set_ylim(-1, 1.2)
ax.set_yticks([])
ax.set_xlabel("average rank (1 = best)")
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
