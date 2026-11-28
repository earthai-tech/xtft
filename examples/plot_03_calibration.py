"""
Calibrate any quantile forecast
===============================

The calibration layer of :mod:`xtft.evaluation` works on the
quantile forecasts of any model. This example takes a forecaster
whose intervals are too narrow, and a test period in which the
noise grows halfway through, and compares three corrections:
split-conformal quantile regression (CQR), adaptive conformal
inference (ACI), and a band that covers whole forecast paths.
"""

# %%
# Forecasts with overconfident quantiles
# --------------------------------------
# The forecaster believes the noise has standard deviation 0.7;
# it is 1 in the validation period and rises to 1.6 in the second
# half of the test period.
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from xtft.evaluation import (
    AdaptiveConformalCalibrator,
    ConformalQuantileCalibrator,
    SimultaneousConformalBand,
    coverage,
    path_coverage,
)

rng = np.random.default_rng(1)
quantiles = [0.05, 0.1, 0.5, 0.9, 0.95]
z = norm.ppf(quantiles)
H = 7  # horizon


def simulate(n, sigma):
    """Targets and quantile forecasts of `n` forecast windows."""
    mean = rng.normal(size=(n, 1)) + np.zeros((n, H))
    y = mean + sigma[:, None] * rng.normal(size=(n, H))
    q = mean[..., None] + 0.7 * z
    return y, q


y_val, q_val = simulate(300, np.ones(300))
sigma_test = np.r_[np.ones(300), np.full(300, 1.6)]
y_test, q_test = simulate(600, sigma_test)
# One origin per window, one step apart: step h of window n
# targets time n + h + 1.
times = np.arange(600)

# %%
# Three corrections
# -----------------
# CQR widens each interval by a margin fitted once on the
# validation period. ACI starts from the same margin and updates
# it after each observed target, so it follows the change.
cqr = ConformalQuantileCalibrator(quantiles).fit(y_val, q_val)
q_cqr = cqr.transform(q_test)
aci = AdaptiveConformalCalibrator(quantiles, gamma=0.01)
q_aci = aci.fit(y_val, q_val).transform(y_test, q_test, times)

lo, hi = quantiles.index(0.1), quantiles.index(0.9)
for name, q in (("raw", q_test), ("CQR", q_cqr), ("ACI", q_aci)):
    first = coverage(y_test[:300], q[:300, :, lo], q[:300, :, hi])
    second = coverage(y_test[300:], q[300:, :, lo], q[300:, :, hi])
    print(f"{name:4s} 80% coverage: {first:.2f} then {second:.2f}")

# %%
# Coverage along the test period
# ------------------------------
# A rolling mean of the per-window coverage shows CQR falling
# below its nominal level once the noise grows, and ACI coming
# back to it.
fig, ax = plt.subplots(figsize=(7, 3))
kernel = np.ones(50) / 50
for name, q in (("raw", q_test), ("CQR", q_cqr), ("ACI", q_aci)):
    inside = (y_test >= q[..., lo]) & (y_test <= q[..., hi])
    ax.plot(np.convolve(inside.mean(1), kernel, "valid"), label=name)
ax.axhline(0.8, color="k", lw=0.8, ls="--")
ax.axvline(300, color="0.6", lw=0.8)
ax.set_xlabel("forecast origin")
ax.set_ylabel("80% coverage (rolling)")
ax.legend(frameon=False, ncol=3)
fig.tight_layout()

# %%
# Bands for whole paths
# ---------------------
# A per-step 80% interval does not contain the whole seven-step
# path 80% of the time. The simultaneous band does; its online
# version also adapts to the change in noise.
band = SimultaneousConformalBand(quantiles, coverage=0.8, base=0.8)
b_lo, b_hi = band.fit(y_val, q_val).transform_online(
    y_test, q_test, times, gamma=0.01
)
print(
    "paths inside per-step ACI interval:",
    round(path_coverage(y_test, q_aci[..., lo], q_aci[..., hi]), 2),
)
print(
    "paths inside band:", round(path_coverage(y_test, b_lo, b_hi), 2)
)
