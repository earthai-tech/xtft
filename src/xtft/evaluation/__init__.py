"""Framework-agnostic forecast evaluation and calibration.

Every model in the benchmarks, XTFT and baselines alike, is scored
and calibrated by these NumPy functions, so differences between
models cannot come from different metric code.
"""

from .conformal import ConformalQuantileCalibrator, conformal_margin
from .metrics import (
    calibration_curve,
    coverage,
    crps_from_quantiles,
    evaluate_quantile_forecast,
    interval_pairs,
    interval_score,
    mae,
    mase,
    mean_width,
    pinball,
    rmse,
    seasonal_scale,
)

__all__ = [
    "ConformalQuantileCalibrator",
    "calibration_curve",
    "conformal_margin",
    "coverage",
    "crps_from_quantiles",
    "evaluate_quantile_forecast",
    "interval_pairs",
    "interval_score",
    "mae",
    "mase",
    "mean_width",
    "pinball",
    "rmse",
    "seasonal_scale",
]
