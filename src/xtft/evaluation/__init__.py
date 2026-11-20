"""Framework-agnostic forecast evaluation and calibration.

Every model in the benchmarks, XTFT and baselines alike, is scored
and calibrated by these NumPy functions, so differences between
models cannot come from different metric code.
"""

from .conformal import (
    AdaptiveConformalCalibrator,
    ConformalQuantileCalibrator,
    conformal_margin,
)
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
from .stats import (
    DieboldMarianoResult,
    FriedmanNemenyiResult,
    diebold_mariano,
    friedman_nemenyi,
)

__all__ = [
    "AdaptiveConformalCalibrator",
    "DieboldMarianoResult",
    "FriedmanNemenyiResult",
    "ConformalQuantileCalibrator",
    "calibration_curve",
    "conformal_margin",
    "coverage",
    "crps_from_quantiles",
    "diebold_mariano",
    "evaluate_quantile_forecast",
    "friedman_nemenyi",
    "interval_pairs",
    "interval_score",
    "mae",
    "mase",
    "mean_width",
    "pinball",
    "rmse",
    "seasonal_scale",
]
