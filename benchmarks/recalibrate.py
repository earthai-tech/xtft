"""Re-calibrate saved forecasts with several conformal settings.

Reads the ``*_preds.npz`` files written by ``run.py`` (validation
and test forecasts), applies static CQR and adaptive conformal
calibration over a grid of step sizes and windows, and writes one
row per (run, method, setting) with the test metrics. No model is
refitted, so a sensitivity study costs seconds.

The grid is a sensitivity analysis: choosing the setting with the
best *test* score would leak test information. Fix the setting in
advance (``run.py`` defaults) and report the grid alongside.

Example
-------
python benchmarks/recalibrate.py results/106748 \
    --gammas 0 0.005 0.02 --windows 0 100 500 --out aci_grid.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xtft.evaluation import (
    AdaptiveConformalCalibrator,
    ConformalQuantileCalibrator,
    evaluate_quantile_forecast,
)

_KEEP = (
    "crps_q",
    "coverage_90",
    "coverage_80",
    "coverage_60",
    "width_80",
    "winkler_80",
    "calibration_error",
)


def _row(meta: dict, method: str, gamma, window, metrics) -> dict:
    return {
        **meta,
        "method": method,
        "gamma": gamma,
        "window": window,
        **{k: metrics.get(k, np.nan) for k in _KEEP},
    }


def recalibrate(path: Path, gammas, windows) -> list[dict]:
    """All calibration variants of one saved run."""
    res = json.loads(
        path.with_name(
            path.name.replace("_preds.npz", ".json")
        ).read_text()
    )
    d = np.load(path)
    tau = tuple(float(t) for t in d["quantiles"])
    y_val, q_val = d["y_val"].astype(float), d["q_val"].astype(float)
    y, q = d["y_test"].astype(float), d["q_test"].astype(float)
    times = d["test_index"][:, 1]
    meta = {k: res[k] for k in ("dataset", "model", "horizon", "seed")}
    rows = [
        _row(
            meta,
            "raw",
            None,
            None,
            evaluate_quantile_forecast(y, q, tau),
        )
    ]
    cqr = ConformalQuantileCalibrator(tau).fit(y_val, q_val)
    rows.append(
        _row(
            meta,
            "cqr",
            None,
            None,
            evaluate_quantile_forecast(y, cqr.transform(q), tau),
        )
    )
    for gamma, window in itertools.product(gammas, windows):
        aci = AdaptiveConformalCalibrator(
            tau, gamma=gamma, window=window or None
        )
        out = aci.fit(y_val, q_val).transform(y, q, times)
        rows.append(
            _row(
                meta,
                "aci",
                gamma,
                window or None,
                evaluate_quantile_forecast(y, out, tau),
            )
        )
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument(
        "--gammas", type=float, nargs="+", default=[0, 0.005, 0.02]
    )
    ap.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[0, 100, 500],
        help="score windows; 0 keeps all scores",
    )
    ap.add_argument("--out", type=Path, default=Path("aci_grid.csv"))
    args = ap.parse_args(argv)
    rows = []
    for root in args.roots:
        for path in sorted(root.rglob("*_preds.npz")):
            rows.extend(recalibrate(path, args.gammas, args.windows))
    if not rows:
        raise SystemExit(f"No *_preds.npz under {args.roots}.")
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    pd.set_option("display.width", 160)
    cols = ["dataset", "model", "method", "gamma", "window"]
    print(
        table[cols + ["coverage_80", "width_80", "crps_q"]]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
