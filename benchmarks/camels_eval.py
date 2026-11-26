"""Hydrological and flood-warning evaluation of the CAMELS-US runs.

Reads the forecasts saved by ``run.py`` (``*_preds.npz``) for the
CAMELS-US study, calibrates each run with adaptive conformal
inference (as in the main tables), and computes per basin:

* the Nash-Sutcliffe efficiency (NSE) of the median forecast over
  the test period [1]_;
* the CRPS, the per-step coverage of the 80% interval and the
  coverage of the whole-week 90% band (normalised max-score band,
  updated online);
* flood-warning skill. A flood occurs in a test week when the
  observed flow exceeds the basin's 95th percentile of
  validation-period flow on any day. A warning is issued when
  (a) the median, (b) the upper bound of the per-step 90%
  interval, or (c) the upper bound of the whole-week 90% band
  exceeds that threshold on any day. Skill is summarised by the
  probability of detection (POD), the false-alarm ratio (FAR) and
  the critical success index (CSI).

Forecasts are in the per-basin standardised units of the panel;
NSE, the thresholds and the warnings are invariant to that affine
scaling, so no back-transformation is needed.

Example
-------
python benchmarks/camels_eval.py /SCRATCH/.../results/107736 \
    --out camels_eval

References
----------
.. [1] Nash, J. E. and Sutcliffe, J. V. (1970). River flow
   forecasting through conceptual models part I: A discussion of
   principles. Journal of Hydrology, 10(3), 282-290.
   https://doi.org/10.1016/0022-1694(70)90255-6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets  # noqa: E402

from xtft.data import PanelData, PanelSpec  # noqa: E402
from xtft.evaluation import (  # noqa: E402
    AdaptiveConformalCalibrator,
    SimultaneousConformalBand,
)

logger = logging.getLogger("camels_eval")
GAMMA = 0.005  # ACI step size, as in run.py


def basin_ids(name: str) -> list[str]:
    """Series order of the panel built by ``run.py`` for `name`."""
    data = datasets.load(name)
    spec = PanelSpec(
        static=data.static, past=data.past, known=data.known
    )
    return list(PanelData.from_frame(data.df, spec).ids)


def nse(y: np.ndarray, f: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency over the observed entries."""
    ok = np.isfinite(y) & np.isfinite(f)
    if ok.sum() < 2:
        return np.nan
    y, f = y[ok], f[ok]
    denom = np.sum((y - y.mean()) ** 2)
    return (
        float(1.0 - np.sum((y - f) ** 2) / denom) if denom else np.nan
    )


def warning_skill(event: np.ndarray, warn: np.ndarray) -> dict:
    """POD, FAR and CSI of binary warnings for binary events."""
    hits = int(np.sum(event & warn))
    misses = int(np.sum(event & ~warn))
    false = int(np.sum(~event & warn))
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false,
        "pod": hits / (hits + misses) if hits + misses else np.nan,
        "far": false / (hits + false) if hits + false else np.nan,
        "csi": (
            hits / (hits + misses + false)
            if hits + misses + false
            else np.nan
        ),
    }


def weekly_crps(y, q, tau) -> np.ndarray:
    """CRPS of each forecast week; NaN when no flow is observed."""
    diff = y[..., None] - q
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return 2.0 * np.nanmean(
            np.maximum(tau * diff, (tau - 1.0) * diff), axis=(1, 2)
        )


def evaluate_run(
    path: Path, ids: list[str]
) -> tuple[list[dict], pd.DataFrame]:
    """Per-basin metrics and weekly CRPS of one saved run.

    Calibration is done separately for each basin: adaptive
    conformal inference and the whole-week band are fitted on the
    basin's own validation weeks and updated along its test weeks.
    Pooling all basins would apply one update per basin and day at
    every origin, multiplying the effective step size by the number
    of basins and making the intervals over-cover.
    """
    z = np.load(path)
    levels = [float(t) for t in z["quantiles"]]
    tau = np.asarray(levels)
    lo80, hi80 = levels.index(0.1), levels.index(0.9)
    hi90 = levels.index(0.95)
    med = int(np.argmin(np.abs(tau - 0.5)))
    y_val, q_val = z["y_val"], z["q_val"]
    y_all, q_all = z["y_test"], z["q_test"]
    series, times = z["test_index"][:, 0], z["test_index"][:, 1]
    val_series = z["val_index"][:, 0]

    rows, weekly = [], []
    for s in np.unique(series):
        m, mv = series == s, val_series == s
        order = np.argsort(times[m])
        t = times[m][order]
        y, q = y_all[m][order], q_all[m][order]
        yv, qv = y_val[mv], q_val[mv]
        qa = (
            AdaptiveConformalCalibrator(levels, gamma=GAMMA)
            .fit(yv, qv)
            .transform(y, q, t)
        )
        band = SimultaneousConformalBand(
            levels, coverage=0.9, base=0.9
        )
        b_lo, b_hi = band.fit(yv, qv).transform_online(
            y, q, t, gamma=GAMMA
        )
        observed = np.isfinite(y)
        has_obs = observed.any(axis=1)
        in80 = (y >= qa[..., lo80]) & (y <= qa[..., hi80])
        path_ok = ((y >= b_lo) & (y <= b_hi)) | ~observed
        crps_raw = weekly_crps(y, q, tau)
        crps_aci = weekly_crps(y, qa, tau)

        thr = np.nanpercentile(yv, 95)
        event = (np.nan_to_num(y, nan=-np.inf) > thr).any(axis=1)
        rules = {
            "median": (qa[..., med] > thr).any(axis=1),
            "step90": (qa[..., hi90] > thr).any(axis=1),
            "band90": (b_hi > thr).any(axis=1),
        }
        basin = ids[int(s)]
        row = {
            "basin": basin,
            "nse": nse(y.ravel(), qa[..., med].ravel()),
            "crps_raw": float(np.nanmean(crps_raw)),
            "crps": float(np.nanmean(crps_aci)),
            "cov80_step": float(in80[observed].mean()),
            "cov90_path": float(path_ok[has_obs].all(axis=1).mean()),
            "n_weeks": int(has_obs.sum()),
            "n_floods": int(event[has_obs].sum()),
        }
        for rule, warn in rules.items():
            sk = warning_skill(event[has_obs], warn[has_obs])
            for k, v in sk.items():
                row[f"{rule}_{k}"] = v
        rows.append(row)
        weekly.append(
            pd.DataFrame({"basin": basin, "time": t, "crps": crps_aci})
        )
    return rows, pd.concat(weekly, ignore_index=True)


def summarise(basins: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (dataset, model, seed); warnings pooled."""
    keys = ["dataset", "model", "seed"]
    g = basins.groupby(keys)
    out = g.agg(
        nse_median=("nse", "median"),
        nse_mean=("nse", "mean"),
        crps_raw=("crps_raw", "mean"),
        crps=("crps", "mean"),
        cov80_step=("cov80_step", "mean"),
        cov90_path=("cov90_path", "mean"),
        n_floods=("n_floods", "sum"),
    )
    for rule in ("median", "step90", "band90"):
        h = g[f"{rule}_hits"].sum()
        mi = g[f"{rule}_misses"].sum()
        fa = g[f"{rule}_false_alarms"].sum()
        out[f"{rule}_pod"] = h / (h + mi)
        out[f"{rule}_far"] = fa / (h + fa)
        out[f"{rule}_csi"] = h / (h + mi + fa)
    return out.reset_index()


def main(argv=None) -> None:
    """Evaluate every CAMELS run under the given result folders."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("camels_eval"))
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    ids_cache: dict[str, list[str]] = {}
    rows, weekly = [], []
    for root in args.roots:
        for jpath in sorted(root.glob("CAMELS-US*.json")):
            res = json.loads(jpath.read_text())
            ppath = jpath.with_name(jpath.stem + "_preds.npz")
            if not ppath.exists():
                logger.warning("no forecasts for %s", jpath.name)
                continue
            name = res["dataset"]
            if name not in ids_cache:
                ids_cache[name] = basin_ids(name)
            label = res["model"] + (
                f"/{res['tag']}" if res.get("tag") else ""
            )
            logger.info("%s", jpath.name)
            run = {
                "dataset": name,
                "model": label,
                "seed": res["seed"],
            }
            per_basin, week = evaluate_run(ppath, ids_cache[name])
            rows += [{**run, **r} for r in per_basin]
            weekly.append(week.assign(**run))
    basins = pd.DataFrame(rows)
    # Weekly CRPS per basin for the Diebold-Mariano tests.
    pd.concat(weekly, ignore_index=True).to_parquet(
        args.out / "camels_weekly.parquet", index=False
    )
    attrs = datasets.camels_attributes()[["gauge_lat", "gauge_lon"]]
    basins = basins.join(attrs, on="basin")
    basins.to_csv(args.out / "camels_basins.csv", index=False)
    summary = summarise(basins)
    summary.to_csv(args.out / "camels_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
