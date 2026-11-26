"""Run one (dataset, model, horizon, seed) benchmark cell.

Protocol (see ``xtft.data.panel`` for the details):

* chronological train / validation / test periods with the
  dataset's conventional validation and test lengths;
* models fitted on the train period, early stopping on the
  validation period;
* forecasts at shared rolling origins in the validation
  (``--val-stride``, default ``H // 4``) and test (``--stride``,
  default ``H``) periods;
* split-conformal calibration (CQR, per horizon step) fitted on
  the validation forecasts and applied to the test forecasts
  (``test_cqr``), and adaptive conformal calibration seeded with
  the same validation scores and updated causally as test targets
  are observed (``test_aci``);
* validation and test forecasts saved (``_preds.npz``, float32) so
  other calibrations can be evaluated without refitting;
* all metrics in train-standardised units of each series, plus
  MASE and MSIS with the in-sample seasonal scale.

Example
-------
python benchmarks/run.py --dataset ETTh1 --model xtft --horizon 96
python benchmarks/run.py --dataset ETTh1 --model nf:PatchTST \
    --horizon 96 --seed 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import forward_fill, inject_missing, load  # noqa: E402
from models import RunContext, make_model  # noqa: E402

from xtft.data import PanelData, PanelSpec, WindowDataset  # noqa: E402
from xtft.evaluation import (  # noqa: E402
    AdaptiveConformalCalibrator,
    ConformalQuantileCalibrator,
    SimultaneousConformalBand,
    evaluate_quantile_forecast,
    interval_pairs,
    path_coverage,
)

logger = logging.getLogger("xtft.bench")

QUANTILES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def _targets(panel, index, input_size, horizon) -> np.ndarray:
    ds = WindowDataset(panel, index, input_size, horizon)
    return np.stack([ds[n]["y"][:, 0] for n in range(len(ds))])


def _seasonal_scales(panel, val_start, season) -> np.ndarray:
    """In-sample mean |y_t - y_{t-m}| per series (scaled units)."""
    out = np.ones(panel.n_series)
    for i, y in enumerate(panel.target):
        part = y[: val_start[i], 0].astype(float)
        m = season if part.size > season else 1
        d = np.abs(part[m:] - part[:-m])
        s = np.nanmean(d) if np.isfinite(d).any() else np.nan
        out[i] = s if np.isfinite(s) and s > 0 else 1.0
    return out


def _write_atomic(path: Path, write) -> None:
    """Write a file via a temporary name, fsync and rename.

    A crash or a lost write on a network file system then leaves
    either the complete file or none, never an empty or partial
    one (an empty result file was once seen on the cluster's
    Lustre scratch). Raises OSError if the file ends up empty.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    if path.stat().st_size == 0:
        raise OSError(f"{path} was written empty.")


def _window_losses(y, q, quantiles) -> dict:
    """Per-window CRPS proxy and median absolute error, (N,) each.

    Same definitions as the aggregate metrics (CRPS = twice the
    mean pinball loss), averaged over the horizon only, so paired
    tests (Diebold-Mariano) can compare models window by window.
    """
    tau = np.asarray(quantiles)
    diff = y[..., None] - q
    pin = np.maximum(tau * diff, (tau - 1.0) * diff)
    # Windows whose targets are all missing (gauge gaps) give NaN;
    # the Diebold-Mariano tests drop them.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        crps = 2.0 * np.nanmean(pin, axis=(1, 2))
        med = q[..., int(np.argmin(np.abs(tau - 0.5)))]
        ae = np.nanmean(np.abs(y - med), axis=1)
    return {"crps": crps, "ae": ae}


def band_metrics(
    y_val, q_val, y_test, q_test, q_test_cqr, times, gamma=0.005
) -> dict:
    """Whole-path coverage and width of five band constructions.

    For each nominal level: the raw and the CQR-calibrated per-step
    intervals read as a band (``raw``, ``cqr``), the simultaneous
    ``max`` and ``bonferroni`` bands fitted on the validation
    forecasts, and the ``max`` band updated causally along the
    test origins (``max_online``, step size `gamma`). Widths are
    mean band widths in the scaled units (``inf`` for an
    unbounded band).
    """
    out = {}
    pairs = {
        round(c, 6): (i, j) for i, j, c in interval_pairs(QUANTILES)
    }
    for level in (0.8, 0.9):
        i, j = pairs[level]
        res = {}
        for name, q in (("raw", q_test), ("cqr", q_test_cqr)):
            lo, hi = q[..., i], q[..., j]
            res[name] = {
                "path_coverage": path_coverage(y_test, lo, hi),
                "width": float(np.mean(hi - lo)),
            }
        for method in ("max", "bonferroni"):
            band = SimultaneousConformalBand(
                QUANTILES, level, method=method
            ).fit(y_val, q_val)
            lo, hi = band.transform(q_test)
            res[method] = {
                "path_coverage": path_coverage(y_test, lo, hi),
                "width": float(np.mean(hi - lo)),
            }
            if method == "max":
                lo, hi = band.transform_online(
                    y_test, q_test, times, gamma=gamma
                )
                res["max_online"] = {
                    "path_coverage": path_coverage(y_test, lo, hi),
                    "width": float(np.mean(hi - lo)),
                }
        out[f"{round(100 * level)}"] = res
    return out


def _metrics(y, q, quantiles, scales) -> dict:
    out = evaluate_quantile_forecast(y, q, quantiles)
    scaled = evaluate_quantile_forecast(
        y / scales[:, None], q / scales[:, None, None], quantiles
    )
    out["mase"] = scaled["mae"]
    for key, value in scaled.items():
        if key.startswith("winkler_"):
            out["msis_" + key.split("_", 1)[1]] = value
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--input-size", type=int, default=96)
    ap.add_argument(
        "--stride",
        type=int,
        default=None,
        help="origin spacing in the test period (default: H)",
    )
    ap.add_argument(
        "--val-stride",
        type=int,
        default=None,
        help=(
            "origin spacing in the validation period, which "
            "provides the CQR calibration windows (default: "
            "max(1, H // 4))"
        ),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="keep the first N series (smoke tests)",
    )
    ap.add_argument(
        "--model-kw", default="{}", help="JSON dict of model overrides"
    )
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument(
        "--no-covariates",
        action="store_true",
        help="drop static, past and known covariates (ablation)",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="suffix for the output names (model variants)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument(
        "--no-preds",
        action="store_true",
        help="do not save the validation and test forecasts",
    )
    ap.add_argument(
        "--missing-rate",
        type=float,
        default=0.0,
        help="fraction of target values removed (gap study)",
    )
    ap.add_argument(
        "--missing-mode",
        choices=("random", "blocks"),
        default="random",
    )
    ap.add_argument(
        "--ffill-all",
        action="store_true",
        help="give every model, XTFT included, forward-filled gaps",
    )
    ap.add_argument(
        "--missing-block",
        type=int,
        default=24,
        help="mean gap length in steps for --missing-mode blocks",
    )
    ap.add_argument(
        "--aci-gamma",
        type=float,
        default=0.005,
        help="step size of adaptive conformal calibration",
    )
    ap.add_argument(
        "--aci-window",
        type=int,
        default=None,
        help="recent scores kept per step by ACI (default: all)",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s"
    )

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    started = time.perf_counter()

    data = load(args.dataset)
    if args.no_covariates:
        data.static, data.past, data.known = (), (), ()
    if args.max_series:
        keep = sorted(data.df["unique_id"].unique())[: args.max_series]
        data.df = data.df[data.df["unique_id"].isin(keep)]
    spec = PanelSpec(
        static=data.static, past=data.past, known=data.known
    )
    clean = None
    if args.missing_rate > 0:
        # Models train and calibrate on the gapped series, as a
        # practitioner would; test scores use the clean values.
        # XTFT receives the gaps as NaN, the other models the
        # usual forward-filled series.
        clean = PanelData.from_frame(data.df, spec)
        gapped = inject_missing(
            data.df,
            args.missing_rate,
            mode=args.missing_mode,
            block=args.missing_block,
            seed=args.seed,
        )
        data.df = forward_fill(gapped)
        panel = PanelData.from_frame(
            data.df if args.ffill_all else gapped, spec
        )
    elif data.df["y"].isna().any():
        # Natural gaps (e.g. streamflow gauges): XTFT receives them
        # as NaN, the other models the forward-filled series, and
        # scores use the observed values only.
        panel = PanelData.from_frame(data.df, spec)
        data.df = forward_fill(data.df)
    else:
        panel = PanelData.from_frame(data.df, spec)
    lengths = np.array([len(t) for t in panel.times])
    test_start = lengths - data.test_size
    val_start = test_start - data.val_size
    if np.any(val_start <= args.input_size + args.horizon):
        raise ValueError("Train period too short for these sizes.")
    panel.fit_scaling(val_start)
    if clean is not None:
        # Clean targets in the units of the gapped panel.
        for i, y in enumerate(clean.target):
            clean.target[i] = (
                (y - panel.loc[i]) / panel.scale[i]
            ).astype(y.dtype)

    h, t_in = args.horizon, args.input_size
    stride = args.stride or h
    # Denser validation origins give CQR more (overlapping, hence
    # dependent) calibration windows: 30 at stride H on ETTh1.
    val_stride = args.val_stride or data.val_stride or max(1, h // 4)
    train_idx = panel.origins(
        np.zeros_like(val_start), val_start, t_in, h, 1
    )
    val_idx = panel.origins(
        val_start - 1, test_start, t_in, h, val_stride
    )
    test_idx = panel.origins(test_start - 1, None, t_in, h, stride)
    logger.info(
        "windows: train %d, val %d, test %d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    ctx = RunContext(
        data=data,
        panel=panel,
        horizon=h,
        input_size=t_in,
        quantiles=QUANTILES,
        val_start=val_start,
        test_start=test_start,
        train_index=train_idx,
        val_index=val_idx,
        seed=args.seed,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        patience=args.patience,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        model_kw=json.loads(args.model_kw),
    )
    model = make_model(args.model, ctx)
    t0 = time.perf_counter()
    model.fit(ctx)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    q_val = model.forecast(ctx, val_idx)
    q_test = model.forecast(ctx, test_idx)
    predict_s = time.perf_counter() - t0

    y_val = _targets(panel, val_idx, t_in, h)
    truth = panel if clean is None else clean
    y_test = _targets(truth, test_idx, t_in, h)
    scales = _seasonal_scales(panel, val_start, data.season)
    cal = ConformalQuantileCalibrator(QUANTILES).fit(y_val, q_val)
    q_test_cal = cal.transform(q_test)
    aci = AdaptiveConformalCalibrator(
        QUANTILES, gamma=args.aci_gamma, window=args.aci_window
    )
    # Origins are positions in each series; all series of a
    # benchmark share the same time axis.
    q_test_aci = aci.fit(y_val, q_val).transform(
        y_test, q_test, times=test_idx[:, 1]
    )

    result = {
        "dataset": args.dataset,
        "model": args.model,
        "tag": args.tag,
        "covariates": {
            "static": list(data.static),
            "past": list(data.past),
            "known": list(data.known),
        },
        "horizon": h,
        "input_size": t_in,
        "stride": stride,
        "val_stride": val_stride,
        "seed": args.seed,
        "n_series": panel.n_series,
        "windows": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "quantiles": QUANTILES,
        "aci": {"gamma": args.aci_gamma, "window": args.aci_window},
        "missing": {
            "rate": args.missing_rate,
            "mode": args.missing_mode,
            "block": args.missing_block,
            "ffill_all": args.ffill_all,
        },
        "metrics": {
            "val": _metrics(
                y_val, q_val, QUANTILES, scales[val_idx[:, 0]]
            ),
            "test": _metrics(
                y_test, q_test, QUANTILES, scales[test_idx[:, 0]]
            ),
            "test_cqr": _metrics(
                y_test, q_test_cal, QUANTILES, scales[test_idx[:, 0]]
            ),
            "test_aci": _metrics(
                y_test, q_test_aci, QUANTILES, scales[test_idx[:, 0]]
            ),
        },
        "bands": band_metrics(
            y_val,
            q_val,
            y_test,
            q_test,
            q_test_cal,
            times=test_idx[:, 1],
            gamma=args.aci_gamma,
        ),
        "seconds": {
            "fit": fit_s,
            "predict": predict_s,
            "total": time.perf_counter() - started,
        },
        "model_info": model.info(),
        "host": platform.node(),
        "torch": torch.__version__,
        "argv": sys.argv[1:] if argv is None else list(argv),
    }
    tag = (
        f"{args.dataset}_{args.model.replace(':', '-')}_H{h}"
        f"_T{t_in}_s{args.seed}"
    )
    if args.missing_rate > 0:
        tag += f"_miss{args.missing_mode[0]}{args.missing_rate:g}"
    if args.tag:
        tag += f"_{args.tag}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{tag}.json"
    # Arrays first, the JSON last: a result file only exists once
    # everything it refers to has been written.
    raw = _window_losses(y_test, q_test, QUANTILES)
    cal_l = _window_losses(y_test, q_test_cal, QUANTILES)
    aci_l = _window_losses(y_test, q_test_aci, QUANTILES)
    _write_atomic(
        args.out_dir / f"{tag}_losses.npz",
        lambda f: np.savez_compressed(
            f,
            test_index=test_idx,
            crps=raw["crps"],
            ae=raw["ae"],
            crps_cqr=cal_l["crps"],
            ae_cqr=cal_l["ae"],
            crps_aci=aci_l["crps"],
            ae_aci=aci_l["ae"],
        ),
    )
    if not args.no_preds:
        f32 = np.float32
        _write_atomic(
            args.out_dir / f"{tag}_preds.npz",
            lambda f: np.savez_compressed(
                f,
                val_index=val_idx,
                y_val=y_val.astype(f32),
                q_val=q_val.astype(f32),
                test_index=test_idx,
                y_test=y_test.astype(f32),
                q_test=q_test.astype(f32),
                quantiles=np.asarray(QUANTILES),
            ),
        )
    text = json.dumps(result, indent=2, default=float).encode()
    _write_atomic(path, lambda f: f.write(text))
    m = result["metrics"]
    logger.info(
        "%s: test MAE %.4f CRPS %.4f cov80 %.3f, CQR %.3f, ACI %.3f",
        tag,
        m["test"]["mae"],
        m["test"]["crps_q"],
        m["test"]["coverage_80"],
        m["test_cqr"]["coverage_80"],
        m["test_aci"]["coverage_80"],
    )
    logger.info("wrote %s", path)
    return result


if __name__ == "__main__":
    main()
