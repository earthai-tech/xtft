"""Run one (dataset, model, horizon, seed) benchmark cell.

Protocol (see ``xtft.data.panel`` for the details):

* chronological train / validation / test periods with the
  dataset's conventional validation and test lengths;
* models fitted on the train period, early stopping on the
  validation period;
* forecasts at shared rolling origins (stride ``--stride``) in the
  validation and test periods;
* split-conformal calibration (CQR, per horizon step) fitted on
  the validation forecasts and applied to the test forecasts;
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
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load  # noqa: E402
from models import RunContext, make_model  # noqa: E402

from xtft.data import PanelData, PanelSpec, WindowDataset  # noqa: E402
from xtft.evaluation import (  # noqa: E402
    ConformalQuantileCalibrator,
    evaluate_quantile_forecast,
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
        help="origin spacing in val/test (default: H)",
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
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--save-preds", action="store_true")
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
    if args.max_series:
        keep = sorted(data.df["unique_id"].unique())[: args.max_series]
        data.df = data.df[data.df["unique_id"].isin(keep)]
    spec = PanelSpec(
        static=data.static, past=data.past, known=data.known
    )
    panel = PanelData.from_frame(data.df, spec)
    lengths = np.array([len(t) for t in panel.times])
    test_start = lengths - data.test_size
    val_start = test_start - data.val_size
    if np.any(val_start <= args.input_size + args.horizon):
        raise ValueError("Train period too short for these sizes.")
    panel.fit_scaling(val_start)

    h, t_in = args.horizon, args.input_size
    stride = args.stride or h
    train_idx = panel.origins(
        np.zeros_like(val_start), val_start, t_in, h, 1
    )
    val_idx = panel.origins(val_start - 1, test_start, t_in, h, stride)
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
    y_test = _targets(panel, test_idx, t_in, h)
    scales = _seasonal_scales(panel, val_start, data.season)
    cal = ConformalQuantileCalibrator(QUANTILES).fit(y_val, q_val)
    q_test_cal = cal.transform(q_test)

    result = {
        "dataset": args.dataset,
        "model": args.model,
        "horizon": h,
        "input_size": t_in,
        "stride": stride,
        "seed": args.seed,
        "n_series": panel.n_series,
        "windows": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "quantiles": QUANTILES,
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
        },
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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{tag}.json"
    path.write_text(json.dumps(result, indent=2, default=float))
    if args.save_preds:
        np.savez_compressed(
            args.out_dir / f"{tag}_preds.npz",
            y_test=y_test,
            q_test=q_test,
            q_test_cqr=q_test_cal,
            test_index=test_idx,
        )
    m = result["metrics"]
    logger.info(
        "%s: test MAE %.4f CRPS %.4f cov80 %.3f -> CQR cov80 %.3f",
        tag,
        m["test"]["mae"],
        m["test"]["crps_q"],
        m["test"]["coverage_80"],
        m["test_cqr"]["coverage_80"],
    )
    logger.info("wrote %s", path)
    return result


if __name__ == "__main__":
    main()
