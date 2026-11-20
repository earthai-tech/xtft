"""Compare speed and size of the PyTorch and Keras XTFT.

Times one optimisation step (forward, backward, update) and one
inference pass on random data, for several history lengths, and
reports parameter counts. Both models use the same width, heads,
horizon, quantiles and scales. The Keras model is the
fusionlab-learn XTFT (``xtft.nn.tf``), with its default extra
components (memory attention, dynamic window).

Usage
-----
python benchmarks/speed/compare_xtft.py --out results/speed.json

Timings are wall-clock medians over ``--repeats`` runs after
``--warmup`` runs, on CPU, with the thread count set by
``--threads``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("xtft.speed")

QUANTILES = [0.1, 0.5, 0.9]


def _timeit(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def make_data(cfg: dict, steps: int, seed: int = 0):
    """Random inputs and targets with the XTFT input contract."""
    rng = np.random.default_rng(seed)
    b, h = cfg["batch"], cfg["horizon"]
    xs = rng.normal(size=(b, cfg["static"])).astype("float32")
    xd = rng.normal(size=(b, steps, cfg["dynamic"])).astype("float32")
    xf = rng.normal(size=(b, steps + h, cfg["future"])).astype(
        "float32"
    )
    y = rng.normal(size=(b, h)).astype("float32")
    return xs, xd, xf, y


def bench_torch(cfg, steps, warmup, repeats):
    import torch

    from xtft.nn.torch.models import XTFT
    from xtft.nn.torch.probabilistic import pinball_loss

    torch.manual_seed(0)
    model = XTFT(
        cfg["static"],
        cfg["dynamic"],
        cfg["future"],
        cfg["horizon"],
        quantiles=QUANTILES,
        hidden_units=cfg["width"],
        num_heads=cfg["heads"],
        scales=tuple(cfg["scales"]),
        dropout=0.1,
    )
    # Fused Adam updates all tensors in one kernel (torch >= 2.4 on
    # CPU); fall back to the multi-tensor path on older releases.
    try:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, fused=True)
    except (RuntimeError, TypeError):
        opt = torch.optim.Adam(
            model.parameters(), lr=1e-3, foreach=True
        )
    xs, xd, xf, y = (
        torch.from_numpy(a) for a in make_data(cfg, steps)
    )

    def train_step():
        opt.zero_grad(set_to_none=True)
        loss = pinball_loss(y, model(xs, xd, xf), QUANTILES)
        loss.backward()
        opt.step()

    def infer():
        with torch.inference_mode():
            model(xs, xd, xf)

    model.train()
    t_train = _timeit(train_step, warmup, repeats)
    model.eval()
    t_infer = _timeit(infer, warmup, repeats)
    return {
        "params": model.num_parameters(),
        "train_step_s": t_train,
        "inference_s": t_infer,
    }


def bench_keras(cfg, steps, warmup, repeats):
    import tensorflow as tf

    from xtft.nn.tf.hybrid import XTFT

    tf.random.set_seed(0)
    model = XTFT(
        static_input_dim=cfg["static"],
        dynamic_input_dim=cfg["dynamic"],
        future_input_dim=cfg["future"],
        forecast_horizon=cfg["horizon"],
        quantiles=QUANTILES,
        hidden_units=cfg["width"],
        attention_units=cfg["width"],
        lstm_units=cfg["width"],
        embed_dim=cfg["width"] // 2,
        num_heads=cfg["heads"],
        scales=list(cfg["scales"]),
        max_window_size=steps,
        dropout_rate=0.1,
    )
    model.compile(optimizer="adam")
    xs, xd, xf, y = make_data(cfg, steps)
    # The Keras quantile loss expects targets as (B, H, O).
    y = y[..., None]
    model.train_on_batch([xs, xd, xf], y)  # build + trace

    def train_step():
        model.train_on_batch([xs, xd, xf], y)

    def infer():
        model([xs, xd, xf], training=False)

    t_train = _timeit(train_step, warmup, repeats)
    t_infer = _timeit(infer, warmup, repeats)
    return {
        "params": int(model.count_params()),
        "train_step_s": t_train,
        "inference_s": t_infer,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, nargs="+", default=[16, 64, 168]
    )
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument(
        "--threads", type=int, default=os.cpu_count() or 1
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--backends", nargs="+", default=["torch", "keras"]
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/speed.json")
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s"
    )

    cfg = {
        "batch": args.batch,
        "horizon": 24,
        "static": 5,
        "dynamic": 8,
        "future": 4,
        "width": args.width,
        "heads": 4,
        "scales": [1, 2, 4],
    }
    if "torch" in args.backends:
        import torch

        torch.set_num_threads(args.threads)
    if "keras" in args.backends:
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(
            args.threads
        )
        tf.config.threading.set_inter_op_parallelism_threads(1)

    runners = {"torch": bench_torch, "keras": bench_keras}
    rows = []
    for steps in args.steps:
        for name in args.backends:
            logger.info("backend=%s T=%d", name, steps)
            try:
                res = runners[name](
                    cfg, steps, args.warmup, args.repeats
                )
                res["status"] = "ok"
            except Exception as exc:  # report, keep benchmarking
                logger.exception("backend %s failed", name)
                res = {"status": f"error: {type(exc).__name__}: {exc}"}
            rows.append({"backend": name, "steps": steps, **res})
            logger.info("%s", rows[-1])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg,
        "threads": args.threads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "host": platform.node(),
        "results": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
