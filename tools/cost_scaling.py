"""Measure how the cost of XTFT scales with T, H and d.

For each configuration the script builds an XTFT model and reports
the number of parameters, the forward FLOPs per window counted by
``torch.utils.flop_counter.FlopCounterMode``, the wall-clock time
of one training step (forward, backward and Adam update) on a
batch, and the inference time per window. One factor varies at a
time around a reference configuration (T = 168, H = 24, d = 64,
D = 3 past and F = 8 known variables, 11 quantiles), which matches
the electricity price benchmark.

Run (CPU, 8 threads)::

    python tools/cost_scaling.py --threads 8 --out cost_scaling.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from xtft.nn.torch.models import XTFT
from xtft.nn.torch.probabilistic import pinball_loss

QUANTILES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
REF = {"T": 168, "H": 24, "d": 64}
GRID = {
    "T": [48, 96, 168, 336, 672, 1344],
    "H": [12, 24, 96, 192, 336, 720],
    "d": [16, 32, 64, 128, 256],
}
D_PAST, F_KNOWN = 3, 8


def _batch(b: int, t: int, h: int):
    g = torch.Generator().manual_seed(0)
    xd = torch.randn(b, t, D_PAST, generator=g)
    xf = torch.randn(b, t + h, F_KNOWN, generator=g)
    y = torch.randn(b, h, generator=g)
    return xd, xf, y


def measure(
    t: int, h: int, d: int, batch: int, steps: int
) -> dict[str, float]:
    """Cost of one configuration (see module docstring)."""
    torch.manual_seed(0)
    model = XTFT(
        0,
        D_PAST,
        F_KNOWN,
        h,
        quantiles=QUANTILES,
        hidden_units=d,
        num_heads=4,
        instance_norm=True,
    )
    xd, xf, y = _batch(batch, t, h)
    model.eval()
    # FlopCounterMode does not count the fused recurrent kernels;
    # add 2 * L * G * h * (n_in + h) per recurrent layer, with G = 4
    # gates for an LSTM and 3 for a GRU (L steps, h units).
    rnn_flops = []

    def _rnn_hook(mod, inp, _out) -> None:
        gates = 4 if isinstance(mod, torch.nn.LSTM) else 3
        steps = inp[0].shape[1] if mod.batch_first else inp[0].shape[0]
        size_in = mod.input_size
        for _ in range(mod.num_layers):
            rnn_flops.append(
                2
                * steps
                * gates
                * mod.hidden_size
                * (size_in + mod.hidden_size)
            )
            size_in = mod.hidden_size

    hooks = [
        m.register_forward_hook(_rnn_hook)
        for m in model.modules()
        if isinstance(m, (torch.nn.LSTM, torch.nn.GRU))
    ]
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(None, xd[:1], xf[:1])
    for hk in hooks:
        hk.remove()
    flops = fc.get_total_flops() + sum(rnn_flops)

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step() -> None:
        opt.zero_grad()
        q = model(None, xd, xf)
        pinball_loss(y, q, QUANTILES).backward()
        opt.step()

    step()  # warm-up
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    train_s = (time.perf_counter() - t0) / steps

    model.eval()
    xdi, xfi, _ = _batch(4 * batch, t, h)
    with torch.no_grad():
        model(None, xdi, xfi)
        t0 = time.perf_counter()
        for _ in range(3):
            model(None, xdi, xfi)
        infer_s = (time.perf_counter() - t0) / 3 / (4 * batch)
    return {
        "params": model.num_parameters(),
        "gflops_per_window": flops / 1e9,
        "train_step_s": train_s,
        "infer_ms_per_window": 1e3 * infer_s,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("cost.csv"))
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    rows = []
    for factor, values in GRID.items():
        for v in values:
            cfg = {**REF, factor: v}
            m = measure(
                cfg["T"], cfg["H"], cfg["d"], args.batch, args.steps
            )
            rows.append({"factor": factor, **cfg, **m})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
