"""Check XTFT on the GPU of this machine.

Prints the PyTorch/CUDA set-up, runs the device test suite, compares
GPU and CPU outputs, and times a training step (float32, bf16 and,
on CUDA, fp16 autocast) on a medium-sized model. Works on any CUDA
machine, e.g. a workstation, a cluster GPU node or a free Colab or
Kaggle notebook::

    pip install -e ".[torch,dev]"
    python tools/gpu_check.py

Exits with status 1 if no GPU is visible or a check fails.
"""

from __future__ import annotations

import argparse
import copy
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from xtft.nn.torch.models import XTFT
from xtft.nn.torch.probabilistic import pinball_loss

ROOT = Path(__file__).resolve().parents[1]
LEVELS = [0.05, 0.1, 0.5, 0.9, 0.95]


def _device() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return None


def _batch(device, batch=256, steps=168, horizon=24, gaps=0.1):
    g = torch.Generator().manual_seed(0)
    xs = torch.randn(batch, 4, generator=g)
    xd = torch.randn(batch, steps, 6, generator=g)
    xd[torch.rand(xd.shape, generator=g) < gaps] = float("nan")
    xf = torch.randn(batch, steps + horizon, 5, generator=g)
    y = torch.randn(batch, horizon, generator=g)
    return [t.to(device) for t in (xs, xd, xf, y)]


def _model():
    torch.manual_seed(0)
    return XTFT(
        4,
        6,
        5,
        24,
        quantiles=LEVELS,
        hidden_units=64,
        num_heads=4,
        scales=(1, 2, 4),
        instance_norm=True,
        season=24,
    )


def _time_step(model, device, amp, repeats=20):
    xs, xd, xf, y = _batch(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device.type, enabled=amp == "fp16")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(
            device.type, dtype=dtype, enabled=bool(amp)
        ):
            out = model(xs, xd, xf)
        loss = pinball_loss(y, out.float(), LEVELS)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss

    for _ in range(3):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        loss = step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1e3, float(loss)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--device",
        default=None,
        help="force a device, e.g. cpu for a dry run of this script",
    )
    args = ap.parse_args(argv)
    py, tv = platform.python_version(), torch.__version__
    print(f"python {py}, torch {tv}")
    device = torch.device(args.device) if args.device else _device()
    if device is None:
        print("No CUDA or MPS device visible: nothing to check.")
        return 1
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(
            f"CUDA {torch.version.cuda}, {props.name}, "
            f"capability {props.major}.{props.minor}, "
            f"{props.total_memory / 2**30:.1f} GiB"
        )
    ok = True

    print("\n[1] device test suite")
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-rs",
            "tests/torch/test_devices.py",
        ],
        cwd=ROOT,
    )
    ok &= res.returncode == 0

    print(
        "\n[2] GPU vs CPU outputs (with gaps, instance norm, season)"
    )
    model = _model().eval()
    xs, xd, xf, _ = _batch("cpu", batch=64)
    with torch.no_grad():
        ref = model(xs, xd, xf)
        gpu = copy.deepcopy(model).to(device)
        out = gpu(*(t.to(device) for t in (xs, xd, xf))).cpu()
    err = float((out - ref).abs().max())
    print(f"max |gpu - cpu| = {err:.2e}")
    ok &= err < 1e-3

    print("\n[3] training step time (batch 256, T=168, H=24, d=64)")
    amps = [None, "bf16"] + (["fp16"] if device.type == "cuda" else [])
    cpu_ms, _ = _time_step(
        _model().train(), torch.device("cpu"), None, 5
    )
    print(f"  cpu  fp32: {cpu_ms:8.1f} ms/step")
    for amp in amps:
        try:
            ms, loss = _time_step(
                _model().to(device).train(), device, amp
            )
            print(
                f"  {device.type} {amp or 'fp32'}: {ms:8.1f} ms/step "
                f"(x{cpu_ms / ms:.1f} vs cpu), loss {loss:.4f}"
            )
            ok &= bool(torch.isfinite(torch.tensor(loss)))
        except RuntimeError as exc:  # e.g. bf16 on an old GPU
            print(f"  {device.type} {amp}: not supported ({exc})")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
