"""Device and precision tests (CPU always; CUDA and MPS if present).

Run on a GPU machine with ``pytest tests/torch/test_devices.py`` or
``python tools/gpu_check.py``.
"""

import copy

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from xtft.data import PanelData, PanelSpec, WindowDataset  # noqa: E402
from xtft.nn.torch.models import XTFT  # noqa: E402
from xtft.nn.torch.trainer import (  # noqa: E402
    TrainConfig,
    fit,
    predict,
)

LEVELS = [0.1, 0.5, 0.9]
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if (
    getattr(torch.backends, "mps", None)
    and torch.backends.mps.is_available()
):
    DEVICES.append("mps")
ACCELERATORS = [d for d in DEVICES if d != "cpu"]


def make_model(**kw):
    torch.manual_seed(0)
    return XTFT(
        2,
        3,
        2,
        6,
        quantiles=LEVELS,
        hidden_units=16,
        num_heads=2,
        **kw,
    )


def make_inputs(gaps=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    xs = torch.randn(5, 2, generator=g)
    xd = torch.randn(5, 30, 3, generator=g)
    xf = torch.randn(5, 36, 2, generator=g)
    if gaps:
        drop = torch.rand(xd.shape, generator=g) < gaps
        xd = torch.where(drop, torch.full_like(xd, float("nan")), xd)
    return xs, xd, xf


def panel_windows(n=300, gaps=0.0):
    rng = np.random.default_rng(0)
    t = np.arange(n)
    y = np.sin(2 * np.pi * t / 12) + 0.1 * rng.normal(size=n)
    y[rng.random(n) < gaps] = np.nan
    df = pd.DataFrame({"unique_id": "a", "ds": t, "y": y})
    panel = PanelData.from_frame(df, PanelSpec())
    panel.fit_scaling(np.array([200]))
    train = panel.origins(np.array([0]), np.array([200]), 24, 6, 1)
    test = panel.origins(np.array([199]), None, 24, 6, 6)
    return (
        WindowDataset(panel, train, 24, 6, fill_value=None),
        WindowDataset(panel, test, 24, 6, fill_value=None),
    )


@pytest.mark.parametrize("device", ACCELERATORS or ["skip"])
@pytest.mark.parametrize(
    "kw",
    [{}, {"instance_norm": True, "season": 6}],
    ids=["plain", "norm+season"],
)
def test_accelerator_matches_cpu(device, kw):
    if device == "skip":
        pytest.skip("no CUDA or MPS device")
    model = make_model(**kw).eval()
    inputs = make_inputs(gaps=0.2)
    with torch.no_grad():
        ref = model(*inputs)
        moved = copy.deepcopy(model).to(device)
        out = moved(*(x.to(device) for x in inputs)).cpu()
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_fit_and_predict_on_device(device):
    train, test = panel_windows(gaps=0.1)
    model = make_model_for_panel()
    cfg = TrainConfig(
        max_epochs=2, steps_per_epoch=5, batch_size=16, device=device
    )
    hist = fit(model, train, test, LEVELS, cfg)
    assert np.isfinite(hist["train_loss"]).all()
    assert next(model.parameters()).device.type == device
    q = predict(model, test, device=device)
    assert q.shape == (len(test), 6, 1, 3) and np.isfinite(q).all()


def make_model_for_panel(**kw):
    torch.manual_seed(0)
    return XTFT(
        0,
        1,
        0,
        6,
        quantiles=LEVELS,
        hidden_units=16,
        num_heads=2,
        **kw,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_bf16_autocast_trains(device):
    if device == "mps":
        pytest.skip("bf16 autocast support on MPS varies")
    train, test = panel_windows()
    model = make_model_for_panel()
    cfg = TrainConfig(
        max_epochs=2,
        steps_per_epoch=5,
        batch_size=16,
        device=device,
        amp="bf16",
    )
    hist = fit(model, train, test, LEVELS, cfg)
    assert np.isfinite(hist["train_loss"]).all()
    # Parameters stay float32 under autocast.
    assert next(model.parameters()).dtype == torch.float32
    q = predict(model, test, device=device, amp="bf16")
    assert q.dtype == np.float32 and np.isfinite(q).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fp16_with_grad_scaling_on_cuda():
    train, test = panel_windows(gaps=0.1)
    model = make_model_for_panel()
    cfg = TrainConfig(
        max_epochs=2,
        steps_per_epoch=5,
        batch_size=16,
        device="cuda",
        amp="fp16",
    )
    hist = fit(model, train, test, LEVELS, cfg)
    assert np.isfinite(hist["train_loss"]).all()


def test_amp_validation():
    train, test = panel_windows()
    model = make_model_for_panel()
    with pytest.raises(ValueError, match="fp16"):
        fit(
            model,
            train,
            test,
            LEVELS,
            TrainConfig(device="cpu", amp="fp16"),
        )
    with pytest.raises(ValueError, match="amp"):
        fit(
            model,
            train,
            test,
            LEVELS,
            TrainConfig(device="cpu", amp="fp8"),
        )
