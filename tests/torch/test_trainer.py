"""Tests for the PyTorch training loop."""

import math

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

T, H = 24, 6


@pytest.fixture(scope="module")
def datasets():
    t = np.arange(400)
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": f"s{i}",
                    "ds": t,
                    "y": np.sin(2 * np.pi * t / 12 + i),
                }
            )
            for i in range(3)
        ]
    )
    panel = PanelData.from_frame(df, PanelSpec())
    panel.fit_scaling(np.full(3, 300))
    tr = panel.origins(np.zeros(3, int), np.full(3, 300), T, H, 1)
    va = panel.origins(np.full(3, 299), None, T, H, H)
    return (
        WindowDataset(panel, tr, T, H),
        WindowDataset(panel, va, T, H),
    )


def make_model(quantiles=(0.1, 0.5, 0.9)):
    torch.manual_seed(0)
    return XTFT(
        0,
        1,
        0,
        H,
        quantiles=list(quantiles),
        hidden_units=16,
        num_heads=2,
        dropout=0.0,
    )


def test_fit_reduces_validation_loss(datasets):
    train, val = datasets
    model = make_model()
    hist = fit(
        model,
        train,
        val,
        [0.1, 0.5, 0.9],
        TrainConfig(max_epochs=8, steps_per_epoch=20, lr=3e-3),
    )
    assert hist["epochs"] == 8
    assert hist["best_val_loss"] < hist["val_loss"][0]
    assert hist["best_val_loss"] == min(hist["val_loss"])


def test_best_weights_are_restored(datasets):
    train, val = datasets
    model = make_model()
    hist = fit(
        model,
        train,
        val,
        [0.1, 0.5, 0.9],
        TrainConfig(max_epochs=4, steps_per_epoch=10),
    )
    q = predict(model, val)
    y = np.stack([val[n]["y"] for n in range(len(val))])
    from xtft.nn.torch.probabilistic import pinball_loss

    loss = float(
        pinball_loss(
            torch.from_numpy(y), torch.from_numpy(q), [0.1, 0.5, 0.9]
        )
    )
    assert loss == pytest.approx(hist["best_val_loss"], rel=1e-4)


def test_early_stopping(datasets):
    train, val = datasets
    # The first epoch always improves on +inf; with a threshold no
    # later epoch can meet, training stops after 1 + patience.
    hist = fit(
        make_model(),
        train,
        val,
        [0.1, 0.5, 0.9],
        TrainConfig(
            max_epochs=50, steps_per_epoch=2, patience=2, min_delta=1e6
        ),
    )
    assert hist["epochs"] == 3
    assert hist["best_epoch"] == 0


def test_seed_makes_training_reproducible(datasets):
    train, val = datasets
    cfg = TrainConfig(max_epochs=2, steps_per_epoch=5, seed=3)
    h1 = fit(make_model([0.5]), train, val, [0.5], cfg)
    h2 = fit(make_model([0.5]), train, val, [0.5], cfg)
    assert h1["val_loss"] == h2["val_loss"]


def test_predict_shape_and_no_validation(datasets):
    train, val = datasets
    model = make_model()
    hist = fit(
        model,
        train,
        None,
        [0.1, 0.5, 0.9],
        TrainConfig(max_epochs=1, steps_per_epoch=3),
    )
    assert math.isnan(hist["best_val_loss"])
    assert predict(model, val).shape == (len(val), H, 1, 3)


def test_empty_training_set_raises(datasets):
    _, val = datasets
    empty = WindowDataset(val.panel, np.zeros((0, 2), int), T, H)
    with pytest.raises(ValueError, match="Empty"):
        fit(make_model(), empty, val, [0.1, 0.5, 0.9])
