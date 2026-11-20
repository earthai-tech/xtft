"""Smoke tests: the ported Keras models build and run forward.

These tests guard the port from fusionlab-learn (import rewiring and
the Keras 3 fixes). They check output shapes only; numerical
behaviour is tested against the PyTorch implementation later.
"""

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from xtft.backends import use_backend  # noqa: E402

BATCH, PAST, HORIZON = 8, 10, 3
QUANTILES = [0.1, 0.5, 0.9]
DIMS = {
    "static_input_dim": 4,
    "dynamic_input_dim": 5,
    "future_input_dim": 2,
}


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(0)
    x_static = rng.normal(size=(BATCH, 4)).astype("float32")
    x_dynamic = rng.normal(size=(BATCH, PAST, 5)).astype("float32")
    x_future = rng.normal(size=(BATCH, PAST + HORIZON, 2)).astype(
        "float32"
    )
    return [x_static, x_dynamic, x_future]


@pytest.mark.parametrize("name", ["XTFT", "SuperXTFT"])
def test_xtft_family_forward(name, inputs):
    import xtft.nn

    with use_backend("tensorflow"):
        cls = getattr(xtft.nn, name)
    assert cls.__module__.startswith("xtft.nn.tf.")
    model = cls(
        **DIMS,
        forecast_horizon=HORIZON,
        quantiles=QUANTILES,
        scales=[1, 2],
        max_window_size=PAST,
    )
    out = model(inputs)
    assert tuple(out.shape) == (BATCH, HORIZON, len(QUANTILES))
    assert np.all(np.isfinite(np.asarray(out)))


def test_tft_forward(inputs):
    import xtft.nn

    with use_backend("tensorflow"):
        cls = xtft.nn.TFT
    model = cls(**DIMS, forecast_horizon=HORIZON, quantiles=QUANTILES)
    out = model(inputs)
    assert tuple(out.shape) == (BATCH, HORIZON, len(QUANTILES))
