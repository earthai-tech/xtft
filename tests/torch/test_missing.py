"""Tests for native missing-value handling in XTFT."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from xtft.data import PanelData, PanelSpec, WindowDataset  # noqa: E402
from xtft.nn.torch.layers import (  # noqa: E402
    MissingValueDecay,
    VariableSelection,
    observation_features,
)
from xtft.nn.torch.models import XTFT  # noqa: E402
from xtft.nn.torch.probabilistic import pinball_loss  # noqa: E402

NAN = float("nan")
LEVELS = [0.1, 0.5, 0.9]


def with_gaps(x, rate, seed=0):
    g = torch.Generator().manual_seed(seed)
    drop = torch.rand(x.shape, generator=g) < rate
    return torch.where(drop, torch.full_like(x, NAN), x)


def test_observation_features_match_a_loop():
    x = with_gaps(torch.randn(3, 20, 4), 0.4)
    m, last, delta = observation_features(x)
    for b in range(3):
        for v in range(4):
            prev, since = 0.0, None
            for t in range(20):
                val = x[b, t, v].item()
                if np.isfinite(val):
                    prev, since = val, 0
                else:
                    since = t + 1 if since is None else since + 1
                assert m[b, t, v] == float(np.isfinite(val))
                assert last[b, t, v].item() == pytest.approx(prev)
                assert delta[b, t, v].item() == since


def test_decay_keeps_observed_values_and_is_finite():
    x = with_gaps(torch.randn(2, 30, 3), 0.5, seed=1)
    layer = MissingValueDecay(3)
    x_hat, mask, _ = layer(x)
    observed = mask.bool()
    assert torch.equal(x_hat[observed], x[observed])
    assert torch.isfinite(x_hat).all()


def test_decay_moves_toward_the_mean_with_time():
    x = torch.tensor([[[4.0]] + [[NAN]] * 6])
    x_hat, _, _ = MissingValueDecay(1)(x)
    vals = x_hat[0, :, 0]
    assert torch.all(vals[1:] <= vals[:-1]) and torch.all(vals > 0)


def test_missing_terms_vanish_on_complete_inputs():
    torch.manual_seed(0)
    vsn = VariableSelection(3, 8, missing_aware=True)
    x = torch.randn(2, 5, 3)
    ones, zeros = torch.ones_like(x), torch.zeros_like(x)
    a, wa = vsn(x)
    b, wb = vsn(x, mask=ones, delta=zeros)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(wa, wb)
    with pytest.raises(ValueError, match="both"):
        vsn(x, mask=ones)


def make(**kw):
    torch.manual_seed(0)
    kw.setdefault("quantiles", LEVELS)
    return XTFT(2, 3, 1, 4, hidden_units=16, num_heads=2, **kw)


def inputs(rate=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    xs = torch.randn(6, 2, generator=g)
    xd = torch.randn(6, 24, 3, generator=g)
    xf = torch.randn(6, 28, 1, generator=g)
    return xs, with_gaps(xd, rate, seed) if rate else xd, xf


def test_nan_inputs_give_finite_outputs_and_gradients():
    model = make()
    xs, xd, xf = inputs(rate=0.3)
    q = model(xs, xd, xf)
    assert torch.isfinite(q).all()
    y = torch.randn(6, 4)
    y[0, 1] = NAN
    pinball_loss(y, q, model.quantiles).backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
    assert model.missing_decay.weight.grad.abs().sum() > 0


def test_missing_parameters_do_not_affect_complete_data():
    model = make().eval()
    xs, xd, xf = inputs()
    with torch.no_grad():
        base = model(xs, xd, xf)
        model.past_vsn.missing_u.add_(5.0)
        model.past_vsn.missing_s.add_(5.0)
        model.missing_decay.weight.add_(3.0)
        moved = model(xs, xd, xf)
    torch.testing.assert_close(base, moved)


def test_without_missing_nan_propagates():
    model = make(missing=False).eval()
    xs, xd, xf = inputs(rate=0.3)
    with torch.no_grad():
        assert torch.isnan(model(xs, xd, xf)).any()


def test_instance_norm_with_gaps_is_affine_equivariant():
    torch.manual_seed(0)
    model = XTFT(
        0,
        1,
        0,
        4,
        quantiles=LEVELS,
        hidden_units=16,
        num_heads=2,
        instance_norm=True,
    ).eval()
    xd = with_gaps(torch.randn(5, 24, 1), 0.3, seed=2)
    with torch.no_grad():
        base = model(None, xd, None)
        moved = model(None, 2.5 * xd - 7.0, None)
    torch.testing.assert_close(
        moved, 2.5 * base - 7.0, rtol=1e-4, atol=1e-4
    )


def test_fully_missing_window_is_finite():
    model = make(instance_norm=True).eval()
    xs, xd, xf = inputs()
    xd[2] = NAN
    with torch.no_grad():
        assert torch.isfinite(model(xs, xd, xf)).all()


def test_window_dataset_fill_value():
    y = np.arange(20.0)
    y[[3, 9]] = np.nan
    df = pd.DataFrame({"unique_id": "a", "ds": np.arange(20), "y": y})
    panel = PanelData.from_frame(df, PanelSpec())
    idx = panel.origins(np.array([0]), None, 10, 2, 1)
    filled = WindowDataset(panel, idx, 10, 2)[0]["past"]
    kept = WindowDataset(panel, idx, 10, 2, fill_value=None)[0]["past"]
    assert np.isfinite(filled).all()
    assert np.isnan(kept[[3, 9], 0]).all()
