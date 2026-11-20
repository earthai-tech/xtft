"""Tests for the quantile head and the pinball loss."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xtft.nn.torch.probabilistic import (  # noqa: E402
    NonCrossingQuantileHead,
    QuantileLoss,
    pinball_loss,
    validate_quantiles,
)

LEVELS = [0.05, 0.1, 0.5, 0.9, 0.95]


@pytest.mark.parametrize(
    "bad", [[], [0.0, 0.5], [0.5, 1.0], [0.5, 0.1], [0.2, 0.2]]
)
def test_validate_quantiles_rejects(bad):
    with pytest.raises(ValueError):
        validate_quantiles(bad)


@pytest.mark.parametrize(
    "levels", [LEVELS, [0.1, 0.9], [0.2, 0.4, 0.6, 0.8], [0.3]]
)
def test_head_never_crosses(levels):
    torch.manual_seed(0)
    head = NonCrossingQuantileHead(6, output_dim=2, quantiles=levels)
    # Large inputs stress the monotone construction.
    q = head(50 * torch.randn(64, 4, 6))
    assert q.shape == (64, 4, 2, len(levels))
    if len(levels) > 1:
        assert torch.all(q.diff(dim=-1) >= 0)


def test_head_anchor_prefers_median():
    head = NonCrossingQuantileHead(4, quantiles=[0.1, 0.5, 0.7, 0.9])
    assert head.anchor == 1


def test_head_point_forecast_shape():
    head = NonCrossingQuantileHead(4, output_dim=3)
    assert head(torch.randn(2, 5, 4)).shape == (2, 5, 3)


def test_pinball_matches_numpy():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(7, 3))
    q = rng.normal(size=(7, 3, len(LEVELS)))
    u = y[..., None] - q
    tau = np.array(LEVELS)
    expected = np.maximum(tau * u, (tau - 1) * u).mean()
    got = pinball_loss(torch.tensor(y), torch.tensor(q), LEVELS)
    assert float(got) == pytest.approx(expected, rel=1e-12)


def test_pinball_ignores_missing_targets():
    y = torch.tensor([1.0, float("nan"), 3.0])
    q = torch.tensor([[1.5], [1e6], [2.0]])
    got = pinball_loss(y, q, [0.5])
    expected = (0.5 * 0.5 + 0.5 * 1.0) / 2
    assert float(got) == pytest.approx(expected)


def test_pinball_all_missing_is_zero_with_gradient():
    q = torch.zeros(4, 2, requires_grad=True)
    loss = pinball_loss(torch.full((4,), float("nan")), q, [0.1, 0.9])
    loss.backward()
    assert float(loss) == 0.0
    assert torch.all(torch.isfinite(q.grad))


def test_pinball_minimiser_is_empirical_quantile():
    torch.manual_seed(0)
    y = torch.randn(4000)
    grid = torch.linspace(-3, 3, 601)
    for tau in (0.1, 0.5, 0.9):
        losses = torch.stack(
            [
                pinball_loss(y, torch.full((4000, 1), float(c)), [tau])
                for c in grid
            ]
        )
        best = float(grid[losses.argmin()])
        assert best == pytest.approx(
            float(torch.quantile(y, tau)), abs=0.02
        )


def test_pinball_shape_checks():
    with pytest.raises(ValueError):
        pinball_loss(torch.zeros(3), torch.zeros(4, 2), [0.1, 0.9])
    with pytest.raises(ValueError):
        pinball_loss(torch.zeros(3), torch.zeros(3, 3), [0.1, 0.9])
    with pytest.raises(ValueError):
        pinball_loss(
            torch.zeros(3), torch.zeros(3, 1), [0.5], reduction="sum"
        )


def test_quantile_loss_module():
    loss = QuantileLoss([0.5])
    assert float(loss(torch.zeros(2), torch.ones(2, 1))) == 0.5
