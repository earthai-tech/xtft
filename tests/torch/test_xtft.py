"""Tests for the PyTorch XTFT model."""

import pytest

torch = pytest.importorskip("torch")

from xtft.nn.torch.models import XTFT  # noqa: E402
from xtft.nn.torch.probabilistic import pinball_loss  # noqa: E402

B, T, H = 6, 16, 4
LEVELS = [0.1, 0.5, 0.9]


def make_inputs(s=3, d=4, f=2, horizon_only=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    xs = torch.randn(B, s, generator=g) if s else None
    xd = torch.randn(B, T, d, generator=g)
    steps = H if horizon_only else T + H
    xf = torch.randn(B, steps, f, generator=g) if f else None
    return xs, xd, xf


def make_model(s=3, d=4, f=2, **kw):
    torch.manual_seed(0)
    kw.setdefault("hidden_units", 16)
    kw.setdefault("num_heads", 2)
    return XTFT(s, d, f, H, **kw)


@pytest.mark.parametrize(
    ("kwargs", "shape"),
    [
        ({"quantiles": LEVELS}, (B, H, 3)),
        ({"quantiles": LEVELS, "output_dim": 2}, (B, H, 2, 3)),
        ({}, (B, H, 1)),
        ({"output_dim": 2}, (B, H, 2)),
    ],
)
def test_output_shapes(kwargs, shape):
    model = make_model(**kwargs)
    assert model(*make_inputs()).shape == shape


@pytest.mark.parametrize(("s", "f"), [(0, 2), (3, 0), (0, 0)])
def test_optional_static_and_known_inputs(s, f):
    model = make_model(s=s, f=f, quantiles=LEVELS)
    out = model(*make_inputs(s=s, f=f))
    assert out.shape == (B, H, 3)


def test_horizon_only_known_inputs():
    model = make_model(quantiles=LEVELS)
    out = model(*make_inputs(horizon_only=True))
    assert out.shape == (B, H, 3)


def test_sequence_input_matches_positional():
    model = make_model(quantiles=LEVELS).eval()
    inputs = make_inputs()
    assert torch.equal(model(list(inputs)), model(*inputs))


@pytest.mark.parametrize("cell", ["lstm", "gru"])
@pytest.mark.parametrize("scales", [(1,), (1, 2), (1, 3, 5)])
def test_cells_and_scales(cell, scales):
    model = make_model(cell=cell, scales=scales, quantiles=LEVELS)
    out, info = model(*make_inputs(), return_details=True)
    assert out.shape == (B, H, 3)
    assert info["scale_attribution"].shape == (B, H, len(scales))


def test_ablation_switches_run():
    model = make_model(
        antialias=False, align="start", local_attention=False
    )
    out, info = model(*make_inputs(), return_details=True)
    assert out.shape == (B, H, 1)
    assert info["local_windows"] == []


def test_quantiles_never_cross():
    model = make_model(quantiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    out = model(*make_inputs(seed=3))
    assert torch.all(out.diff(dim=-1) > 0)


def test_details_are_valid_distributions():
    model = make_model(quantiles=LEVELS, scales=(1, 2, 4))
    _, info = model(*make_inputs(), return_details=True)
    for key, shape in (
        ("static_weights", (B, 3)),
        ("past_weights", (B, T, 6)),
        ("future_weights", (B, H, 2)),
        ("scale_attribution", (B, H, 3)),
    ):
        w = info[key]
        assert w.shape == shape
        assert torch.allclose(
            w.sum(-1), torch.ones(shape[:-1]), atol=1e-5
        )
    assert len(info["local_windows"]) == 3
    assert torch.all(info["cross_windows"] > 0)


def test_known_future_beyond_horizon_start_affects_output():
    # Known inputs of the forecast steps must reach the forecast.
    model = make_model(quantiles=LEVELS).eval()
    xs, xd, xf = make_inputs()
    xf2 = xf.clone()
    xf2[:, -1] += 5.0
    assert not torch.allclose(model(xs, xd, xf), model(xs, xd, xf2))


def test_deterministic_in_eval_mode():
    model = make_model(quantiles=LEVELS).eval()
    inputs = make_inputs()
    assert torch.equal(model(*inputs), model(*inputs))


def test_all_parameters_receive_gradients():
    model = make_model(quantiles=LEVELS)
    xs, xd, xf = make_inputs()
    y = torch.randn(B, H)
    pinball_loss(y, model(xs, xd, xf), LEVELS).backward()
    missing = [
        n
        for n, p in model.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all()
    ]
    assert missing == []


def test_learns_a_seasonal_signal():
    # Sanity check of trainability: fit y_t = sin(2 pi t / 8) with
    # a phase given as a static input.
    torch.manual_seed(0)
    n = 64
    phase = torch.rand(n, 1) * 6.28
    t = torch.arange(T + H, dtype=torch.float32)
    series = torch.sin(2 * torch.pi * t / 8 + phase)
    xd = series[:, :T, None]
    y = series[:, T:]
    model = XTFT(
        1,
        1,
        0,
        H,
        quantiles=[0.5],
        hidden_units=16,
        num_heads=2,
        dropout=0.0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(150):
        opt.zero_grad()
        loss = pinball_loss(y, model(phase, xd, None), [0.5])
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.25 * losses[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scales": (2, 4)},
        {"scales": ()},
        {"hidden_units": 10, "num_heads": 4},
        {"dropout": 1.0},
        {"quantiles": [0.9, 0.1]},
    ],
)
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        make_model(**kwargs)


def test_constructor_rejects_bad_dims():
    with pytest.raises(ValueError):
        XTFT(0, 0, 0, H)
    with pytest.raises(ValueError):
        XTFT(1, 1, 0, 0)


@pytest.mark.parametrize(
    "bad",
    [
        lambda xs, xd, xf: (xs, xd[..., :2], xf),
        lambda xs, xd, xf: (xs[:, :1], xd, xf),
        lambda xs, xd, xf: (None, xd, xf),
        lambda xs, xd, xf: (xs, xd, xf[:, :-1]),
        lambda xs, xd, xf: (xs, xd, None),
        lambda xs, xd, xf: (xs, xd[0], xf),
    ],
)
def test_forward_rejects_bad_shapes(bad):
    model = make_model()
    with pytest.raises(ValueError):
        model(*bad(*make_inputs()))


def test_num_parameters_counts_trainables():
    model = make_model()
    total = sum(p.numel() for p in model.parameters())
    assert model.num_parameters() == total
    model.head.proj.weight.requires_grad_(False)
    assert model.num_parameters() < total
    assert model.num_parameters(trainable_only=False) == total


@pytest.mark.parametrize("quantiles", [LEVELS, None])
@pytest.mark.parametrize("output_dim", [1, 2])
def test_instance_norm_is_affine_equivariant(quantiles, output_dim):
    # Target-only history: shifting and scaling it (a > 0) must
    # shift and scale every quantile the same way.
    model = make_model(
        s=0,
        d=output_dim,
        f=0,
        quantiles=quantiles,
        output_dim=output_dim,
        instance_norm=True,
    ).eval()
    _, xd, _ = make_inputs(s=0, d=output_dim, f=0)
    a, b = 3.5, -12.0
    with torch.no_grad():
        base = model(None, xd, None)
        moved = model(None, a * xd + b, None)
    torch.testing.assert_close(
        moved, a * base + b, rtol=1e-4, atol=1e-4
    )


def test_instance_norm_keeps_quantiles_ordered():
    model = make_model(quantiles=LEVELS, instance_norm=True).eval()
    xs, xd, xf = make_inputs()
    with torch.no_grad():
        q = model(xs, 100.0 + 50.0 * xd, xf)
    assert torch.all(q[..., 1:] >= q[..., :-1])


def test_instance_norm_flat_window_is_finite():
    model = make_model(
        s=0, d=1, f=0, quantiles=LEVELS, instance_norm=True
    ).eval()
    xd = torch.full((B, T, 1), 7.0)
    with torch.no_grad():
        q = model(None, xd, None)
    assert torch.isfinite(q).all()


def test_instance_norm_needs_target_channels():
    with pytest.raises(ValueError, match="instance_norm"):
        make_model(d=1, output_dim=2, instance_norm=True)


def test_default_is_single_scale():
    model = make_model(quantiles=LEVELS)
    assert model.scales == (1,)
    _, info = model(*make_inputs(), return_details=True)
    assert info["scale_attribution"].shape == (B, H, 1)
    assert torch.allclose(
        info["scale_attribution"], torch.ones(B, H, 1)
    )
