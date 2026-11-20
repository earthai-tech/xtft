"""Tests for the PyTorch XTFT layers."""

import math

import pytest

torch = pytest.importorskip("torch")

from xtft.nn.torch.layers import (  # noqa: E402
    AntiAliasedDecimator,
    CrossScaleAttention,
    GatedResidualNetwork,
    LocalWindowAttention,
    MultiResolutionEncoder,
    VariableSelection,
    windowed_sinc_lowpass,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- low-pass design -------------------------------------------------


@pytest.mark.parametrize("factor", [2, 3, 4, 8])
def test_lowpass_unit_dc_gain_and_symmetry(factor):
    h = windowed_sinc_lowpass(factor)
    assert h.shape == (2 * factor,)
    assert math.isclose(float(h.sum()), 1.0, rel_tol=1e-6)
    assert torch.allclose(h, h.flip(0), atol=1e-7)


@pytest.mark.parametrize("factor", [2, 4])
def test_lowpass_attenuates_above_new_nyquist(factor):
    h = windowed_sinc_lowpass(factor, taps_per_factor=4).double()
    response = torch.fft.rfft(h, n=1024).abs()
    omega = torch.linspace(0, math.pi, response.numel())
    stop = response[omega > 1.5 * math.pi / factor]
    assert float(response[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(stop.max()) < 0.1


def test_lowpass_identity_for_factor_one():
    assert torch.equal(windowed_sinc_lowpass(1), torch.ones(1))


def test_lowpass_rejects_bad_factor():
    with pytest.raises(ValueError):
        windowed_sinc_lowpass(0)


# --- decimator -------------------------------------------------------


def test_decimator_origin_keeps_latest_step():
    x = torch.arange(10.0).view(1, 10, 1)
    out = AntiAliasedDecimator(4, antialias=False)(x)
    assert out.flatten().tolist() == [1.0, 5.0, 9.0]


def test_decimator_start_matches_plain_striding():
    x = torch.randn(2, 10, 3)
    dec = AntiAliasedDecimator(4, antialias=False, align="start")
    assert torch.equal(dec(x), x[:, ::4])


@pytest.mark.parametrize("steps", [7, 8, 9, 16])
def test_decimator_length(steps):
    out = AntiAliasedDecimator(3)(torch.randn(2, steps, 4))
    assert out.shape == (2, math.ceil(steps / 3), 4)


def test_decimator_preserves_constant_series():
    x = torch.full((2, 12, 3), 5.0)
    out = AntiAliasedDecimator(4)(x)
    assert torch.allclose(out, torch.full_like(out, 5.0), atol=1e-5)


def test_decimator_is_causal():
    dec = AntiAliasedDecimator(2)
    x = torch.randn(1, 12, 2)
    y = x.clone()
    y[:, -1] += 100.0
    # Only the coarse sample taken at the last step may change.
    assert torch.allclose(dec(x)[:, :-1], dec(y)[:, :-1])
    assert not torch.allclose(dec(x)[:, -1], dec(y)[:, -1])


def test_decimator_suppresses_aliasing():
    # Period 2.5 steps: above the Nyquist limit after s = 4
    # (period 8), so plain striding aliases it to a slow wave.
    t = torch.arange(400.0)
    x = torch.sin(2 * math.pi * t / 2.5).view(1, -1, 1)
    raw = AntiAliasedDecimator(4, antialias=False)(x)
    filt = AntiAliasedDecimator(4, learnable=False)(x)
    assert float(filt[:, 10:].std()) < 0.2 * float(raw[:, 10:].std())


def test_decimator_kernel_trainable_flag():
    assert isinstance(
        AntiAliasedDecimator(2).kernel, torch.nn.Parameter
    )
    fixed = AntiAliasedDecimator(2, learnable=False).kernel
    assert not isinstance(fixed, torch.nn.Parameter)


@pytest.mark.parametrize(
    "kwargs", [{"factor": 0}, {"factor": 2, "align": "end"}]
)
def test_decimator_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        AntiAliasedDecimator(**kwargs)


# --- encoder ---------------------------------------------------------


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_encoder_shapes_and_state(cell):
    enc = MultiResolutionEncoder(
        8, (1, 2, 4), cell=cell, context_features=5
    )
    outs, state = enc(torch.randn(3, 10, 8), torch.randn(3, 5))
    assert [o.shape[1] for o in outs] == [10, 5, 3]
    if cell == "lstm":
        assert state[0].shape == (1, 3, 8)
    else:
        assert state.shape == (1, 3, 8)


@pytest.mark.parametrize("scales", [(), (2, 1), (1, 1), (0, 1)])
def test_encoder_rejects_bad_scales(scales):
    with pytest.raises(ValueError):
        MultiResolutionEncoder(8, scales)


# --- attention -------------------------------------------------------


def test_local_attention_is_causal():
    att = LocalWindowAttention(8, num_heads=2).eval()
    x = torch.randn(2, 9, 8)
    y = x.clone()
    y[:, -1] += 10.0
    assert torch.allclose(att(x)[:, :-1], att(y)[:, :-1], atol=1e-6)


def test_local_attention_windows_are_positive_and_spread():
    att = LocalWindowAttention(8, num_heads=4, init_window=16.0)
    w = att.windows()
    assert torch.all(w > 0)
    assert float(w.min()) == pytest.approx(1.0, rel=1e-4)
    assert float(w.max()) == pytest.approx(16.0, rel=1e-4)


def test_local_attention_rejects_bad_heads():
    with pytest.raises(ValueError):
        LocalWindowAttention(10, num_heads=4)


def test_cross_scale_attribution_is_a_distribution():
    att = CrossScaleAttention(8, scales=(1, 2, 4), num_heads=2)
    mem = [torch.randn(2, n, 8) for n in (8, 4, 2)]
    out, pi = att(torch.randn(2, 3, 8), mem)
    assert out.shape == (2, 3, 8)
    assert pi.shape == (2, 3, 3)
    assert torch.all(pi >= 0)
    assert torch.allclose(pi.sum(-1), torch.ones(2, 3), atol=1e-6)


def test_cross_scale_rejects_wrong_memory_count():
    att = CrossScaleAttention(8, scales=(1, 2), num_heads=2)
    with pytest.raises(ValueError):
        att(torch.randn(2, 3, 8), [torch.randn(2, 4, 8)])


# --- GRN and variable selection --------------------------------------


def test_grn_context_broadcasts_over_time():
    grn = GatedResidualNetwork(4, 8, 6, context_features=3)
    assert grn(torch.randn(2, 5, 4), torch.randn(2, 3)).shape == (
        2,
        5,
        6,
    )


def test_grn_without_context_rejects_context():
    grn = GatedResidualNetwork(4, 8)
    with pytest.raises(ValueError):
        grn(torch.randn(2, 4), torch.randn(2, 3))


def test_variable_selection_weights_form_simplex():
    vsn = VariableSelection(5, 8, context_features=3)
    z, w = vsn(torch.randn(4, 6, 5), torch.randn(4, 3))
    assert z.shape == (4, 6, 8)
    assert torch.all(w >= 0)
    assert torch.allclose(w.sum(-1), torch.ones(4, 6), atol=1e-6)


def test_variable_selection_rejects_wrong_width():
    with pytest.raises(ValueError):
        VariableSelection(3, 8)(torch.randn(2, 4))
    with pytest.raises(ValueError):
        VariableSelection(0, 8)
