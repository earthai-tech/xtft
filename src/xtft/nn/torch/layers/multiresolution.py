"""Anti-aliased multi-resolution temporal encoding (PyTorch).

The original XTFT built coarse views of the history with
``x[:, ::s]``: no low-pass filter (so frequencies above the new
Nyquist limit fold back as aliases) and a grid anchored at the
oldest step (so, unless :math:`s` divides :math:`T-1`, the most
recent observation was dropped). This module replaces it with a
filtered, origin-aligned decimator.

References
----------
.. [1] Oppenheim, A. V. and Schafer, R. W. (2010). Discrete-Time
   Signal Processing, 3rd ed. Pearson. Sections 4.6 (sampling rate
   reduction) and 7.5 (FIR design by windowing).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

__all__ = [
    "AntiAliasedDecimator",
    "MultiResolutionEncoder",
    "windowed_sinc_lowpass",
]


def windowed_sinc_lowpass(factor: int, taps_per_factor: int = 2):
    r"""Design a Hann-windowed sinc low-pass filter for decimation.

    Parameters
    ----------
    factor : int
        Decimation factor :math:`s \geq 1`.
    taps_per_factor : int, default=2
        Filter length is :math:`L = \text{taps\_per\_factor}\,s`.

    Returns
    -------
    torch.Tensor
        Kernel of shape ``(L,)``, dtype float32, summing to one
        (unit gain at zero frequency).

    Raises
    ------
    ValueError
        If `factor` or `taps_per_factor` is smaller than 1.

    Notes
    -----
    With centre :math:`m = (L-1)/2` and the Hann window
    :math:`w[n] = \tfrac12\bigl(1 - \cos\frac{2\pi (n+1)}{L+1}
    \bigr)`, which is non-zero at both ends,

    .. math::

        h[n] \propto \frac{1}{s}\,
            \mathrm{sinc}\!\Bigl(\frac{n - m}{s}\Bigr)\, w[n],
        \qquad n = 0, \dots, L-1,

    with :math:`\mathrm{sinc}(u) = \sin(\pi u)/(\pi u)`, i.e. an
    ideal low-pass with cutoff :math:`\omega_c = \pi/s` rad/sample
    (the Nyquist frequency after keeping one sample in
    :math:`s`) truncated by the window [1]_. The length is even so
    that no tap falls on a zero of the sinc: with the odd length
    :math:`2s+1` both end taps vanish and the newest sample would
    get no weight. For :math:`s = 1` the kernel is the unit
    impulse.

    Examples
    --------
    >>> from xtft.nn.torch.layers import windowed_sinc_lowpass
    >>> h = windowed_sinc_lowpass(2)
    >>> h.shape, round(float(h.sum()), 6)
    (torch.Size([4]), 1.0)
    """
    if factor < 1 or taps_per_factor < 1:
        raise ValueError(
            "factor and taps_per_factor must be >= 1, got "
            f"{factor} and {taps_per_factor}."
        )
    if factor == 1:
        return torch.ones(1)
    length = taps_per_factor * factor
    n = torch.arange(length, dtype=torch.float64)
    centre = (length - 1) / 2
    kernel = torch.sinc((n - centre) / factor) / factor
    window = 0.5 * (
        1 - torch.cos(2 * math.pi * (n + 1) / (length + 1))
    )
    kernel = kernel * window
    return (kernel / kernel.sum()).float()


class AntiAliasedDecimator(nn.Module):
    r"""Causal low-pass filter followed by origin-aligned decimation.

    Parameters
    ----------
    factor : int
        Decimation factor :math:`s \geq 1`.
    antialias : bool, default=True
        Filter before decimating. ``False`` keeps raw samples
        (the ablation that reproduces plain striding).
    learnable : bool, default=True
        Let the filter taps be trained, starting from
        :func:`windowed_sinc_lowpass`.
    align : {"origin", "start"}, default="origin"
        Anchor of the coarse grid. ``"origin"`` keeps the most
        recent step :math:`T-1`; ``"start"`` keeps step 0, as
        ``x[:, ::s]`` does (ablation only).
    taps_per_factor : int, default=2
        Filter length factor, see :func:`windowed_sinc_lowpass`.

    Notes
    -----
    For a sequence :math:`x_0, \dots, x_{T-1}` the filter is
    applied causally,
    :math:`y_t = \sum_{n=0}^{L-1} h[n]\, x_{t-L+1+n}`, with the
    first value repeated to pad :math:`t < 0`, so :math:`y_t` uses
    only :math:`x_{\le t}`. The kernel is renormalised to unit DC
    gain at every call, which keeps the level of the series
    unchanged while it is trained. The output keeps
    :math:`y_t` for :math:`t \in \{T-1, T-1-s, \dots\}`, giving
    :math:`T_s = \lceil T/s \rceil` steps. A causal linear-phase
    filter delays the signal by :math:`(L-1)/2` steps; a learnable
    kernel can trade some of that delay against attenuation.

    Examples
    --------
    >>> import torch
    >>> from xtft.nn.torch.layers import AntiAliasedDecimator
    >>> dec = AntiAliasedDecimator(4)
    >>> dec(torch.randn(2, 10, 3)).shape
    torch.Size([2, 3, 3])
    >>> x = torch.arange(10.).view(1, 10, 1)
    >>> AntiAliasedDecimator(4, antialias=False)(x).flatten()
    tensor([1., 5., 9.])
    """

    def __init__(
        self,
        factor: int,
        *,
        antialias: bool = True,
        learnable: bool = True,
        align: Literal["origin", "start"] = "origin",
        taps_per_factor: int = 2,
    ) -> None:
        super().__init__()
        if not isinstance(factor, int) or factor < 1:
            raise ValueError(
                f"factor must be an int >= 1, got {factor}."
            )
        if align not in ("origin", "start"):
            raise ValueError(
                f"align must be 'origin' or 'start', got {align!r}."
            )
        self.factor = factor
        self.antialias = antialias and factor > 1
        self.align = align
        kernel = windowed_sinc_lowpass(factor, taps_per_factor)
        if self.antialias and learnable:
            self.kernel = nn.Parameter(kernel)
        else:
            self.register_buffer("kernel", kernel)

    def forward(self, x: Tensor) -> Tensor:
        """Filter and decimate along time.

        Parameters
        ----------
        x : torch.Tensor
            Sequence of shape ``(B, T, d)``.

        Returns
        -------
        torch.Tensor
            Coarse sequence of shape ``(B, ceil(T / s), d)``, in
            chronological order.
        """
        if self.factor == 1:
            return x
        steps = x.shape[1]
        if self.antialias:
            kernel = self.kernel
            # Unit DC gain; keep the sign to avoid a 0-crossing.
            total = kernel.sum()
            safe = torch.where(
                total.abs() < 1e-3,
                torch.full_like(total, 1e-3).copysign(total),
                total,
            )
            kernel = kernel / safe
            channels = x.shape[-1]
            seq = x.transpose(1, 2)
            pad = kernel.numel() - 1
            seq = torch.cat(
                [seq[..., :1].expand(-1, -1, pad), seq], dim=-1
            )
            # conv1d computes cross-correlation, so flip the taps.
            weight = kernel.flip(0).view(1, 1, -1)
            weight = weight.expand(channels, 1, -1)
            seq = nn.functional.conv1d(seq, weight, groups=channels)
            x = seq.transpose(1, 2)
        start = (
            (steps - 1) % self.factor if self.align == "origin" else 0
        )
        return x[:, start :: self.factor]


class MultiResolutionEncoder(nn.Module):
    r"""Encode the history at several time resolutions.

    Parameters
    ----------
    d_model : int
        Feature width :math:`d` of inputs and outputs.
    scales : sequence of int, default=(1, 2, 4)
        Strictly increasing decimation factors; each must be
        :math:`\geq 1`.
    cell : {"lstm", "gru"}, default="lstm"
        Recurrent encoder used at every scale.
    context_features : int or None, default=None
        Size of a static context used to initialise the recurrent
        state at every scale (as TFT does for its LSTM).
    antialias : bool, default=True
        See :class:`AntiAliasedDecimator`.
    align : {"origin", "start"}, default="origin"
        See :class:`AntiAliasedDecimator`.

    Notes
    -----
    For each scale :math:`s`, the history :math:`z_{0:T-1}` is
    decimated to :math:`z^{(s)} = D_s(h_s * z)` and encoded by a
    scale-specific recurrent network,
    :math:`e^{(s)}_k = \mathrm{RNN}_s(z^{(s)}_k, e^{(s)}_{k-1})`,
    :math:`k = 0, \dots, T_s - 1`. The cost is
    :math:`O(d^2 \sum_s T/s) \le O(2 T d^2)` for dyadic scales,
    i.e. at most twice the single-resolution cost.

    Examples
    --------
    >>> import torch
    >>> from xtft.nn.torch.layers import MultiResolutionEncoder
    >>> enc = MultiResolutionEncoder(8, scales=(1, 3))
    >>> outs, state = enc(torch.randn(2, 10, 8))
    >>> [tuple(o.shape) for o in outs]
    [(2, 10, 8), (2, 4, 8)]
    """

    def __init__(
        self,
        d_model: int,
        scales: Sequence[int] = (1, 2, 4),
        *,
        cell: Literal["lstm", "gru"] = "lstm",
        context_features: int | None = None,
        antialias: bool = True,
        align: Literal["origin", "start"] = "origin",
    ) -> None:
        super().__init__()
        scales = tuple(scales)
        if not scales or any(
            not isinstance(s, int) or s < 1 for s in scales
        ):
            raise ValueError(
                f"scales must be positive ints, got {scales}."
            )
        if list(scales) != sorted(set(scales)):
            raise ValueError(
                f"scales must be strictly increasing, got {scales}."
            )
        if cell not in ("lstm", "gru"):
            raise ValueError(
                f"cell must be 'lstm' or 'gru', got {cell!r}."
            )
        self.scales = scales
        self.cell = cell
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.decimators = nn.ModuleList(
            AntiAliasedDecimator(s, antialias=antialias, align=align)
            for s in scales
        )
        self.rnns = nn.ModuleList(
            rnn_cls(d_model, d_model, batch_first=True) for _ in scales
        )
        n_states = 2 if cell == "lstm" else 1
        self.init_state = (
            nn.ModuleList(
                nn.Linear(context_features, n_states * d_model)
                for _ in scales
            )
            if context_features
            else None
        )

    def _initial_state(self, i: int, context: Tensor | None):
        if context is None or self.init_state is None:
            return None
        states = torch.tanh(self.init_state[i](context))
        states = states.unsqueeze(0).contiguous()
        if self.cell == "lstm":
            h0, c0 = states.chunk(2, dim=-1)
            return h0.contiguous(), c0.contiguous()
        return states

    def forward(
        self, x: Tensor, context: Tensor | None = None
    ) -> tuple[list[Tensor], object]:
        """Encode `x` at every scale.

        Parameters
        ----------
        x : torch.Tensor
            History of shape ``(B, T, d)``.
        context : torch.Tensor or None, optional
            Static context of shape ``(B, context_features)``.

        Returns
        -------
        outputs : list of torch.Tensor
            One tensor per scale, shape ``(B, ceil(T / s), d)``.
        final_state : tensor or tuple of tensors
            Final recurrent state of the finest scale, used to
            initialise the decoder (``(h, c)`` for LSTM).
        """
        outputs, final_state = [], None
        for i, (dec, rnn) in enumerate(
            zip(self.decimators, self.rnns, strict=True)
        ):
            out, state = rnn(dec(x), self._initial_state(i, context))
            outputs.append(out)
            if i == 0:
                final_state = state
        return outputs, final_state
