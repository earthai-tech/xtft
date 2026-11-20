"""Hierarchical attention for XTFT (PyTorch).

Two stages replace the original "hierarchical" attention, which
only added the outputs of two attention modules:

1. :class:`LocalWindowAttention` attends causally *within* one
   time resolution, with a per-head learned window.
2. :class:`CrossScaleAttention` lets each forecast step attend
   *across* the tokens of all resolutions at once and reports how
   much attention mass each resolution receives.

Both windows are soft and differentiable: a linear penalty on the
time distance, in the spirit of ALiBi [1]_, whose slope is learned.

References
----------
.. [1] Press, O., Smith, N. A. and Lewis, M. (2022). Train short,
   test long: Attention with linear biases enables input length
   extrapolation. International Conference on Learning
   Representations (ICLR).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .grn import GatedLinearUnit

__all__ = ["CrossScaleAttention", "LocalWindowAttention"]


def _inverse_softplus(y: float) -> float:
    """Return :math:`x` such that ``softplus(x) == y`` (``y > 0``)."""
    return math.log(math.expm1(y))


class _WindowSlopes(nn.Module):
    r"""Per-head slopes :math:`\lambda_h = \mathrm{softplus}(\rho_h)`.

    Heads start at windows spread geometrically between 1 and
    `init_window` steps, so different heads begin at different
    time ranges.
    """

    def __init__(self, num_heads: int, init_window: float) -> None:
        super().__init__()
        if init_window < 1:
            raise ValueError(
                f"init_window must be >= 1, got {init_window}."
            )
        windows = (
            torch.logspace(0, math.log10(init_window), num_heads)
            if num_heads > 1
            else torch.tensor([float(init_window)])
        )
        rho = [_inverse_softplus(1.0 / w) for w in windows.tolist()]
        self.rho = nn.Parameter(torch.tensor(rho))

    def forward(self) -> Tensor:
        return nn.functional.softplus(self.rho)

    def windows(self) -> Tensor:
        r"""Effective windows :math:`1/\lambda_h` in time steps."""
        return 1.0 / self.forward()


class LocalWindowAttention(nn.Module):
    r"""Causal self-attention with a learned soft window per head.

    Parameters
    ----------
    d_model : int
        Feature width :math:`d`; must be divisible by `num_heads`.
    num_heads : int, default=4
        Number of heads :math:`n_h`.
    init_window : float, default=8.0
        Largest initial window (in steps of this resolution).
    dropout : float, default=0.0
        Dropout on attention weights and on the gated output.

    Notes
    -----
    For queries :math:`q_i`, keys :math:`k_j` of head :math:`h`
    with width :math:`d_k = d / n_h`,

    .. math::

        A^{h}_{ij} = \mathrm{softmax}_j\Bigl(
            \frac{q_i^\top k_j}{\sqrt{d_k}}
            - \lambda_h\,(i - j)\Bigr),
        \qquad j \le i,

    and :math:`A^h_{ij} = 0` for :math:`j > i`. The slope
    :math:`\lambda_h > 0` is learned; at equal content scores the
    weight decays by a factor :math:`e` every
    :math:`W_h = 1/\lambda_h` steps, so :math:`W_h` is the learned
    window. The block output is
    :math:`\mathrm{LayerNorm}(x + \mathrm{GLU}(\mathrm{MHA}(x)))`.
    The cost is :math:`O(T^2 d)` for a sequence of length
    :math:`T`, applied to the coarse sequences of
    :class:`~xtft.nn.torch.layers.MultiResolutionEncoder`.

    Examples
    --------
    >>> import torch
    >>> from xtft.nn.torch.layers import LocalWindowAttention
    >>> att = LocalWindowAttention(8, num_heads=2)
    >>> att(torch.randn(3, 6, 8)).shape
    torch.Size([3, 6, 8])
    >>> att.windows().shape
    torch.Size([2])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        init_window: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError(
                f"d_model={d_model} is not divisible by "
                f"num_heads={num_heads}."
            )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.slopes = _WindowSlopes(num_heads, init_window)
        self.gate = GatedLinearUnit(d_model, d_model, dropout)
        self.norm = nn.LayerNorm(d_model)

    def windows(self) -> Tensor:
        """Return the learned windows :math:`W_h`, shape ``(n_h,)``."""
        return self.slopes.windows()

    def _bias(self, length: int, device, dtype) -> Tensor:
        pos = torch.arange(length, device=device)
        dist = (pos.view(-1, 1) - pos.view(1, -1)).to(dtype)
        bias = -self.slopes().to(dtype).view(-1, 1, 1) * dist
        future = dist < 0
        return bias.masked_fill(future, float("-inf"))

    def forward(self, x: Tensor) -> Tensor:
        """Apply windowed causal self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Sequence of shape ``(B, T, d)``.

        Returns
        -------
        torch.Tensor
            Sequence of shape ``(B, T, d)``.
        """
        batch, length, d_model = x.shape
        qkv = self.qkv(x).view(
            batch, length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        bias = self._bias(length, x.device, q.dtype)
        attended = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias.unsqueeze(0),
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(
            batch, length, d_model
        )
        return self.norm(x + self.gate(self.out(attended)))


class CrossScaleAttention(nn.Module):
    r"""Attention from forecast steps to all resolutions at once.

    Parameters
    ----------
    d_model : int
        Feature width :math:`d`; divisible by `num_heads`.
    scales : sequence of int
        Decimation factors of the memories, finest first; used to
        express token ages in original time steps.
    num_heads : int, default=4
        Number of heads.
    init_window : float, default=16.0
        Largest initial recency window, in original time steps.
    dropout : float, default=0.0
        Dropout on attention weights and on the gated output.

    Notes
    -----
    Let :math:`m^{(s)}_k`, :math:`k = 0, \dots, T_s - 1`, be the
    tokens of scale :math:`s`; token :math:`k` summarises the
    history up to :math:`a^{(s)}_k = (T_s - 1 - k)\, s` steps
    before the forecast origin. With a learned scale embedding
    :math:`e_s`, keys and values are built from
    :math:`m^{(s)}_k + e_s` over the union of all scales. For a
    forecast-step query :math:`q_\tau`,

    .. math::

        A^{h}_{\tau,(s,k)} = \mathrm{softmax}_{(s,k)}\Bigl(
            \frac{q_\tau^\top k_{s,k}}{\sqrt{d_k}}
            - \mu_h\, a^{(s)}_k \Bigr),

    with learned :math:`\mu_h > 0`. The *scale attribution* of
    step :math:`\tau` is the head-averaged attention mass per
    scale,
    :math:`\pi_{\tau, s} = \frac{1}{n_h}\sum_h \sum_k
    A^h_{\tau,(s,k)}`, so :math:`\sum_s \pi_{\tau,s} = 1`. The
    cost is :math:`O(H d \sum_s T_s)` for :math:`H` forecast
    steps, linear in the history length.

    Examples
    --------
    >>> import torch
    >>> from xtft.nn.torch.layers import CrossScaleAttention
    >>> att = CrossScaleAttention(8, scales=(1, 2), num_heads=2)
    >>> mem = [torch.randn(2, 6, 8), torch.randn(2, 3, 8)]
    >>> out, pi = att(torch.randn(2, 4, 8), mem)
    >>> out.shape, pi.shape
    (torch.Size([2, 4, 8]), torch.Size([2, 4, 2]))
    >>> torch.allclose(pi.sum(-1), torch.ones(2, 4))
    True
    """

    def __init__(
        self,
        d_model: int,
        scales: Sequence[int],
        num_heads: int = 4,
        init_window: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError(
                f"d_model={d_model} is not divisible by "
                f"num_heads={num_heads}."
            )
        self.scales = tuple(scales)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale_embedding = nn.Parameter(
            torch.zeros(len(self.scales), d_model)
        )
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.slopes = _WindowSlopes(num_heads, init_window)
        self.gate = GatedLinearUnit(d_model, d_model, dropout)
        self.norm = nn.LayerNorm(d_model)

    def windows(self) -> Tensor:
        """Return the learned recency windows, shape ``(n_h,)``."""
        return self.slopes.windows()

    def forward(
        self, query: Tensor, memories: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Attend from forecast steps to multi-resolution memories.

        Parameters
        ----------
        query : torch.Tensor
            Forecast-step features, shape ``(B, H, d)``.
        memories : sequence of torch.Tensor
            One tensor per scale, shape ``(B, T_s, d)``, in the
            order of `scales`.

        Returns
        -------
        output : torch.Tensor
            Shape ``(B, H, d)``.
        scale_attribution : torch.Tensor
            Shape ``(B, H, S)``; rows sum to one.

        Raises
        ------
        ValueError
            If the number of memories differs from the number of
            scales.
        """
        if len(memories) != len(self.scales):
            raise ValueError(
                f"Expected {len(self.scales)} memories, got "
                f"{len(memories)}."
            )
        batch, horizon, d_model = query.shape
        tokens, ages, sizes = [], [], []
        pairs = zip(memories, self.scales, strict=True)
        for i, (mem, s) in enumerate(pairs):
            tokens.append(mem + self.scale_embedding[i])
            n = mem.shape[1]
            sizes.append(n)
            ages.append(
                torch.arange(n - 1, -1, -1, device=mem.device) * s
            )
        memory = torch.cat(tokens, dim=1)
        age = torch.cat(ages).to(query.dtype)
        q = (
            self.q(query)
            .view(batch, horizon, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k, v = self.kv(memory).chunk(2, dim=-1)
        k = k.view(batch, -1, self.num_heads, self.head_dim)
        v = v.view(batch, -1, self.num_heads, self.head_dim)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        bias = -self.slopes().to(query.dtype).view(-1, 1, 1) * age
        weights = torch.softmax(scores + bias, dim=-1)
        attended = self.attn_dropout(weights) @ v
        attended = attended.transpose(1, 2).reshape(
            batch, horizon, d_model
        )
        output = self.norm(query + self.gate(self.out(attended)))
        per_scale = torch.stack(
            [w.sum(-1) for w in weights.split(sizes, dim=-1)], dim=-1
        )
        return output, per_scale.mean(dim=1)
