"""Vectorised variable selection network (PyTorch).

Variable selection gives XTFT instance-wise feature weights, the
first of its two interpretability outputs (the second is the
per-scale attention of
:class:`~xtft.nn.torch.layers.CrossScaleAttention`).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .grn import GatedResidualNetwork

__all__ = ["VariableSelection"]


class VariableSelection(nn.Module):
    r"""Embed scalar variables and combine them with learned weights.

    Parameters
    ----------
    num_variables : int
        Number :math:`V \geq 1` of scalar input variables.
    d_model : int
        Embedding width :math:`d`.
    context_features : int or None, default=None
        Size of an optional static context that conditions the
        selection weights.
    dropout : float, default=0.0
        Dropout probability in the weight network.
    missing_aware : bool, default=False
        Add learned terms for missing inputs to the embeddings; the
        forward pass then accepts `mask` and `delta`.

    Notes
    -----
    Each scalar :math:`x_j` is embedded as
    :math:`\xi_j = w_j x_j + b_j \in \mathbb{R}^d`; with
    `missing_aware`, the observation mask :math:`m_j` and the time
    since the last observation :math:`\delta_j` add
    :math:`(1 - m_j)\, u_j + \log(1 + \delta_j)\, s_j` with learned
    :math:`u_j, s_j \in \mathbb{R}^d`, a term that vanishes for
    observed inputs (:math:`m_j = 1, \delta_j = 0`). The embedding
    is refined by a per-variable two-layer network

    .. math::

        \tilde\xi_j = \mathrm{LayerNorm}\bigl(\xi_j
            + U_j\,\mathrm{ELU}(A_j \xi_j + a_j) + u_j\bigr).

    The selection weights and the output are

    .. math::

        v &= \mathrm{softmax}\bigl(\mathrm{GRN}_v(
            [\xi_1, \dots, \xi_V], c)\bigr) \in \Delta^{V-1}, \\
        z &= \textstyle\sum_{j=1}^{V} v_j\, \tilde\xi_j .

    This follows the TFT variable selection network [1]_ with one
    deliberate change: the per-variable GRNs of TFT are replaced by
    the gate-free network above, evaluated for all variables in a
    single batched contraction (``einsum``), so the cost is
    :math:`O(V d^2)` per time step without a Python loop over
    variables.

    References
    ----------
    .. [1] Lim, B., Arik, S. O., Loeff, N. and Pfister, T. (2021).
       Temporal Fusion Transformers for interpretable multi-horizon
       time series forecasting. International Journal of
       Forecasting, 37(4), 1748-1764.
       https://doi.org/10.1016/j.ijforecast.2021.03.012

    Examples
    --------
    >>> import torch
    >>> from xtft.nn.torch.layers import VariableSelection
    >>> vsn = VariableSelection(num_variables=3, d_model=8)
    >>> z, w = vsn(torch.randn(2, 5, 3))
    >>> z.shape, w.shape
    (torch.Size([2, 5, 8]), torch.Size([2, 5, 3]))
    >>> torch.allclose(w.sum(-1), torch.ones(2, 5))
    True
    """

    def __init__(
        self,
        num_variables: int,
        d_model: int,
        context_features: int | None = None,
        dropout: float = 0.0,
        missing_aware: bool = False,
    ) -> None:
        super().__init__()
        if num_variables < 1:
            raise ValueError(
                f"num_variables must be >= 1, got {num_variables}."
            )
        self.num_variables = num_variables
        self.d_model = d_model
        v, d = num_variables, d_model
        self.embed_w = nn.Parameter(torch.empty(v, d))
        self.embed_b = nn.Parameter(torch.zeros(v, d))
        self.missing_aware = missing_aware
        if missing_aware:
            self.missing_u = nn.Parameter(torch.zeros(v, d))
            self.missing_s = nn.Parameter(torch.zeros(v, d))
        self.ff_a = nn.Parameter(torch.empty(v, d, d))
        self.ff_a_b = nn.Parameter(torch.zeros(v, d))
        self.ff_u = nn.Parameter(torch.empty(v, d, d))
        self.ff_u_b = nn.Parameter(torch.zeros(v, d))
        self.norm = nn.LayerNorm(d)
        self.weight_grn = GatedResidualNetwork(
            in_features=v * d,
            hidden_features=d,
            out_features=v,
            context_features=context_features,
            dropout=dropout,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise weights: unit-variance embeddings, uniform FF."""
        nn.init.normal_(self.embed_w, std=1.0)
        if self.missing_aware:
            nn.init.normal_(self.missing_u, std=0.1)
            nn.init.normal_(self.missing_s, std=0.1)
        bound = 1.0 / math.sqrt(self.d_model)
        nn.init.uniform_(self.ff_a, -bound, bound)
        # Small output weights: the refinement starts near identity.
        nn.init.uniform_(self.ff_u, -0.1 * bound, 0.1 * bound)

    def forward(
        self,
        x: Tensor,
        context: Tensor | None = None,
        *,
        mask: Tensor | None = None,
        delta: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Select and combine variables.

        Parameters
        ----------
        x : torch.Tensor
            Inputs of shape ``(B, V)`` or ``(B, T, V)``.
        context : torch.Tensor or None, optional
            Static context of shape ``(B, context_features)``.
        mask, delta : torch.Tensor or None, optional
            Observation mask (1 observed, 0 missing) and steps since
            the last observation, both shaped like `x`; used only
            when the layer is `missing_aware`. Give both or neither.

        Returns
        -------
        z : torch.Tensor
            Combined embedding, shape ``(B, d)`` or ``(B, T, d)``.
        weights : torch.Tensor
            Selection weights, shape ``(B, V)`` or ``(B, T, V)``;
            non-negative and summing to one over the last axis.

        Raises
        ------
        ValueError
            If the last dimension of `x` is not `num_variables`, or
            only one of `mask` and `delta` is given.
        """
        if x.shape[-1] != self.num_variables:
            raise ValueError(
                f"Expected {self.num_variables} variables in the last "
                f"dimension, got {x.shape[-1]}."
            )
        if (mask is None) != (delta is None):
            raise ValueError("Pass both mask and delta, or neither.")
        xi = x.unsqueeze(-1) * self.embed_w + self.embed_b
        if self.missing_aware and mask is not None:
            xi = (
                xi
                + (1.0 - mask).unsqueeze(-1) * self.missing_u
                + torch.log1p(delta).unsqueeze(-1) * self.missing_s
            )
        hidden = torch.einsum("...vd,vde->...ve", xi, self.ff_a)
        hidden = nn.functional.elu(hidden + self.ff_a_b)
        refined = torch.einsum("...vd,vde->...ve", hidden, self.ff_u)
        refined = self.norm(xi + refined + self.ff_u_b)
        logits = self.weight_grn(xi.flatten(-2), context)
        weights = torch.softmax(logits, dim=-1)
        z = (weights.unsqueeze(-1) * refined).sum(dim=-2)
        return z, weights
