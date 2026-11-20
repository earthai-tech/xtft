"""Gated residual building blocks (PyTorch).

Implements the gated linear unit (GLU) and the gated residual
network (GRN) of the Temporal Fusion Transformer [1]_, which XTFT
uses for every position-wise non-linear transform.

References
----------
.. [1] Lim, B., Arik, S. O., Loeff, N. and Pfister, T. (2021).
   Temporal Fusion Transformers for interpretable multi-horizon
   time series forecasting. International Journal of
   Forecasting, 37(4), 1748-1764.
   https://doi.org/10.1016/j.ijforecast.2021.03.012
.. [2] Dauphin, Y. N., Fan, A., Auli, M. and Grangier, D. (2017).
   Language modeling with gated convolutional networks.
   Proceedings of the 34th International Conference on Machine
   Learning (ICML), PMLR 70, 933-941.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["GatedLinearUnit", "GatedResidualNetwork"]


class GatedLinearUnit(nn.Module):
    r"""Gated linear unit with optional dropout.

    Parameters
    ----------
    in_features : int
        Size :math:`d_{in}` of the last input dimension.
    out_features : int
        Size :math:`d_{out}` of the last output dimension.
    dropout : float, default=0.0
        Dropout probability applied to the input, in ``[0, 1)``.

    Notes
    -----
    For an input :math:`\gamma \in \mathbb{R}^{d_{in}}`,

    .. math::

        \mathrm{GLU}(\gamma) = \sigma(W_4 \gamma + b_4)
        \odot (W_5 \gamma + b_5),

    where :math:`\sigma` is the logistic sigmoid and :math:`\odot`
    the element-wise product [2]_. The sigmoid gate lets the
    network suppress a whole branch, which is how XTFT can switch
    off components that do not help on a given dataset.

    Examples
    --------
    >>> import torch
    >>> from xtft.nn.torch.layers import GatedLinearUnit
    >>> glu = GatedLinearUnit(8, 4)
    >>> glu(torch.zeros(2, 5, 8)).shape
    torch.Size([2, 5, 4])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # One matmul for value and gate halves.
        self.proj = nn.Linear(in_features, 2 * out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the gate.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(..., in_features)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(..., out_features)``.
        """
        value, gate = self.proj(self.dropout(x)).chunk(2, dim=-1)
        return torch.sigmoid(gate) * value


class GatedResidualNetwork(nn.Module):
    r"""Gated residual network with optional context.

    Parameters
    ----------
    in_features : int
        Size of the last input dimension.
    hidden_features : int
        Width :math:`d_h` of the hidden layer.
    out_features : int or None, default=None
        Size of the output. ``None`` keeps `in_features`. When it
        differs from `in_features`, the skip path is a linear
        projection.
    context_features : int or None, default=None
        Size of an optional context vector :math:`c`. ``None``
        disables the context path.
    dropout : float, default=0.0
        Dropout probability inside the gate, in ``[0, 1)``.

    Notes
    -----
    Following [1]_, for input :math:`a` and context :math:`c`,

    .. math::

        \eta_2 &= \mathrm{ELU}(W_2 a + W_3 c + b_2), \\
        \eta_1 &= W_1 \eta_2 + b_1, \\
        \mathrm{GRN}(a, c) &= \mathrm{LayerNorm}\bigl(
            S a + \mathrm{GLU}(\eta_1)\bigr),

    where :math:`S` is the identity when the input and output
    sizes agree and a learned projection otherwise. A context of
    shape ``(B, C)`` is broadcast over the time axis of a
    ``(B, T, d)`` input.

    Examples
    --------
    >>> import torch
    >>> from xtft.nn.torch.layers import GatedResidualNetwork
    >>> grn = GatedResidualNetwork(6, 16, 4, context_features=3)
    >>> grn(torch.zeros(2, 7, 6), torch.zeros(2, 3)).shape
    torch.Size([2, 7, 4])
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        context_features: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.skip = (
            nn.Identity()
            if out_features == in_features
            else nn.Linear(in_features, out_features)
        )
        self.fc_in = nn.Linear(in_features, hidden_features)
        self.fc_ctx = (
            nn.Linear(context_features, hidden_features, bias=False)
            if context_features
            else None
        )
        self.fc_hidden = nn.Linear(hidden_features, hidden_features)
        self.glu = GatedLinearUnit(
            hidden_features, out_features, dropout
        )
        self.norm = nn.LayerNorm(out_features)

    def forward(
        self, x: Tensor, context: Tensor | None = None
    ) -> Tensor:
        """Apply the network.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(..., in_features)``.
        context : torch.Tensor or None, optional
            Context of shape ``(B, context_features)``, broadcast
            over the intermediate axes of `x`. Ignored when the
            network was built without a context path.

        Returns
        -------
        torch.Tensor
            Output of shape ``(..., out_features)``.

        Raises
        ------
        ValueError
            If a context is given but the network has no context
            path.
        """
        hidden = self.fc_in(x)
        if context is not None:
            if self.fc_ctx is None:
                raise ValueError(
                    "This GatedResidualNetwork was built without "
                    "context_features; pass context=None."
                )
            ctx = self.fc_ctx(context)
            # (B, d_h) -> (B, 1, ..., 1, d_h) to broadcast over time.
            while ctx.dim() < hidden.dim():
                ctx = ctx.unsqueeze(1)
            hidden = hidden + ctx
        hidden = self.fc_hidden(nn.functional.elu(hidden))
        return self.norm(self.skip(x) + self.glu(hidden))
