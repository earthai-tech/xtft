"""Monotone (non-crossing) quantile output head (PyTorch)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

__all__ = ["NonCrossingQuantileHead", "validate_quantiles"]


def validate_quantiles(
    quantiles: Sequence[float],
) -> tuple[float, ...]:
    r"""Check quantile levels and return them as a tuple.

    Parameters
    ----------
    quantiles : sequence of float
        Levels :math:`\tau_k`, strictly increasing, each in
        ``(0, 1)``.

    Returns
    -------
    tuple of float
        The validated levels.

    Raises
    ------
    ValueError
        If the sequence is empty, not strictly increasing, or has
        a level outside ``(0, 1)``.

    Examples
    --------
    >>> from xtft.nn.torch.probabilistic import validate_quantiles
    >>> validate_quantiles([0.1, 0.5, 0.9])
    (0.1, 0.5, 0.9)
    """
    levels = tuple(float(q) for q in quantiles)
    if not levels:
        raise ValueError("quantiles must not be empty.")
    if any(not 0.0 < q < 1.0 for q in levels):
        raise ValueError(
            f"quantiles must lie in (0, 1), got {levels}."
        )
    if any(
        b <= a for a, b in zip(levels[:-1], levels[1:], strict=True)
    ):
        raise ValueError(
            f"quantiles must be strictly increasing, got {levels}."
        )
    return levels


class NonCrossingQuantileHead(nn.Module):
    r"""Project features to quantiles that cannot cross.

    Parameters
    ----------
    in_features : int
        Feature width :math:`d`.
    output_dim : int, default=1
        Number :math:`O` of target variables.
    quantiles : sequence of float or None, default=None
        Strictly increasing levels in ``(0, 1)``. ``None`` gives a
        point forecast.

    Notes
    -----
    A linear map gives raw values :math:`r_1, \dots, r_K` per
    target. With anchor index :math:`m` (the level 0.5 if present,
    otherwise :math:`\lfloor K/2 \rfloor` in zero-based order),

    .. math::

        \hat q_m = r_m, \qquad
        \hat q_{k} = \hat q_{k-1} + \mathrm{softplus}(r_k)
            \ (k > m), \qquad
        \hat q_{k} = \hat q_{k+1} - \mathrm{softplus}(r_k)
            \ (k < m).

    Because :math:`\mathrm{softplus} \ge 0`, the quantiles are
    non-decreasing in :math:`\tau` for every input (strictly
    increasing in exact arithmetic; for extreme raw values the
    softplus underflows to zero and two levels tie), so predictive
    intervals are always valid. Independent heads, as in the
    original XTFT and TFT, can cross.

    Examples
    --------
    >>> import torch
    >>> from xtft.nn.torch.probabilistic import (
    ...     NonCrossingQuantileHead)
    >>> head = NonCrossingQuantileHead(8, quantiles=[0.1, 0.5, 0.9])
    >>> q = head(torch.randn(4, 3, 8))
    >>> q.shape
    torch.Size([4, 3, 1, 3])
    >>> bool((q.diff(dim=-1) >= 0).all())
    True
    """

    def __init__(
        self,
        in_features: int,
        output_dim: int = 1,
        quantiles: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError(
                f"output_dim must be >= 1, got {output_dim}."
            )
        self.output_dim = output_dim
        self.quantiles = (
            None
            if quantiles is None
            else validate_quantiles(quantiles)
        )
        n_q = 1 if self.quantiles is None else len(self.quantiles)
        self.n_quantiles = n_q
        if self.quantiles is None:
            self.anchor = 0
        elif 0.5 in self.quantiles:
            self.anchor = self.quantiles.index(0.5)
        else:
            self.anchor = n_q // 2
        self.proj = nn.Linear(in_features, output_dim * n_q)

    def forward(self, x: Tensor) -> Tensor:
        """Compute quantiles.

        Parameters
        ----------
        x : torch.Tensor
            Features of shape ``(..., d)``.

        Returns
        -------
        torch.Tensor
            ``(..., O, K)`` with quantiles, or ``(..., O)`` for a
            point forecast.
        """
        raw = self.proj(x)
        if self.quantiles is None:
            return raw
        raw = raw.unflatten(-1, (self.output_dim, self.n_quantiles))
        m = self.anchor
        centre = raw[..., m : m + 1]
        steps = nn.functional.softplus(raw)
        upper = centre + steps[..., m + 1 :].cumsum(dim=-1)
        lower = centre - steps[..., :m].flip(-1).cumsum(dim=-1)
        return torch.cat([lower.flip(-1), centre, upper], dim=-1)
