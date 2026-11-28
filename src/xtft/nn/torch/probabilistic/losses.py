"""Training losses for quantile forecasts (PyTorch)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from .quantile_head import validate_quantiles

__all__ = ["QuantileLoss", "pinball_loss"]


def pinball_loss(
    y_true: Tensor,
    y_pred: Tensor,
    quantiles: Sequence[float],
    reduction: Literal["mean", "none"] = "mean",
) -> Tensor:
    r"""Pinball (quantile) loss, ignoring missing targets.

    Parameters
    ----------
    y_true : torch.Tensor
        Targets of shape ``(...,)``; non-finite values (NaN, inf)
        mark missing observations and are excluded.
    y_pred : torch.Tensor
        Quantile forecasts of shape ``(..., K)``, aligned with
        `y_true` on the leading axes.
    quantiles : sequence of float
        The :math:`K` levels of the last axis of `y_pred`.
    reduction : {"mean", "none"}, default="mean"
        ``"mean"`` averages over observed entries and levels;
        ``"none"`` returns the per-entry loss (zero where the
        target is missing).

    Returns
    -------
    torch.Tensor
        Scalar, or a tensor shaped like `y_pred`. A batch with no
        observed target gives ``0`` (with gradient), not NaN.

    Raises
    ------
    ValueError
        If the shapes do not match or the levels are invalid.

    Notes
    -----
    For residual :math:`u = y - \hat q_\tau`,

    .. math::

        \rho_\tau(u) = \max\bigl(\tau u,\, (\tau - 1) u\bigr),

    whose expectation is minimised by the :math:`\tau`-quantile of
    :math:`y` [1]_. The mean over observed entries is

    .. math::

        \mathcal{L} = \frac{1}{K\,|\Omega|} \sum_{i \in \Omega}
            \sum_{k=1}^{K} \rho_{\tau_k}(y_i - \hat q_{i,\tau_k}),

    with :math:`\Omega` the set of finite targets.

    References
    ----------
    .. [1] Koenker, R. and Bassett, G. (1978). Regression
       quantiles. Econometrica, 46(1), 33-50.
       https://doi.org/10.2307/1913643

    Examples
    --------
    >>> import torch
    >>> from xtft.nn import pinball_loss
    >>> y = torch.tensor([1.0, float("nan")])
    >>> q = torch.tensor([[0.0, 1.0, 2.0], [5.0, 5.0, 5.0]])
    >>> round(float(pinball_loss(y, q, [0.1, 0.5, 0.9])), 4)
    0.0667
    """
    levels = validate_quantiles(quantiles)
    if y_pred.shape[:-1] != y_true.shape:
        raise ValueError(
            f"y_pred shape {tuple(y_pred.shape)} does not match "
            f"y_true shape {tuple(y_true.shape)} plus a quantile axis."
        )
    if y_pred.shape[-1] != len(levels):
        raise ValueError(
            f"y_pred has {y_pred.shape[-1]} quantiles, expected "
            f"{len(levels)}."
        )
    tau = torch.tensor(
        levels, dtype=y_pred.dtype, device=y_pred.device
    )
    observed = torch.isfinite(y_true)
    target = torch.where(observed, y_true, torch.zeros_like(y_true))
    residual = target.unsqueeze(-1) - y_pred
    loss = torch.maximum(tau * residual, (tau - 1) * residual)
    loss = loss * observed.unsqueeze(-1).to(loss.dtype)
    if reduction == "none":
        return loss
    if reduction != "mean":
        raise ValueError(
            f"reduction must be 'mean' or 'none', got {reduction!r}."
        )
    count = observed.sum().clamp(min=1) * len(levels)
    return loss.sum() / count


class QuantileLoss(nn.Module):
    """Module wrapper of :func:`pinball_loss`.

    Parameters
    ----------
    quantiles : sequence of float
        Levels of the last axis of the predictions.

    See Also
    --------
    pinball_loss : The functional form and its definition.

    Examples
    --------
    >>> import torch
    >>> from xtft.nn import QuantileLoss
    >>> loss = QuantileLoss([0.5])
    >>> float(loss(torch.zeros(3), torch.ones(3, 1)))
    0.5
    """

    def __init__(self, quantiles: Sequence[float]) -> None:
        super().__init__()
        self.quantiles = validate_quantiles(quantiles)

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Return the mean pinball loss (see :func:`pinball_loss`)."""
        return pinball_loss(y_true, y_pred, self.quantiles)
