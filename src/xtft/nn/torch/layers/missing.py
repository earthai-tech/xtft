r"""Native handling of missing inputs (PyTorch).

Real series have gaps: sensors drop out, surveys skip epochs,
markets close. Most deep forecasters require complete inputs, so
users impute first and the model cannot tell an observed value from
a guess. Here a missing value is given as ``NaN`` and the model
derives, for every variable and time step, whether it was observed
and how long ago it was last observed, then imputes with a learned,
per-variable decay toward the mean [1]_. The mask and the elapsed
time are also passed to the variable embeddings
(:class:`~xtft.nn.torch.layers.VariableSelection`), so the model
can learn when missingness itself is informative.

References
----------
.. [1] Che, Z., Purushotham, S., Cho, K., Sontag, D. and Liu, Y.
   (2018). Recurrent neural networks for multivariate time series
   with missing values. Scientific Reports, 8, 6085.
   https://doi.org/10.1038/s41598-018-24271-9
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["MissingValueDecay", "observation_features"]


def observation_features(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    r"""Mask, last observed value and elapsed time of a sequence.

    Parameters
    ----------
    x : torch.Tensor
        Inputs of shape ``(B, T, V)``; missing values are ``NaN``
        (infinite values are treated as missing too).

    Returns
    -------
    mask : torch.Tensor
        ``(B, T, V)``, 1 where observed, 0 where missing (dtype of
        `x`).
    last : torch.Tensor
        ``(B, T, V)``, the most recent observed value at or before
        each step; 0 before the first observation.
    delta : torch.Tensor
        ``(B, T, V)``, steps since the last observation (0 when
        observed); :math:`t + 1` at step :math:`t` (0-based) if
        nothing has been observed yet.

    Notes
    -----
    With :math:`m_t \in \{0, 1\}` the mask and
    :math:`\ell_t = \max\{s \le t : m_s = 1\}` the index of the last
    observation, ``last`` is :math:`x_{\ell_t}` and ``delta`` is
    :math:`t - \ell_t`. Both are computed without a loop over time
    (cumulative maximum of the observed indices), in
    :math:`O(B T V)`.

    Examples
    --------
    >>> import torch
    >>> from xtft.nn.torch.layers.missing import observation_features
    >>> nan = float("nan")
    >>> x = torch.tensor([[[nan], [2.0], [nan], [nan], [5.0]]])
    >>> m, last, delta = observation_features(x)
    >>> m[0, :, 0].tolist()
    [0.0, 1.0, 0.0, 0.0, 1.0]
    >>> last[0, :, 0].tolist()
    [0.0, 2.0, 2.0, 2.0, 5.0]
    >>> delta[0, :, 0].tolist()
    [1.0, 0.0, 1.0, 2.0, 0.0]
    """
    if x.dim() != 3:
        raise ValueError(
            f"x must have shape (B, T, V), got {x.shape}."
        )
    observed = torch.isfinite(x)
    steps = torch.arange(x.shape[1], device=x.device).view(1, -1, 1)
    steps = steps.expand_as(x)
    last_idx = torch.where(observed, steps, torch.full_like(steps, -1))
    last_idx = torch.cummax(last_idx, dim=1).values
    seen = last_idx >= 0
    filled = torch.where(observed, x, torch.zeros_like(x))
    last = filled.gather(1, last_idx.clamp(min=0))
    last = torch.where(seen, last, torch.zeros_like(last))
    delta = torch.where(seen, steps - last_idx, steps + 1)
    return observed.to(x.dtype), last, delta.to(x.dtype)


class MissingValueDecay(nn.Module):
    r"""Impute missing inputs by a learned decay toward the mean.

    Parameters
    ----------
    num_variables : int
        Number :math:`V \ge 1` of input variables.

    Notes
    -----
    For variable :math:`j` with mask :math:`m_t`, last observed
    value :math:`x^{\mathrm{last}}_t` and elapsed time
    :math:`\delta_t` (see :func:`observation_features`), the input
    becomes

    .. math::

        \gamma_t &= \exp\bigl(-\max(0,\; w_j \delta_t + b_j)\bigr),
        \\
        \tilde x_t &= m_t x_t + (1 - m_t)\bigl(\gamma_t
            x^{\mathrm{last}}_t + (1 - \gamma_t)\, \bar x_j\bigr),

    the input decay of GRU-D [1]_, with :math:`\bar x_j = 0`
    because XTFT's inputs are standardised (0 is the training mean,
    or the window mean under instance normalisation). A recently
    observed value is carried forward; a stale one fades to the
    mean at a rate learned per variable (:math:`w_j \ge 0` is not
    enforced, but :math:`\gamma_t \le 1` always holds). Observed
    entries are returned unchanged.

    Examples
    --------
    >>> import torch
    >>> from xtft.nn.torch.layers.missing import MissingValueDecay
    >>> nan = float("nan")
    >>> x = torch.tensor([[[1.0], [nan], [nan]]])
    >>> x_hat, mask, delta = MissingValueDecay(1)(x)
    >>> x_hat[0, 0, 0].item(), bool(torch.isfinite(x_hat).all())
    (1.0, True)
    >>> bool(x_hat[0, 1, 0] >= x_hat[0, 2, 0] >= 0)
    True
    """

    def __init__(self, num_variables: int) -> None:
        super().__init__()
        if num_variables < 1:
            raise ValueError(
                f"num_variables must be >= 1, got {num_variables}."
            )
        self.num_variables = num_variables
        # w = 0.1, b = 0: half-life of about 7 steps at start.
        self.weight = nn.Parameter(torch.full((num_variables,), 0.1))
        self.bias = nn.Parameter(torch.zeros(num_variables))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Impute `x` and return it with its mask and elapsed time.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, T, V)`` with ``NaN`` for missing values.

        Returns
        -------
        x_hat : torch.Tensor
            Imputed inputs, same shape, all finite.
        mask, delta : torch.Tensor
            As returned by :func:`observation_features`.

        Raises
        ------
        ValueError
            If `x` is not 3-D with `num_variables` variables.
        """
        if x.dim() != 3 or x.shape[-1] != self.num_variables:
            raise ValueError(
                f"x must have shape (B, T, {self.num_variables}), "
                f"got {tuple(x.shape)}."
            )
        mask, last, delta = observation_features(x)
        gamma = torch.exp(-torch.relu(self.weight * delta + self.bias))
        imputed = gamma * last
        x_obs = torch.where(mask.bool(), x, torch.zeros_like(x))
        x_hat = mask * x_obs + (1.0 - mask) * imputed
        return x_hat, mask, delta
