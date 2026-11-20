"""Cross-scale Temporal Fusion Transformer (XTFT), PyTorch.

XTFT forecasts :math:`H` future steps of :math:`O` targets for
many related series that share three kinds of covariates: static
attributes, observed past inputs and inputs known in advance. It
keeps the gated, variable-selecting design of the Temporal Fusion
Transformer (TFT) [1]_ and changes how the history is encoded:

* the history is encoded at several resolutions, each obtained by
  causal anti-aliasing and origin-aligned decimation
  (:class:`~xtft.nn.torch.layers.MultiResolutionEncoder`) [2]_;
* attention is hierarchical: causal attention with a learned soft
  window inside each resolution
  (:class:`~xtft.nn.torch.layers.LocalWindowAttention`), then
  attention from every forecast step across all resolutions
  (:class:`~xtft.nn.torch.layers.CrossScaleAttention`) [3]_;
* quantiles come from a monotone head
  (:class:`~xtft.nn.torch.probabilistic.NonCrossingQuantileHead`),
  so predictive intervals never cross.

References
----------
.. [1] Lim, B., Arik, S. O., Loeff, N. and Pfister, T. (2021).
   Temporal Fusion Transformers for interpretable multi-horizon
   time series forecasting. International Journal of
   Forecasting, 37(4), 1748-1764.
   https://doi.org/10.1016/j.ijforecast.2021.03.012
.. [2] Oppenheim, A. V. and Schafer, R. W. (2010). Discrete-Time
   Signal Processing, 3rd ed. Pearson.
.. [3] Press, O., Smith, N. A. and Lewis, M. (2022). Train short,
   test long: Attention with linear biases enables input length
   extrapolation. ICLR.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from ..layers import (
    CrossScaleAttention,
    GatedLinearUnit,
    GatedResidualNetwork,
    LocalWindowAttention,
    MissingValueDecay,
    MultiResolutionEncoder,
    VariableSelection,
)
from ..probabilistic import NonCrossingQuantileHead

__all__ = ["XTFT"]

# Variance floor of the instance normalisation (flat windows).
_NORM_EPS = 1e-5


def _window_stats(target: Tensor) -> tuple[Tensor, Tensor]:
    """Mean and scale over observed steps, shape ``(B, 1, O)``.

    Non-finite entries are ignored; a window with no observed value
    gets mean 0 and scale 1 (the standardised defaults).
    """
    observed = torch.isfinite(target)
    count = observed.sum(dim=1, keepdim=True)
    safe = count.clamp(min=1).to(target.dtype)
    values = torch.where(observed, target, torch.zeros_like(target))
    mu = values.sum(dim=1, keepdim=True) / safe
    centred = torch.where(
        observed, target - mu, torch.zeros_like(target)
    )
    var = (centred**2).sum(dim=1, keepdim=True) / safe
    sigma = torch.sqrt(var + _NORM_EPS)
    empty = count == 0
    mu = torch.where(empty, torch.zeros_like(mu), mu)
    sigma = torch.where(empty, torch.ones_like(sigma), sigma)
    return mu, sigma


def _shape_of(x: Tensor | None) -> tuple[int, ...] | None:
    """Return ``tuple(x.shape)``, or ``None`` if `x` is missing."""
    return None if x is None else tuple(x.shape)


class _GatedSkip(nn.Module):
    """``LayerNorm(skip + GLU(x))``, the TFT gated residual."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.glu = GatedLinearUnit(d_model, d_model, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        return self.norm(skip + self.glu(x))


class XTFT(nn.Module):
    r"""Cross-scale Temporal Fusion Transformer.

    Parameters
    ----------
    static_input_dim : int
        Number :math:`S \ge 0` of static variables; 0 disables the
        static path.
    dynamic_input_dim : int
        Number :math:`D \ge 1` of observed past variables.
    future_input_dim : int
        Number :math:`F \ge 0` of known future variables; 0
        disables the known-input path.
    forecast_horizon : int
        Number :math:`H \ge 1` of steps to forecast.
    output_dim : int, default=1
        Number :math:`O \ge 1` of target variables.
    quantiles : sequence of float or None, default=None
        Strictly increasing levels in ``(0, 1)``; ``None`` gives a
        point forecast.
    hidden_units : int, default=64
        Model width :math:`d`; divisible by `num_heads`.
    num_heads : int, default=4
        Attention heads in both attention stages.
    scales : sequence of int, default=(1, 2, 4)
        Strictly increasing decimation factors of the history,
        starting with 1. ``(1,)`` removes the multi-resolution
        component.
    cell : {"lstm", "gru"}, default="lstm"
        Recurrent cell of the encoders and the decoder.
    antialias : bool, default=True
        Low-pass filter before decimation (ablation switch).
    align : {"origin", "start"}, default="origin"
        Grid anchor of the coarse scales (ablation switch;
        ``"start"`` reproduces ``x[:, ::s]``).
    local_attention : bool, default=True
        Use windowed attention inside each resolution.
    local_window : float, default=8.0
        Largest initial window of the local attention, in steps of
        each resolution.
    cross_window : float, default=16.0
        Largest initial recency window of the cross-scale
        attention, in original steps.
    dropout : float, default=0.1
        Dropout probability, in ``[0, 1)``.
    instance_norm : bool, default=False
        Standardise the target history of each window by its own
        mean and standard deviation and map the forecast back
        (reversible instance normalisation [1]_). Requires the
        first `output_dim` channels of `x_dynamic` to be the past
        targets. Use it for non-stationary series whose level
        drifts outside the training range.
    missing : bool, default=True
        Accept ``NaN`` for missing values in `x_dynamic` and
        handle them natively: learned decay imputation [2]_ and
        missingness-aware variable embeddings (see Notes). Missing
        static or known inputs are set to 0 (the standardised
        mean). With ``False``, ``NaN`` inputs give ``NaN`` outputs.
    season : int or None, default=None
        Seasonal lag, in steps, of the gap filling: a long gap
        follows the value one season earlier
        (:class:`~xtft.nn.torch.layers.MissingValueDecay`). Only
        used with `missing`; :func:`xtft.data.dominant_periods`
        estimates it from the training data.

    Attributes
    ----------
    quantiles : tuple of float or None
        The validated quantile levels.

    Raises
    ------
    ValueError
        If a dimension, the horizon, the width/head combination or
        the dropout rate is invalid.

    See Also
    --------
    xtft.nn.torch.probabilistic.QuantileLoss : Training loss.
    xtft.nn.torch.layers.MultiResolutionEncoder : History encoder.

    Notes
    -----
    Let :math:`x^{s} \in \mathbb{R}^S`,
    :math:`x^{p}_{t} \in \mathbb{R}^{D}` and
    :math:`x^{f}_{t} \in \mathbb{R}^{F}` be the static, past and
    known inputs, :math:`t = 0, \dots, T-1` the history and
    :math:`\tau = 1, \dots, H` the forecast steps. The forward pass
    is

    .. math::

        (z^{s}, v^{s}) &= \mathrm{VSN}_s(x^{s}), \quad
        c_{\mathrm{sel}}, c_{\mathrm{enr}}, c_{\mathrm{st}}
            = \mathrm{GRN}_{\{\cdot\}}(z^{s}), \\
        (z_t, v_t) &= \mathrm{VSN}_p([x^{p}_t, x^{f}_t];
            c_{\mathrm{sel}}), \\
        e^{(s)} &= \mathrm{LWA}_s\bigl(\mathrm{RNN}_s(
            D_s(h_s * z); c_{\mathrm{st}})\bigr),
            \quad s \in \mathcal{S}, \\
        u_\tau &= \mathrm{GRN}_e\bigl(\mathrm{RNN}_{d}(
            \mathrm{VSN}_f(x^{f}_{T-1+\tau}) + p_\tau);
            c_{\mathrm{enr}}\bigr), \\
        (g_\tau, \pi_\tau) &= \mathrm{CSA}\bigl(u_\tau,
            \{e^{(s)}\}_{s \in \mathcal{S}}\bigr), \\
        \hat q_{\tau} &= \mathrm{Head}\bigl(\mathrm{LayerNorm}(
            u_\tau + \mathrm{GLU}(\mathrm{GRN}_o(g_\tau)))\bigr),

    where :math:`p_\tau` is a learned horizon embedding, the
    decoder RNN starts from the final state of the finest
    encoder, LWA and CSA are the local and cross-scale attention
    stages, and gated skip connections around the recurrent layers
    are omitted from the display. :math:`v^{s}, v_t` (and the
    analogous future weights) are the variable importances and
    :math:`\pi_\tau \in \Delta^{|\mathcal{S}|-1}` the scale
    attribution. Missing static or known inputs remove the
    corresponding terms.

    The forecast is direct (all :math:`H` steps in one pass), so
    errors do not accumulate through recursion. For dyadic scales
    the cost per series is
    :math:`O\bigl(T d^2 + d\sum_s (T/s)^2 + H d T + H d^2\bigr)`.

    Inputs and outputs follow the fusionlab-learn XTFT: the known
    inputs cover the history and the horizon, shape
    ``(B, T + H, F)``; a horizon-only array ``(B, H, F)`` is also
    accepted. Missing inputs are passed as ``NaN`` (handled when
    `missing` is true, see below).

    With `instance_norm`, the target channels :math:`y_t \in
    \mathbb{R}^{O}` of the history are replaced by
    :math:`(y_t - \mu) / \sigma` with :math:`\mu` and
    :math:`\sigma^2 + \epsilon` the per-window mean and variance
    over :math:`t = 0, \dots, T-1` (:math:`\epsilon = 10^{-5}`),
    and the output becomes :math:`\mu + \sigma \hat q_\tau`. The
    map is increasing, so the quantiles still do not cross; the
    loss is computed on the restored scale.

    With `missing`, the observation mask :math:`m_t` and the steps
    since the last observation :math:`\delta_t` of every past
    variable are derived from the ``NaN`` pattern
    (:func:`~xtft.nn.torch.layers.observation_features`). The mean
    and variance of the instance normalisation are taken over
    observed steps only (:math:`\mu = 0, \sigma = 1` for a window
    without observed target). Missing entries are then imputed as
    :math:`\gamma_t x^{\mathrm{last}}_t` with the learned decay
    :math:`\gamma_t = \exp(-\max(0, w \delta_t + b))`
    (:class:`~xtft.nn.torch.layers.MissingValueDecay`), and the
    past variable embeddings receive :math:`(1 - m_t) u +
    \log(1 + \delta_t) s`, so the selection weights can use
    missingness itself. On complete data these terms vanish.
    Missing targets in the loss are skipped by
    :func:`~xtft.nn.torch.probabilistic.pinball_loss`.

    References
    ----------
    .. [1] Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H. and
       Choo, J. (2022). Reversible instance normalization for
       accurate time-series forecasting against distribution
       shift. *International Conference on Learning
       Representations (ICLR)*.
    .. [2] Che, Z., Purushotham, S., Cho, K., Sontag, D. and
       Liu, Y. (2018). Recurrent neural networks for multivariate
       time series with missing values. *Scientific Reports*, 8,
       6085. https://doi.org/10.1038/s41598-018-24271-9

    Examples
    --------
    >>> import torch
    >>> from xtft.nn.torch.models import XTFT
    >>> _ = torch.manual_seed(0)
    >>> model = XTFT(static_input_dim=3, dynamic_input_dim=4,
    ...              future_input_dim=2, forecast_horizon=5,
    ...              quantiles=[0.1, 0.5, 0.9], hidden_units=16)
    >>> xs = torch.randn(8, 3)
    >>> xp = torch.randn(8, 24, 4)
    >>> xf = torch.randn(8, 24 + 5, 2)
    >>> model(xs, xp, xf).shape
    torch.Size([8, 5, 3])
    >>> out, info = model(xs, xp, xf, return_details=True)
    >>> sorted(info)  # doctest: +NORMALIZE_WHITESPACE
    ['cross_windows', 'future_weights', 'local_windows',
     'past_weights', 'scale_attribution', 'static_weights']
    >>> info["scale_attribution"].shape
    torch.Size([8, 5, 3])
    """

    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        forecast_horizon: int,
        *,
        output_dim: int = 1,
        quantiles: Sequence[float] | None = None,
        hidden_units: int = 64,
        num_heads: int = 4,
        scales: Sequence[int] = (1, 2, 4),
        cell: Literal["lstm", "gru"] = "lstm",
        antialias: bool = True,
        align: Literal["origin", "start"] = "origin",
        local_attention: bool = True,
        local_window: float = 8.0,
        cross_window: float = 16.0,
        dropout: float = 0.1,
        instance_norm: bool = False,
        missing: bool = True,
        season: int | None = None,
    ) -> None:
        super().__init__()
        for name, value, low in (
            ("static_input_dim", static_input_dim, 0),
            ("dynamic_input_dim", dynamic_input_dim, 1),
            ("future_input_dim", future_input_dim, 0),
            ("forecast_horizon", forecast_horizon, 1),
            ("hidden_units", hidden_units, 1),
            ("num_heads", num_heads, 1),
        ):
            if not isinstance(value, int) or value < low:
                raise ValueError(
                    f"{name} must be an int >= {low}, got {value!r}."
                )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout}."
            )
        if not scales or tuple(scales)[0] != 1:
            # The finest scale feeds the gated skip and starts the
            # decoder, so it must keep the full time resolution.
            raise ValueError(
                f"scales must start with 1, got {tuple(scales)}."
            )
        if hidden_units % num_heads:
            raise ValueError(
                f"hidden_units={hidden_units} is not divisible by "
                f"num_heads={num_heads}."
            )
        if instance_norm and output_dim > dynamic_input_dim:
            raise ValueError(
                "instance_norm needs the output_dim target channels "
                "at the start of x_dynamic, but output_dim="
                f"{output_dim} > dynamic_input_dim="
                f"{dynamic_input_dim}."
            )
        d = hidden_units
        self.instance_norm = bool(instance_norm)
        self.missing = bool(missing)
        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.future_input_dim = future_input_dim
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        self.scales = tuple(scales)
        self.cell = cell

        ctx = d if static_input_dim else None
        if static_input_dim:
            self.static_vsn = VariableSelection(
                static_input_dim, d, dropout=dropout
            )
            self.ctx_select = GatedResidualNetwork(
                d, d, dropout=dropout
            )
            self.ctx_enrich = GatedResidualNetwork(
                d, d, dropout=dropout
            )
            self.ctx_state = GatedResidualNetwork(
                d, d, dropout=dropout
            )
        if self.missing:
            self.missing_decay = MissingValueDecay(
                dynamic_input_dim, season=season
            )
        self.past_vsn = VariableSelection(
            dynamic_input_dim + future_input_dim,
            d,
            context_features=ctx,
            dropout=dropout,
            missing_aware=self.missing,
        )
        if future_input_dim:
            self.future_vsn = VariableSelection(
                future_input_dim,
                d,
                context_features=ctx,
                dropout=dropout,
            )
        self.horizon_embedding = nn.Parameter(
            torch.zeros(forecast_horizon, d)
        )
        self.encoder = MultiResolutionEncoder(
            d,
            self.scales,
            cell=cell,
            context_features=ctx,
            antialias=antialias,
            align=align,
        )
        self.encoder_skip = _GatedSkip(d, dropout)
        self.local_attention = (
            nn.ModuleList(
                LocalWindowAttention(
                    d, num_heads, local_window, dropout
                )
                for _ in self.scales
            )
            if local_attention
            else None
        )
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.decoder = rnn_cls(d, d, batch_first=True)
        self.decoder_skip = _GatedSkip(d, dropout)
        self.enrich = GatedResidualNetwork(
            d, d, context_features=ctx, dropout=dropout
        )
        self.cross_scale = CrossScaleAttention(
            d, self.scales, num_heads, cross_window, dropout
        )
        self.output_grn = GatedResidualNetwork(d, d, dropout=dropout)
        self.output_skip = _GatedSkip(d, dropout)
        self.head = NonCrossingQuantileHead(d, output_dim, quantiles)
        self.quantiles = self.head.quantiles

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.

        Parameters
        ----------
        trainable_only : bool, default=True
            Count only parameters with ``requires_grad``.

        Returns
        -------
        int
            Number of scalar parameters.
        """
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )

    def _split_inputs(self, x_static, x_dynamic, x_future):
        if x_dynamic is None and isinstance(x_static, (list, tuple)):
            if len(x_static) != 3:
                raise ValueError(
                    "Pass (x_static, x_dynamic, x_future) or a "
                    "sequence of these three tensors."
                )
            x_static, x_dynamic, x_future = x_static
        if x_dynamic is None or x_dynamic.dim() != 3:
            raise ValueError("x_dynamic must have shape (B, T, D).")
        batch, steps, dyn = x_dynamic.shape
        if dyn != self.dynamic_input_dim:
            raise ValueError(
                f"x_dynamic has {dyn} variables, expected "
                f"{self.dynamic_input_dim}."
            )
        if self.static_input_dim:
            if x_static is None or tuple(x_static.shape) != (
                batch,
                self.static_input_dim,
            ):
                raise ValueError(
                    "x_static must have shape "
                    f"({batch}, {self.static_input_dim}), got "
                    f"{_shape_of(x_static)}."
                )
        past_known = future_known = None
        if self.future_input_dim:
            h = self.forecast_horizon
            if (
                x_future is None
                or x_future.dim() != 3
                or x_future.shape[0] != batch
                or x_future.shape[2] != self.future_input_dim
                or x_future.shape[1] not in (h, steps + h)
            ):
                raise ValueError(
                    "x_future must have shape "
                    f"({batch}, {steps + h} or {h}, "
                    f"{self.future_input_dim}), got "
                    f"{_shape_of(x_future)}."
                )
            future_known = x_future[:, -h:]
            if x_future.shape[1] == steps + h:
                past_known = x_future[:, :steps]
            else:
                # Known inputs absent from the history: zero-fill the
                # past slots so the past VSN keeps a fixed width.
                past_known = x_dynamic.new_zeros(
                    batch, steps, self.future_input_dim
                )
        return x_static, x_dynamic, past_known, future_known

    def forward(
        self,
        x_static: Tensor | Sequence[Tensor] | None,
        x_dynamic: Tensor | None = None,
        x_future: Tensor | None = None,
        *,
        return_details: bool = False,
    ):
        """Forecast the next `forecast_horizon` steps.

        Parameters
        ----------
        x_static : torch.Tensor or None
            Static inputs ``(B, S)``; ``None`` when ``S == 0``. A
            sequence ``(x_static, x_dynamic, x_future)`` is also
            accepted as the only argument.
        x_dynamic : torch.Tensor
            Observed past inputs ``(B, T, D)``.
        x_future : torch.Tensor or None
            Known inputs ``(B, T + H, F)`` or ``(B, H, F)``;
            ``None`` when ``F == 0``.
        return_details : bool, default=False
            Also return the interpretability outputs.

        Returns
        -------
        forecast : torch.Tensor
            ``(B, H, K)`` for quantiles with ``O == 1``,
            ``(B, H, O, K)`` for quantiles with ``O > 1`` and
            ``(B, H, O)`` for point forecasts.
        details : dict of torch.Tensor
            Only if `return_details`: ``static_weights`` ``(B, S)``
            or ``None``, ``past_weights`` ``(B, T, D + F)``,
            ``future_weights`` ``(B, H, F)`` or ``None``,
            ``scale_attribution`` ``(B, H, len(scales))``,
            ``local_windows`` (list of ``(num_heads,)`` per scale,
            empty without local attention) and ``cross_windows``
            ``(num_heads,)``.

        Raises
        ------
        ValueError
            If an input has the wrong shape.
        """
        x_static, x_dynamic, past_known, future_known = (
            self._split_inputs(x_static, x_dynamic, x_future)
        )
        if self.missing:
            x_static, past_known, future_known = (
                None if a is None else torch.nan_to_num(a, nan=0.0)
                for a in (x_static, past_known, future_known)
            )
        batch = x_dynamic.shape[0]
        c_sel = c_enr = c_state = static_w = None
        if self.static_input_dim:
            z_static, static_w = self.static_vsn(x_static)
            c_sel = self.ctx_select(z_static)
            c_enr = self.ctx_enrich(z_static)
            c_state = self.ctx_state(z_static)

        if self.instance_norm:
            o = self.output_dim
            target = x_dynamic[..., :o]
            mu, sigma = _window_stats(target)  # (B, 1, O) each
            x_dynamic = torch.cat(
                [(target - mu) / sigma, x_dynamic[..., o:]], dim=-1
            )

        mask = delta = None
        if self.missing:
            x_dynamic, mask, delta = self.missing_decay(x_dynamic)
            if past_known is not None:
                mask = torch.cat(
                    [mask, torch.ones_like(past_known)], dim=-1
                )
                delta = torch.cat(
                    [delta, torch.zeros_like(past_known)], dim=-1
                )

        past = x_dynamic
        if past_known is not None:
            past = torch.cat([x_dynamic, past_known], dim=-1)
        z_past, past_w = self.past_vsn(
            past, c_sel, mask=mask, delta=delta
        )

        memories, state = self.encoder(z_past, c_state)
        # Gated skip from the selected inputs to the finest scale.
        memories[0] = self.encoder_skip(memories[0], z_past)
        if self.local_attention is not None:
            memories = [
                att(m)
                for att, m in zip(
                    self.local_attention, memories, strict=True
                )
            ]

        future_w = None
        horizon = self.horizon_embedding.expand(batch, -1, -1)
        if future_known is not None:
            z_future, future_w = self.future_vsn(future_known, c_sel)
            tokens = z_future + horizon
        else:
            tokens = horizon
        decoded, _ = self.decoder(tokens, state)
        decoded = self.decoder_skip(decoded, tokens)
        enriched = self.enrich(decoded, c_enr)
        fused, scale_attr = self.cross_scale(enriched, memories)
        out = self.output_skip(self.output_grn(fused), decoded)
        forecast = self.head(out)  # (B, H, O, K) or (B, H, O)
        if self.instance_norm:
            if self.quantiles is not None:
                mu, sigma = mu[..., None], sigma[..., None]
            forecast = mu + sigma * forecast
        if self.quantiles is not None and self.output_dim == 1:
            forecast = forecast.squeeze(-2)

        if not return_details:
            return forecast
        details = {
            "static_weights": static_w,
            "past_weights": past_w,
            "future_weights": future_w,
            "scale_attribution": scale_attr,
            "local_windows": (
                [att.windows() for att in self.local_attention]
                if self.local_attention is not None
                else []
            ),
            "cross_windows": self.cross_scale.windows(),
        }
        return forecast, details
