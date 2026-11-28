The model
=========

.. currentmodule:: xtft.nn.torch.models

:class:`XTFT` keeps the covariate-handling structure of the
Temporal Fusion Transformer (TFT) [1]_: gated residual networks,
variable selection for each group of inputs, static contexts that
condition the rest of the network, a recurrent encoder and
decoder, and attention from the horizon to the history. It changes
that structure where robustness, cost or calibration require.

What differs from TFT
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 36 44

   * - Component
     - TFT
     - XTFT
   * - Level of the series
     - fixed scaling with training statistics
     - optional reversible normalisation of each window, on
       observed steps (``instance_norm``)
   * - Missing inputs
     - imputed before training
     - masks, gap lengths and a learned decay; missingness enters
       the embeddings (``missing``)
   * - Variable selection
     - a gated network per variable
     - batched per-variable networks, :math:`O(V d^2)` per step
   * - History attention
     - one masked self-attention over all :math:`T + H` positions
     - causal attention over the history with a learned recency
       window per head (``local_attention``)
   * - Horizon to history
     - inside the same self-attention
     - decoder states query the history with a learned recency
       bias
   * - Attention cost
     - :math:`O\bigl((T + H)^2 d\bigr)`
     - :math:`O(T^2 d + H T d)`
   * - Quantiles
     - one linear output per level, may cross
     - monotone head, never crossing
   * - Calibration
     - none
     - CQR, ACI and whole-path bands in :mod:`xtft.evaluation`,
       for any model

Building blocks
---------------

The gated residual network (GRN), for an input :math:`a` and an
optional context :math:`c`, computes

.. math::

   \mathrm{GRN}(a, c) = \mathrm{LayerNorm}\bigl(P a +
   \mathrm{GLU}(W_1 \eta + b_1)\bigr), \qquad
   \eta = \mathrm{ELU}(W_2 a + W_3 c + b_2),

with the gated linear unit :math:`\mathrm{GLU}(\gamma) =
\sigma(W_4 \gamma + b_4) \odot (W_5 \gamma + b_5)` and a projection
:math:`P` (the identity when sizes agree). The sigmoid gate can
close a whole branch, so the network can switch off what does not
help.

A variable selection network (VSN) embeds each scalar input
:math:`x_j` of a group of :math:`V` variables as
:math:`\xi_j = w_j x_j + b_j \in \mathbb{R}^d`, refines each
embedding with its own small network, and mixes them with softmax
weights computed from all embeddings and the static context:

.. math::

   v = \mathrm{softmax}\bigl(\mathrm{GRN}_v([\xi_1, \dots,
   \xi_V], c)\bigr), \qquad z = \sum_{j=1}^{V} v_j\, \tilde\xi_j .

The weights :math:`v` are returned by the forward pass (see
`Interpretability outputs`_).

Encoder, attention and decoder
------------------------------

An LSTM (or GRU, ``cell="gru"``) reads the selected past inputs
from an initial state given by the static context. Causal
multi-head attention with a learned recency window follows:

.. math::

   \omega^{(n)}_{ts} = \mathrm{softmax}_{s \le t}\Bigl(
   \frac{q_t^\top k_s}{\sqrt{d / n_a}} - \lambda_n (t - s)\Bigr),
   \qquad \lambda_n = \mathrm{softplus}(\rho_n) > 0,

so that, at equal content scores, the weight of a step decays by
a factor :math:`e` every :math:`1/\lambda_n` steps; the initial
windows are log-spaced up to ``local_window`` steps. A second LSTM
reads the known inputs of the horizon with a learned step
embedding, and each of its states queries the encoded history
with a recency bias whose initial windows reach
``cross_window`` steps. A monotone head turns the result into
quantiles for all :math:`H` steps at once:

.. math::

   \hat q_{h,m} = r_{h,m}, \qquad
   \hat q_{h,k} = \hat q_{h,k-1} + \mathrm{softplus}(r_{h,k})
   \ (k > m), \qquad
   \hat q_{h,k} = \hat q_{h,k+1} - \mathrm{softplus}(r_{h,k})
   \ (k < m),

where :math:`m` indexes the median. The forecast is direct, so
errors do not accumulate through recursion.

Inputs and outputs
------------------

.. code-block:: python

   from xtft.nn import XTFT

   model = XTFT(
       static_input_dim=S,        # 0 disables the static path
       dynamic_input_dim=O + D,   # past targets first
       future_input_dim=F,        # 0 disables the known path
       forecast_horizon=H,
       quantiles=[0.1, 0.5, 0.9], # None: point forecast
   )
   q = model(x_static, x_dynamic, x_future)

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Argument
     - Shape
     - Notes
   * - ``x_static``
     - ``(B, S)``
     - ``None`` when ``S == 0``
   * - ``x_dynamic``
     - ``(B, T, O + D)``
     - targets first; ``NaN`` for missing values
   * - ``x_future``
     - ``(B, T + H, F)`` or ``(B, H, F)``
     - ``None`` when ``F == 0``
   * - output
     - ``(B, H, K)``
     - ``(B, H, O, K)`` for several targets, ``(B, H, O)`` for a
       point forecast

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Option
     - Default
     - Effect
   * - ``hidden_units``
     - 64
     - model width :math:`d`, divisible by ``num_heads``
   * - ``num_heads``
     - 4
     - heads of both attention stages
   * - ``instance_norm``
     - ``False``
     - reversible normalisation of each window; use it when the
       level drifts (see :doc:`missing_data`)
   * - ``missing``
     - ``True``
     - native handling of ``NaN`` in the past inputs
   * - ``season``
     - ``None``
     - seasonal lag used to fill long gaps
   * - ``local_attention``
     - ``True``
     - windowed attention over the history
   * - ``local_window``, ``cross_window``
     - 8, 16
     - largest initial recency windows, in steps
   * - ``scales``
     - ``(1,)``
     - add coarser views of the history, e.g. ``(1, 2, 4)``, for
       the anti-aliased multi-resolution encoder
   * - ``cell``
     - ``"lstm"``
     - recurrent cell, ``"lstm"`` or ``"gru"``
   * - ``dropout``
     - 0.1
     - dropout probability

.. note::

   The multi-resolution encoder (``scales`` with more than one
   entry) neither improved nor degraded accuracy in our ablations
   on short and long inputs, so the single-scale encoder is the
   default.

Interpretability outputs
------------------------

``return_details=True`` also returns a dictionary:

.. code-block:: python

   q, details = model(x_static, x_dynamic, x_future,
                      return_details=True)
   details["past_weights"]      # (B, T, D + F) selection weights
   details["future_weights"]    # (B, H, F)
   details["static_weights"]    # (B, S)
   details["local_windows"]     # learned windows per head
   details["cross_windows"]

The selection weights show how the network routes its inputs.
They are not a test of whether an input is relevant: in a
synthetic check, a covariate of pure noise kept a weight close to
that of an informative one. To measure what a covariate
contributes, train without it and compare the scores.

Cost
----

For dyadic scales the cost per series and forward pass is
:math:`O\bigl(T d^2 + d \sum_s (T/s)^2 + H d T + H d^2\bigr)`; with
the default single scale, the attention costs
:math:`O(T^2 d + H T d)` against :math:`O\bigl((T + H)^2 d\bigr)`
for TFT. :meth:`XTFT.num_parameters` counts the parameters.

References
----------

.. [1] Lim, B., Arık, S. Ö., Loeff, N. and Pfister, T. (2021).
   Temporal Fusion Transformers for interpretable multi-horizon
   time series forecasting. *International Journal of
   Forecasting*, 37(4), 1748-1764.
   https://doi.org/10.1016/j.ijforecast.2021.03.012
