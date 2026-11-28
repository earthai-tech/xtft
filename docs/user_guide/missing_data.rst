Missing data and level shift
============================

Two properties of real panels break models trained on clean,
stationary data: gaps in the inputs, and a level that moves
outside the range seen in training. XTFT handles both inside the
network. The example :doc:`/auto_examples/plot_02_missing_data` shows
them at work.

Gaps as NaN
-----------

With ``missing=True`` (the default), XTFT accepts ``NaN`` in the
past inputs. Build the windows with ``fill_value=None`` so that
the gaps reach the model:

.. code-block:: python

   from xtft.data import WindowDataset

   train_ds = WindowDataset(panel, idx, T, H, fill_value=None)

For each past variable :math:`j` the model derives the
observation indicator :math:`m_{t,j} \in \{0, 1\}` and the number
of steps :math:`\delta_{t,j}` since the last observation. A
missing entry is replaced by
:math:`\gamma_{t,j}\, x^{\mathrm{last}}_{t,j}`, the last observed
value shrunk towards the standardised mean by a learned decay [1]_,

.. math::

   \gamma_{t,j} = \exp\bigl(-\max\{0,\ w_j \delta_{t,j} + b_j\}\bigr),

and the embedding of the variable receives two learned terms,

.. math::

   \xi_{t,j} = w_j\, \check x_{t,j} + b_j
   + (1 - m_{t,j})\, u_j + \log(1 + \delta_{t,j})\, s_j ,

so that the variable selection can depend on missingness itself:
an input missing for a long time can be down-weighted. On complete
data these terms vanish. With ``season`` set, long gaps follow the
value one season earlier instead of decaying to the mean.

Missing *targets* in the horizon are simply skipped by the loss
and by the metrics of :mod:`xtft.evaluation`. Missing static or
known inputs are set to 0, the standardised mean.

.. warning::

   With ``missing=False``, a ``NaN`` input gives a ``NaN``
   forecast. Fill the gaps upstream, or keep the default.

Level shift
-----------

With ``instance_norm=True``, the target history of every window
is standardised by its own statistics and the output is mapped
back (reversible instance normalisation [2]_):

.. math::

   \tilde y_t = \frac{y_t - \mu}{\sigma}, \qquad
   \hat q^{\,\mathrm{out}}_{h,k} = \mu + \sigma\, \hat q_{h,k},

with :math:`\mu` and :math:`\sigma^2 + \epsilon` the mean and
variance of the observed history targets
(:math:`\epsilon = 10^{-5}`). The back-transformation is
increasing, so the quantiles stay ordered. The network learns the
shape of the series relative to its recent level, and a shift of
the level after training moves the forecast with it.

.. tip::

   Use ``instance_norm=True`` when series drift (prices, demand
   with trends, flows after a change in a catchment). On strictly
   stationary series it can discard useful information about the
   absolute level; compare both settings on the validation period.

References
----------

.. [1] Che, Z., Purushotham, S., Cho, K., Sontag, D. and Liu, Y.
   (2018). Recurrent neural networks for multivariate time series
   with missing values. *Scientific Reports*, 8, 6085.
   https://doi.org/10.1038/s41598-018-24271-9
.. [2] Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H. and Choo,
   J. (2022). Reversible instance normalization for accurate
   time-series forecasting against distribution shift.
   *International Conference on Learning Representations*.
