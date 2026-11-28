Data
====

.. currentmodule:: xtft.data

A forecasting *panel* is a set of related series observed on the
same regular time grid. xtft reads it from a data frame in long
format and turns it into windows that every model in the package
consumes the same way.

Column roles
------------

:class:`PanelSpec` assigns each column one role:

.. list-table::
   :header-rows: 1
   :widths: 14 40 46

   * - Role
     - Meaning
     - Examples
   * - ``target``
     - What is forecast; one or more columns.
     - sales, price, river flow
   * - ``static``
     - Constant within a series, numeric.
     - store size, catchment area, aridity
   * - ``past``
     - Observed only up to the forecast origin.
     - realised temperature, past demand of other products
   * - ``known``
     - Known over the forecast horizon.
     - calendar, planned prices, published load forecasts

The past targets are always added to the past inputs, so they do
not need to be listed in ``past``. A column cannot have two
roles.

.. code-block:: python

   from xtft.data import PanelData, PanelSpec

   spec = PanelSpec(
       target=("y",),
       static=("area", "aridity"),
       past=("precip_obs",),
       known=("dayofyear_sin", "dayofyear_cos"),
   )
   panel = PanelData.from_frame(df, spec)

:meth:`PanelData.from_frame` checks that every series is sorted,
regularly sampled and has constant static values. Gaps must be
present as rows with ``NaN`` values; a missing row is an irregular
step and raises an error.

Chronological splits and scaling
--------------------------------

Splits are made in time, per series.
:meth:`PanelData.split_index` returns, for every series, the
index of the first step at or after a cutoff;
:meth:`PanelData.fit_scaling` then standardises each series with
the mean and standard deviation of its own training period:

.. math::

   \tilde x_{t,c} = \frac{x_{t,c} - \mu_{i,c}}{\sigma_{i,c}},

where :math:`\mu_{i,c}` and :math:`\sigma_{i,c}` are taken over
the training steps of series :math:`i` for column :math:`c`
(``NaN`` skipped). Static columns are standardised across series.
:meth:`PanelData.inverse_target` maps forecasts back to the
original units.

Windows
-------

A window of series :math:`i` at origin :math:`t_0` holds the
history :math:`t_0 - T + 1, \dots, t_0` and the targets
:math:`t_0 + 1, \dots, t_0 + H`. :meth:`PanelData.origins` lists
the admissible origins of a period, and :class:`WindowDataset`
builds each window when it is requested:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Key
     - Shape
     - Content
   * - ``static``
     - ``(S,)``
     - static attributes
   * - ``past``
     - ``(T, O + D)``
     - targets, then observed covariates, over the history
   * - ``future``
     - ``(T + H, F)``
     - known covariates over history and horizon
   * - ``y``
     - ``(H, O)``
     - targets of the horizon; ``NaN`` where unobserved

The items work with :class:`torch.utils.data.DataLoader` and its
default collate function. Missing inputs are replaced by
``fill_value`` (0, the training mean after scaling) unless
``fill_value=None``, which keeps them as ``NaN`` for a model that
handles them itself; see :doc:`missing_data`.

.. tip::

   The stride of the origins controls the overlap of the windows:
   ``stride=1`` for training (every origin), ``stride=H`` for a
   test period scored once per step.

Calendar features and seasonal periods
--------------------------------------

:func:`add_calendar_features` adds cyclical encodings
:math:`(\sin 2\pi c/P, \cos 2\pi c/P)` of the hour, day of week,
day of month, day of year, month or minute, so that the last and
first value of a cycle are neighbours. Use them as ``known``
inputs.

:func:`dominant_periods` estimates the strongest periodicities of
one or several series from a periodogram with a significance test
on the peaks. Its first period is a good value for the ``season``
option of XTFT, which fills long gaps from the value one season
earlier.

.. code-block:: python

   from xtft.data import dominant_periods

   periods = dominant_periods(panel.target, max_period=400)
   season = periods[0] if periods else None   # in steps

See also
--------

* :doc:`/getting_started/first_forecast` for the whole pipeline.
* :mod:`xtft.data` in the :doc:`API reference </api/index>`.
