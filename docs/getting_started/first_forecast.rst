First forecast
==============

This page walks from a data frame in long format to calibrated
quantile forecasts. The complete, runnable script, with its
synthetic data and figure, is the example
:doc:`/auto_examples/plot_01_quickstart`.

1. Describe the columns
-----------------------

xtft reads panels in *long format*: one row per series and time
step, with an identifier column, a time column, the target and
the covariates. :class:`~xtft.data.PanelSpec` gives each column
its role.

.. code-block:: python

   from xtft.data import PanelData, PanelSpec, add_calendar_features

   df, calendar = add_calendar_features(df, features=["dayofweek"])
   spec = PanelSpec(
       id_col="unique_id",       # series identifier
       time_col="ds",            # regular time steps
       target=("y",),
       static=("level",),        # constant within a series
       past=("temp",),           # observed up to the origin only
       known=calendar,           # known over the horizon
   )
   panel = PanelData.from_frame(df, spec)

Each series must be regularly sampled. Gaps are rows with ``NaN``
values, not missing rows.

2. Split in time and scale
--------------------------

Periods are split chronologically, and the scaling statistics of
every series come from its training period only, so nothing of
the validation or test period leaks into the inputs.

.. code-block:: python

   import pandas as pd

   # Per-series index of the first validation and test steps.
   val_start = panel.split_index(pd.Timestamp("2023-03-04"))
   test_start = panel.split_index(pd.Timestamp("2023-05-03"))
   panel.fit_scaling(val_start)

``split_index`` takes a value of the time column's type: a
timestamp for dates, an integer for integer steps.

3. Cut forecasting windows
--------------------------

A window has a history of ``T`` steps and a horizon of ``H``
steps. :meth:`~xtft.data.PanelData.origins` lists the origins
whose targets fall in a period, and
:class:`~xtft.data.WindowDataset` cuts the windows lazily.

.. code-block:: python

   import numpy as np
   from xtft.data import WindowDataset

   T, H = 56, 14
   first = np.zeros(panel.n_series, dtype=np.int64)
   train_idx = panel.origins(first, val_start, T, H, stride=1)
   val_idx = panel.origins(val_start - 1, test_start, T, H, stride=1)
   test_idx = panel.origins(test_start - 1, None, T, H, stride=H)

   train_ds = WindowDataset(panel, train_idx, T, H)
   val_ds = WindowDataset(panel, val_idx, T, H)
   test_ds = WindowDataset(panel, test_idx, T, H)

The past inputs of a window hold the targets first, then the
observed covariates, so ``dynamic_input_dim`` is the number of
targets plus the number of ``past`` columns.

4. Build and train the model
----------------------------

.. code-block:: python

   import torch
   from xtft.nn import XTFT, TrainConfig, fit

   quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
   torch.manual_seed(0)
   model = XTFT(
       static_input_dim=1,
       dynamic_input_dim=2,            # y and temp
       future_input_dim=len(calendar),
       forecast_horizon=H,
       quantiles=quantiles,
       instance_norm=True,
   )
   history = fit(
       model, train_ds, val_ds, quantiles,
       TrainConfig(max_epochs=50, batch_size=64, patience=5),
   )

:func:`~xtft.nn.torch.trainer.fit` minimises the pinball loss
with Adam, stops early on the validation loss and restores the
best weights.

5. Forecast and return to original units
----------------------------------------

.. code-block:: python

   from xtft.nn import predict

   q = predict(model, test_ds)                  # (N, H, 1, K), scaled
   q = panel.inverse_target(q, test_idx[:, 0])[:, :, 0]

6. Calibrate and score
----------------------

Quantiles learned on the training period are often too narrow or
too wide on new data. Fit a calibrator on the validation
forecasts and apply it to the test forecasts, then score them.

.. code-block:: python

   from xtft.evaluation import (
       AdaptiveConformalCalibrator,
       evaluate_quantile_forecast,
   )

   def targets(ds):
       y = np.stack([ds[n]["y"] for n in range(len(ds))])
       return panel.inverse_target(y, ds.index[:, 0])[:, :, 0]

   q_val = predict(model, val_ds)
   q_val = panel.inverse_target(q_val, val_idx[:, 0])[:, :, 0]
   y_val, y_test = targets(val_ds), targets(test_ds)

   aci = AdaptiveConformalCalibrator(quantiles).fit(y_val, q_val)
   q_cal = aci.transform(y_test, q, times=test_idx[:, 1])
   evaluate_quantile_forecast(y_test, q_cal, quantiles)

For a panel, run the adaptive calibrator separately for each
series; see :doc:`/user_guide/calibration`.

Next steps
----------

* :doc:`/user_guide/model`: what each option of
  :class:`~xtft.nn.torch.models.XTFT` does, with the equations.
* :doc:`/user_guide/missing_data`: gaps and level shift.
* :doc:`/auto_examples/index`: runnable scripts.
