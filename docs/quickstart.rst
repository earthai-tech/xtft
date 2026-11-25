Quick start
===========

Forecast
--------

.. code-block:: python

   import torch
   from xtft.nn.torch.models import XTFT
   from xtft.nn.torch.probabilistic import pinball_loss

   model = XTFT(
       static_input_dim=4,
       dynamic_input_dim=5,
       future_input_dim=2,
       forecast_horizon=3,
       quantiles=[0.1, 0.5, 0.9],
       instance_norm=True,
   )
   q = model(x_static, x_dynamic, x_future)  # (batch, 3, 3)
   loss = pinball_loss(y, q, model.quantiles)

``x_static`` has shape ``(batch, n_static)``, ``x_dynamic``
``(batch, past_steps, n_dynamic)`` (missing values as ``NaN``) and
``x_future`` ``(batch, past_steps + horizon, n_future)``.
:func:`xtft.nn.torch.trainer.fit` trains with early stopping.

Calibrate
---------

The calibration layer works on the quantile forecasts of any model:

.. code-block:: python

   from xtft.evaluation import (
       AdaptiveConformalCalibrator,
       SimultaneousConformalBand,
       path_coverage,
   )

   aci = AdaptiveConformalCalibrator(quantiles, gamma=0.005)
   q_test_aci = aci.fit(y_val, q_val).transform(
       y_test, q_test, times=origin_times
   )

   band = SimultaneousConformalBand(quantiles, coverage=0.8)
   lower, upper = band.fit(y_val, q_val).transform_online(
       y_test, q_test, times=origin_times
   )
   path_coverage(y_test, lower, upper)
