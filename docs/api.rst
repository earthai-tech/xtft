API reference
=============

Model
-----

.. currentmodule:: xtft.nn.torch

.. autosummary::
   :toctree: generated/

   models.XTFT
   trainer.TrainConfig
   trainer.fit
   trainer.predict

Layers
------

.. autosummary::
   :toctree: generated/

   layers.VariableSelection
   layers.GatedResidualNetwork
   layers.GatedLinearUnit
   layers.LocalWindowAttention
   layers.CrossScaleAttention
   layers.MissingValueDecay
   layers.MultiResolutionEncoder
   layers.observation_features
   probabilistic.NonCrossingQuantileHead
   probabilistic.QuantileLoss
   probabilistic.pinball_loss

Calibration and evaluation
--------------------------

.. currentmodule:: xtft.evaluation

.. autosummary::
   :toctree: generated/

   ConformalQuantileCalibrator
   AdaptiveConformalCalibrator
   SimultaneousConformalBand
   path_coverage
   conformal_margin
   evaluate_quantile_forecast
   crps_from_quantiles
   pinball
   coverage
   interval_score
   calibration_curve
   diebold_mariano
   friedman_nemenyi

Data
----

.. currentmodule:: xtft.data

.. autosummary::
   :toctree: generated/

   PanelData
   PanelSpec
   WindowDataset
   rolling_origins
   add_calendar_features
   dominant_periods
