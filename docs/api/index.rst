API reference
=============

The public API in four parts: the model and its training loop,
the layers it is built from, the data utilities, and the
framework-agnostic evaluation and calibration functions.

.. contents:: On this page
   :local:
   :depth: 1

Model and training
------------------

.. currentmodule:: xtft.nn.torch

.. autosummary::
   :toctree: generated/
   :nosignatures:

   models.XTFT
   trainer.TrainConfig
   trainer.fit
   trainer.predict

Probabilistic output
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   probabilistic.NonCrossingQuantileHead
   probabilistic.QuantileLoss
   probabilistic.pinball_loss
   probabilistic.validate_quantiles

Layers
------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   layers.GatedResidualNetwork
   layers.GatedLinearUnit
   layers.VariableSelection
   layers.LocalWindowAttention
   layers.CrossScaleAttention
   layers.MultiResolutionEncoder
   layers.AntiAliasedDecimator
   layers.windowed_sinc_lowpass
   layers.MissingValueDecay
   layers.observation_features

Data
----

.. currentmodule:: xtft.data

.. autosummary::
   :toctree: generated/
   :nosignatures:

   PanelSpec
   PanelData
   WindowDataset
   rolling_origins
   add_calendar_features
   dominant_periods

Calibration
-----------

.. currentmodule:: xtft.evaluation

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ConformalQuantileCalibrator
   AdaptiveConformalCalibrator
   SimultaneousConformalBand
   path_coverage
   conformal_margin

Metrics
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   evaluate_quantile_forecast
   pinball
   crps_from_quantiles
   coverage
   mean_width
   interval_score
   interval_pairs
   calibration_curve
   mae
   rmse
   mase
   seasonal_scale

Statistical tests
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   diebold_mariano
   DieboldMarianoResult
   friedman_nemenyi
   FriedmanNemenyiResult

Backends
--------

.. currentmodule:: xtft

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_backend
   set_backend
   use_backend
   available_backends
