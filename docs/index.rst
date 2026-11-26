xtft
====

Covariate-aware probabilistic forecasting with calibrated,
shift-robust uncertainty.

XTFT, the eXogenous-aware Temporal Fusion Transformer, is a
temporal fusion network for panels of related series that
share static attributes, observed past inputs and inputs known in
advance. It normalises each window against level shift, handles
missing past inputs natively and returns non-crossing quantiles. A
model-agnostic calibration layer adjusts per-step intervals
(split-conformal and adaptive conformal) and builds bands that
cover whole forecast paths.

.. toctree::
   :maxdepth: 2

   install
   quickstart
   api
