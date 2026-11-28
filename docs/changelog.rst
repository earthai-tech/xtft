Changelog
=========

0.1.0 (unreleased)
------------------

First public version.

* PyTorch XTFT with reversible instance normalisation, native
  handling of missing past inputs, batched variable selection,
  causal attention with learned recency windows, decoder-to-history
  attention and a non-crossing quantile head.
* Training loop with early stopping, random-batch epochs, CPU, CUDA
  and ``mps`` devices and mixed precision.
* Panel utilities: column roles, chronological splits, scaling,
  lazy windows, calendar features, seasonal periods.
* Model-agnostic calibration: split-conformal quantile regression,
  adaptive conformal inference and whole-path bands.
* Metrics, Diebold-Mariano and Friedman/Nemenyi tests.
* Keras XTFT and TFT, behind the
  ``tensorflow`` backend.
* Benchmark scripts for electricity prices, retail sales,
  long-horizon series and CAMELS-US streamflow.
