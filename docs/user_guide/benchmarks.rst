Benchmarks
==========

The ``benchmarks/`` folder of the repository compares XTFT with
other forecasters under one protocol. It is not installed with the
package: run it from a clone, with the baselines installed.

.. code-block:: bash

   git clone https://github.com/earthai-tech/xtft.git
   cd xtft
   pip install -e ".[torch,bench]"

Protocol
--------

Every learned model gets the same input length, horizon, rolling
forecast origins and optimisation budget (optimiser steps and
early-stopping patience). Periods are chronological (train,
validation, test), each configuration runs with several seeds, and
all forecasts are calibrated and scored with
:mod:`xtft.evaluation`.

.. code-block:: bash

   python benchmarks/run.py --dataset ETTh1 --model xtft --horizon 96
   python benchmarks/run.py --dataset ETTh1 --model nf:PatchTST --horizon 96
   python benchmarks/aggregate.py results --metric crps_q --reference xtft

``run.py`` fits one (data set, model, horizon, seed) cell,
forecasts at the shared origins, calibrates (split-conformal and
adaptive) and writes a JSON file of scores, the per-window losses
and the forecasts. ``aggregate.py`` builds the tables, average
ranks, Friedman/Nemenyi tests and Diebold-Mariano tests against a
reference model. ``recalibrate.py`` re-scores saved forecasts with
other calibration settings without refitting.

Models
------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - ``--model``
     - Forecaster
   * - ``xtft``
     - XTFT (PyTorch)
   * - ``nf:<Model>``
     - any neuralforecast model, e.g. ``nf:TFT``, ``nf:PatchTST``,
       ``nf:NBEATSx``, ``nf:DLinear``, ``nf:LSTM``
   * - ``sf:<Model>``
     - any statsforecast model, e.g. ``sf:AutoETS``,
       ``sf:SeasonalNaive``
   * - ``lgb``
     - global LightGBM quantile model with lag and calendar
       features
   * - ``lear``
     - LEAR, the LASSO-estimated autoregressive model for
       day-ahead electricity prices

Data sets
---------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``--dataset``
     - Content
   * - ``EPF-NP``, ``EPF-PJM``, ``EPF-BE``, ``EPF-FR``, ``EPF-DE``
     - day-ahead electricity prices with two day-ahead load or
       generation forecasts as known covariates (open benchmark of
       Lago et al., 2021)
   * - ``M5-StoreDept``
     - M5 retail sales aggregated to 70 store-department series,
       with prices, SNAP days, events and static attributes
   * - ``ETTh1``, ``ETTh2``, ``ETTm1``, ``ETTm2``, ``Electricity``,
       ``Traffic``, ``Exchange``, ``ILI``, ``Weather``
     - long-horizon univariate benchmarks (via datasetsforecast)
   * - ``CAMELS-US``, ``CAMELS-US-Weather``
     - daily streamflow of 614 US basins (Caravan) with ERA5-Land
       forcings and 16 catchment attributes; the second variant
       adds observed future weather as known inputs
   * - ``Synthetic``
     - small series for smoke tests

Data are downloaded on first use into ``data/raw`` (or the folder
named by ``XTFT_DATA``).

Options
-------

.. dropdown:: Missing inputs

   ``--missing-rate 0.2 --missing-mode random`` (or ``blocks``)
   removes target values; the models train on the gapped series
   and are scored on the complete one.

.. dropdown:: Updating during the test period

   ``--model-kw '{"refit_every": 672, "refit_mode": "finetune_es"}'``
   updates XTFT every 672 steps with ``finetune``,
   ``finetune_es`` or ``retrain`` (see :doc:`training`). LEAR
   takes ``recalibrate_every`` in days.

.. dropdown:: Model settings and variants

   ``--model-kw`` passes a JSON dictionary to the model, for
   example ``'{"hidden_units": 128}'``, and ``--tag`` labels the
   variant in the results.

.. dropdown:: Clusters

   ``make_tasks.py`` writes one line per run for a SLURM job
   array; ``jobs/`` holds the batch scripts.
