Package layout
==============

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Content
   * - :mod:`xtft.nn.torch`
     - the reference PyTorch XTFT: ``models``, ``layers``,
       ``probabilistic`` (quantile head and loss) and ``trainer``
   * - :mod:`xtft.data`
     - panels in long format, splits, scaling, windows, calendar
       features and seasonal periods
   * - :mod:`xtft.evaluation`
     - metrics, conformal calibration, whole-path bands and
       statistical tests; NumPy and SciPy only
   * - :mod:`xtft.backends`
     - choice of the deep-learning framework
   * - ``xtft.nn``
     - models resolved for the active backend
   * - ``xtft.nn.tf``
     - Keras implementation of XTFT and TFT
   * - ``xtft.compat``, ``xtft.core``, ``xtft.utils``,
       ``xtft.api``, ``xtft.metrics``
     - support code of the Keras models

The modules in the first four rows are the maintained public API
and are documented in the :doc:`/api/index`. The Keras code is
tested by the Keras backend tests and excluded from the coverage
figure.

Design rules
------------

* The PyTorch path never imports TensorFlow or Keras, and
  :mod:`xtft.evaluation` and :mod:`xtft.data` import neither
  framework, so they can score any model.
* Every model in the benchmarks is scored and calibrated by the
  same functions of :mod:`xtft.evaluation`.
* Randomness is seeded explicitly (``TrainConfig.seed``,
  ``numpy.random.Generator``); nothing depends on global state set
  at import.
