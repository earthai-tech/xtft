User guide
==========

The guide follows the life of a forecast: preparing the panel,
the model and its options, training, missing data and level
shift, calibration of the intervals, and the evaluation of the
result. Each page links to the API reference for the details.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Data
      :link: data
      :link-type: doc
      :class-card: xtft-nav-card

      Column roles, chronological splits, scaling, windows,
      calendar features and seasonal periods.

   .. grid-item-card:: The model
      :link: model
      :link-type: doc
      :class-card: xtft-nav-card

      Architecture, equations, what is new relative to TFT, and
      every option.

   .. grid-item-card:: Training
      :link: training
      :link-type: doc
      :class-card: xtft-nav-card

      Loss, early stopping, devices, mixed precision and
      reproducibility.

   .. grid-item-card:: Missing data and level shift
      :link: missing_data
      :link-type: doc
      :class-card: xtft-nav-card

      Gaps as ``NaN`` and reversible instance normalisation.

   .. grid-item-card:: Calibration
      :link: calibration
      :link-type: doc
      :class-card: xtft-nav-card

      Split-conformal, adaptive conformal and whole-path bands,
      for any model.

   .. grid-item-card:: Evaluation
      :link: evaluation
      :link-type: doc
      :class-card: xtft-nav-card

      Metrics, the Diebold-Mariano test and Friedman/Nemenyi
      ranks.

   .. grid-item-card:: Benchmarks
      :link: benchmarks
      :link-type: doc
      :class-card: xtft-nav-card

      Reproduce the comparisons with other forecasters.

.. toctree::
   :hidden:

   data
   model
   training
   missing_data
   calibration
   evaluation
   benchmarks
