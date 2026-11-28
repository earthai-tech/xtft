Evaluation
==========

.. currentmodule:: xtft.evaluation

Every model in the benchmarks, XTFT and baselines alike, is scored
by the same NumPy functions, so differences between models cannot
come from different metric code. The example
:doc:`/auto_examples/plot_04_compare_forecasters` uses all of them.

Scores
------

:func:`evaluate_quantile_forecast` returns the standard set for a
quantile forecast in one dictionary:

.. code-block:: python

   from xtft.evaluation import evaluate_quantile_forecast, seasonal_scale

   scale = seasonal_scale(y_train, season=7)   # for MASE and MSIS
   scores = evaluate_quantile_forecast(y_test, q_test, quantiles, scale)
   scores["crps_q"], scores["coverage_80"], scores["winkler_80"]

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Key
     - Meaning
   * - ``pinball``
     - mean pinball loss over levels and targets
   * - ``crps_q``
     - continuous ranked probability score approximated from the
       quantiles (see below)
   * - ``calibration_error``
     - mean :math:`|\hat F(\tau) - \tau|` over the levels
   * - ``mae``, ``rmse``, ``mase``
     - errors of the median (``mase`` needs ``scale``)
   * - ``coverage_c``
     - share of targets inside the central interval of nominal
       coverage :math:`c` percent
   * - ``width_c``, ``pinaw_c``
     - mean width of that interval, raw and normalised by the
       range of the targets
   * - ``winkler_c``, ``msis_c``
     - interval score [1]_, raw and scaled (``msis_c`` needs
       ``scale``)

Missing targets are skipped everywhere. With :math:`K` levels
:math:`\tau_k`, the quantile CRPS is twice the mean pinball loss,

.. math::

   \mathrm{CRPS}_Q = \frac{2}{K} \sum_{k=1}^{K}
   \rho_{\tau_k}\bigl(y - \hat q_k\bigr),

which approximates the CRPS of the predictive distribution when
the levels cover :math:`(0, 1)` densely [1]_.
:func:`calibration_curve` returns the observed frequency
:math:`\hat F(\tau_k)` of targets below each quantile, for
reliability diagrams.

Two forecasters: Diebold-Mariano
--------------------------------

:func:`diebold_mariano` tests whether two forecasters have the
same expected loss [2]_, with the small-sample correction of
Harvey, Leybourne and Newbold [3]_. Pass one loss per forecast
origin, in time order, for example the CRPS of each window
averaged over the series that share the origin:

.. code-block:: python

   from xtft.evaluation import diebold_mariano

   res = diebold_mariano(crps_a, crps_b, horizon=1)
   res.statistic, res.pvalue, res.mean_difference

``horizon`` is the overlap of consecutive forecasts: the forecast
horizon divided by the origin stride, rounded up (1 for windows
that do not overlap). A negative statistic means that the first
forecaster has the lower loss.

Many forecasters: Friedman and Nemenyi
--------------------------------------

:func:`friedman_nemenyi` ranks :math:`k` models over :math:`N`
tasks (rows: data sets and horizons; columns: models; lower is
better), tests that all average ranks are equal, and returns the
Nemenyi critical difference: two models whose average ranks
differ by less than it are not significantly different at level
``alpha`` [4]_.

.. code-block:: python

   from xtft.evaluation import friedman_nemenyi

   res = friedman_nemenyi(scores, alpha=0.05)   # scores: (N, k)
   res.average_ranks, res.pvalue, res.critical_difference

References
----------

.. [1] Gneiting, T. and Raftery, A. E. (2007). Strictly proper
   scoring rules, prediction, and estimation. *Journal of the
   American Statistical Association*, 102(477), 359-378.
   https://doi.org/10.1198/016214506000001437
.. [2] Diebold, F. X. and Mariano, R. S. (1995). Comparing
   predictive accuracy. *Journal of Business & Economic
   Statistics*, 13(3), 253-263.
   https://doi.org/10.1080/07350015.1995.10524599
.. [3] Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing
   the equality of prediction mean squared errors.
   *International Journal of Forecasting*, 13(2), 281-291.
   https://doi.org/10.1016/S0169-2070(96)00719-4
.. [4] Demšar, J. (2006). Statistical comparisons of classifiers
   over multiple data sets. *Journal of Machine Learning
   Research*, 7, 1-30.
