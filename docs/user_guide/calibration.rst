Calibration
===========

.. currentmodule:: xtft.evaluation

A model trained on one period can return quantiles whose coverage
differs from the nominal level on the next. The calibration layer
corrects this for the quantile forecasts of **any** model, XTFT or
not, from forecasts and outcomes of a validation period that
precedes the test period. It needs only NumPy arrays:

* ``y``: targets, shape ``(N, H)``;
* ``q``: quantile forecasts, shape ``(N, H, K)``, in the order of
  increasing levels ``quantiles``;
* ``times``: the integer origin of each forecast, shape ``(N,)``,
  for the adaptive methods.

The example :doc:`/auto_examples/plot_03_calibration` compares the
methods on a forecast whose noise grows during the test period.

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Class
     - Guarantees
     - Use when
   * - :class:`ConformalQuantileCalibrator`
     - per-step coverage, if validation and test errors are
       exchangeable
     - the error distribution is stable
   * - :class:`AdaptiveConformalCalibrator`
     - long-run per-step coverage, also under drift
     - errors drift during the test period
   * - :class:`SimultaneousConformalBand`
     - coverage of the whole path of :math:`H` steps
     - a decision depends on the whole horizon

Split-conformal quantile regression
-----------------------------------

For an interval with levels :math:`(\tau_\ell, \tau_u)` and
nominal coverage :math:`1 - \alpha^* = \tau_u - \tau_\ell`, the
score of a validation forecast at step :math:`h` is [1]_

.. math::

   E_h = \max\{\hat\ell_h - y_h,\ y_h - \hat u_h\}.

The margin :math:`\hat Q_h` is the
:math:`\lceil (n+1)(1 - \alpha^*) \rceil`-th smallest of the
:math:`n` validation scores of step :math:`h`, and the interval
becomes :math:`[\hat\ell_h - \hat Q_h,\ \hat u_h + \hat Q_h]`.
Every symmetric pair of levels is calibrated, and the quantiles
are sorted afterwards so that they stay ordered.

.. code-block:: python

   from xtft.evaluation import ConformalQuantileCalibrator

   cqr = ConformalQuantileCalibrator(quantiles, per_horizon=True)
   q_cal = cqr.fit(y_val, q_val).transform(q_test)

Adaptive conformal inference
----------------------------

Adaptive conformal inference (ACI) [2]_ replaces the fixed level
by a level :math:`\alpha_h` updated each time an outcome is
observed,

.. math::

   \alpha_h \leftarrow \alpha_h + \gamma\,(\alpha^* - \mathrm{err}),
   \qquad \mathrm{err} = \mathbb{1}\{E_h > \hat Q_h\},

and recomputes :math:`\hat Q_h` at each origin from the scores of
targets already observed. A miss widens the next intervals, a hit
narrows them. The update is causal: a target enters the margins
only for origins at which it has already been observed.

.. code-block:: python

   from xtft.evaluation import AdaptiveConformalCalibrator

   aci = AdaptiveConformalCalibrator(quantiles, gamma=0.005)
   q_cal = aci.fit(y_val, q_val).transform(y_test, q_test, times)

.. important::

   **Calibrate each series of a panel separately.** If the
   forecasts of many series share the same origins, a pooled
   calibrator receives one update per series at every origin, so
   its effective step size is :math:`\gamma` times the number of
   series and the intervals over-cover. Loop over the series:

   .. code-block:: python

      q_cal = np.empty_like(q_test)
      for s in np.unique(series_test):
          m, mv = series_test == s, series_val == s
          aci = AdaptiveConformalCalibrator(quantiles, gamma=0.005)
          aci.fit(y_val[mv], q_val[mv])
          q_cal[m] = aci.transform(y_test[m], q_test[m], times[m])

   In a study of 614 river basins, pooled ACI brought the 80%
   intervals to 95% coverage; per-basin ACI gave 80%.

Bands for whole paths
---------------------

A per-step 80% interval does not contain a whole path of
:math:`H` steps 80% of the time. :class:`SimultaneousConformalBand`
scores each validation path by its worst standardised exceedance,

.. math::

   S = \max_h \frac{\max\{\hat\ell_h - y_h,\ y_h - \hat u_h\}}{w_h},
   \qquad w_h = \max\{(\hat u_h - \hat\ell_h)/2,\ \epsilon\},

takes the conformal quantile :math:`\hat Q` of these scores, and
returns :math:`[\hat\ell_h - \hat Q w_h,\ \hat u_h + \hat Q w_h]`.
A new path then lies entirely in the band with probability at
least :math:`1 - \alpha` when validation and test paths are
exchangeable; scaling by :math:`w_h` makes the band wider where
the forecast is less certain [3]_. ``method="bonferroni"`` gives
the per-step CQR band at level :math:`\alpha / H` [4]_, as a
reference. :meth:`SimultaneousConformalBand.transform_online`
applies the ACI update to whole-path misses.

.. code-block:: python

   from xtft.evaluation import SimultaneousConformalBand, path_coverage

   band = SimultaneousConformalBand(quantiles, coverage=0.9, base=0.8)
   lower, upper = band.fit(y_val, q_val).transform_online(
       y_test, q_test, times, gamma=0.005
   )
   path_coverage(y_test, lower, upper)   # share of covered paths

``base`` chooses the quantile pair that gives the band its shape.

References
----------

.. [1] Romano, Y., Patterson, E. and Candès, E. (2019).
   Conformalized quantile regression. *Advances in Neural
   Information Processing Systems*, 32.
.. [2] Gibbs, I. and Candès, E. (2021). Adaptive conformal
   inference under distribution shift. *Advances in Neural
   Information Processing Systems*, 34.
.. [3] Diquigiovanni, J., Fontana, M. and Vantini, S. (2022).
   Conformal prediction bands for multivariate functional data.
   *Journal of Multivariate Analysis*, 189, 104879.
   https://doi.org/10.1016/j.jmva.2021.104879
.. [4] Stankevičiūtė, K., Alaa, A. M. and van der Schaar, M.
   (2021). Conformal time-series forecasting. *Advances in Neural
   Information Processing Systems*, 34.
