:html_theme.sidebar_secondary.remove:

.. title:: xtft: covariate-aware probabilistic forecasting

.. raw:: html

   <section class="xh-hero">
     <div class="xh-hero-text">
       <img class="xh-wordmark only-light"
            src="_static/logo/xtft-logo.svg" alt="XTFT"/>
       <img class="xh-wordmark only-dark"
            src="_static/logo/xtft-logo-dark.svg" alt="XTFT"/>
       <p class="xh-kicker">eXogenous-aware Temporal Fusion Transformer</p>
       <h1 class="xh-headline">Forecast whole panels,
         <span>with intervals you can trust.</span></h1>
       <p class="xh-lede">Static attributes, observed covariates and
         inputs known in advance, fused in one network that survives
         level shifts and missing data, followed by a calibration
         layer that works for any model.</p>
       <div class="xh-actions">
         <a class="xh-btn xh-btn-primary"
            href="getting_started/index.html">Get started</a>
         <a class="xh-btn" href="auto_examples/index.html">Examples</a>
         <a class="xh-btn"
            href="https://github.com/earthai-tech/xtft">GitHub</a>
       </div>
       <div class="xh-install"><span class="xh-prompt">$</span>
         <code>pip install "xtft[torch]"</code></div>
     </div>
     <div class="xh-hero-art">

.. raw:: html
   :file: _static/hero.html

.. raw:: html

     </div>
   </section>

   <section class="xh-flow" aria-label="Workflow">
     <a class="xh-step" href="user_guide/data.html">
       <span class="xh-num">1</span><strong>Panel</strong>
       <span>long-format data, column roles, windows</span></a>
     <a class="xh-step" href="user_guide/model.html">
       <span class="xh-num">2</span><strong>XTFT</strong>
       <span>non-crossing quantiles for every step</span></a>
     <a class="xh-step" href="user_guide/calibration.html">
       <span class="xh-num">3</span><strong>Calibrate</strong>
       <span>conformal intervals and whole-path bands</span></a>
     <a class="xh-step" href="user_guide/evaluation.html">
       <span class="xh-num">4</span><strong>Evaluate</strong>
       <span>proper scores and significance tests</span></a>
   </section>

What it handles
---------------

.. grid:: 1 2 3 3
   :gutter: 4
   :class-container: xh-features

   .. grid-item::

      :octicon:`stack;1.6em;xh-icon`

      **Three kinds of input**

      Static, observed past and known-future inputs, weighed by
      variable selection at every step.

   .. grid-item::

      :octicon:`pulse;1.6em;xh-icon`

      **Level shift**

      Each window is standardised by its own history, so the
      forecast follows a level never seen in training.

   .. grid-item::

      :octicon:`filter;1.6em;xh-icon`

      **Missing inputs**

      Gaps go in as ``NaN``; masks and a learned decay handle them
      inside the network.

   .. grid-item::

      :octicon:`graph;1.6em;xh-icon`

      **Ordered quantiles**

      A monotone head: quantiles never cross, by construction.

   .. grid-item::

      :octicon:`shield-check;1.6em;xh-icon`

      **Calibration for any model**

      Split-conformal and adaptive conformal intervals that
      follow drifting errors.

   .. grid-item::

      :octicon:`git-compare;1.6em;xh-icon`

      **Fair comparisons**

      One metric and test suite for every model: CRPS, coverage,
      Diebold-Mariano, Friedman/Nemenyi.

Show me the code
----------------

.. tab-set::

   .. tab-item:: Model

      .. code-block:: python

         from xtft.nn.torch.models import XTFT

         model = XTFT(
             static_input_dim=4,
             dynamic_input_dim=5,       # past targets first
             future_input_dim=2,
             forecast_horizon=24,
             quantiles=[0.1, 0.5, 0.9],
             instance_norm=True,
         )
         q = model(x_static, x_dynamic, x_future)   # (batch, 24, 3)

   .. tab-item:: Train

      .. code-block:: python

         from xtft.nn.torch.trainer import TrainConfig, fit, predict

         history = fit(
             model, train_ds, val_ds, quantiles,
             TrainConfig(max_epochs=50, patience=5, device="auto"),
         )
         q_test = predict(model, test_ds)   # (N, H, O, K)

   .. tab-item:: Calibrate

      .. code-block:: python

         from xtft.evaluation import (
             AdaptiveConformalCalibrator,
             SimultaneousConformalBand,
         )

         aci = AdaptiveConformalCalibrator(quantiles).fit(y_val, q_val)
         q_cal = aci.transform(y_test, q_test, times)

         band = SimultaneousConformalBand(quantiles, coverage=0.9)
         lower, upper = band.fit(y_val, q_val).transform_online(
             y_test, q_test, times
         )

   .. tab-item:: Evaluate

      .. code-block:: python

         from xtft.evaluation import (
             diebold_mariano,
             evaluate_quantile_forecast,
         )

         evaluate_quantile_forecast(y_test, q_cal, quantiles)
         diebold_mariano(crps_model_a, crps_model_b, horizon=1)

From the gallery
----------------

.. raw:: html

   <div class="xh-gallery">
     <a href="auto_examples/plot_01_quickstart.html">
       <img src="_images/sphx_glr_plot_01_quickstart_thumb.png" alt=""/>
       <span>Forecast a panel of series</span></a>
     <a href="auto_examples/plot_02_missing_data.html">
       <img src="_images/sphx_glr_plot_02_missing_data_thumb.png" alt=""/>
       <span>Missing inputs and level shift</span></a>
     <a href="auto_examples/plot_03_calibration.html">
       <img src="_images/sphx_glr_plot_03_calibration_thumb.png" alt=""/>
       <span>Calibrate any quantile forecast</span></a>
     <a href="auto_examples/plot_04_compare_forecasters.html">
       <img src="_images/sphx_glr_plot_04_compare_forecasters_thumb.png" alt=""/>
       <span>Compare forecasters</span></a>
   </div>

Where to next
-------------

.. raw:: html

   <div class="xh-next">
     <a href="getting_started/first_forecast.html"><strong>First
       forecast</strong><span>From a data frame to calibrated
       quantiles in six steps.</span></a>
     <a href="user_guide/index.html"><strong>User guide</strong>
       <span>The model and its equations, training, missing data,
       calibration, evaluation.</span></a>
     <a href="api/index.html"><strong>API reference</strong>
       <span>Every public class and function, with examples.</span></a>
     <a href="user_guide/benchmarks.html"><strong>Benchmarks</strong>
       <span>Reproduce the comparison with other forecasters.</span></a>
   </div>

.. rst-class:: xh-footnote

A paper describing XTFT is in preparation; until it is published,
please cite the repository. xtft is released under the BSD
3-Clause licence. This documentation describes version |release|.

.. toctree::
   :hidden:

   getting_started/index
   user_guide/index
   auto_examples/index
   api/index
   development/index
   changelog
