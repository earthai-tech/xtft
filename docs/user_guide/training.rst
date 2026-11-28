Training
========

.. currentmodule:: xtft.nn.torch.trainer

:func:`fit` is a deliberately small training loop: Adam, gradient
clipping, early stopping on the validation loss with the best
weights restored, and fixed seeds. It trains any module that takes
``(x_static, x_dynamic, x_future)`` and returns quantiles, so XTFT
and its ablations share the same loop.

Objective
---------

The loss is the mean pinball loss over the observed targets,

.. math::

   \mathcal{L} = \frac{1}{K |\Omega|}
   \sum_{(i, t_0, h) \in \Omega} \sum_{k=1}^{K}
   \rho_{\tau_k}\bigl(y_{i, t_0 + h} - \hat q_{i, t_0, h, k}\bigr),
   \qquad \rho_\tau(u) = \max\{\tau u, (\tau - 1) u\},

where :math:`\Omega` holds the windows and steps of the training
set whose target is finite and :math:`\tau_1 < \dots < \tau_K`
are the quantile levels. Missing targets therefore need no
special treatment.

Settings
--------

.. code-block:: python

   from xtft.nn import TrainConfig, fit, predict

   cfg = TrainConfig(
       max_epochs=100,
       batch_size=64,
       lr=1e-3,
       patience=10,          # epochs without improvement
       steps_per_epoch=None, # or a number of random batches
       grad_clip=1.0,
       device="auto",        # CUDA if available, else CPU
       amp=None,             # "bf16" or "fp16"
       seed=0,
   )
   history = fit(model, train_ds, val_ds, quantiles, cfg)
   q = predict(model, test_ds, batch_size=256)

``history`` holds the training and validation loss of every
epoch, the best epoch and its loss, the number of epochs, the
wall time and the configuration.

.. dropdown:: Long panels: epochs of random batches

   A panel of hundreds of series with years of daily data has
   millions of windows. ``steps_per_epoch`` turns an epoch into
   that many batches drawn at random with replacement, so that
   validation and early stopping happen at a useful rhythm. The
   budget is then counted in optimiser steps:
   ``max_epochs * steps_per_epoch``.

.. dropdown:: Devices and mixed precision

   ``device`` accepts ``"auto"``, ``"cpu"``, ``"cuda"``,
   ``"cuda:1"`` or ``"mps"``. With ``amp="bf16"`` the forward pass
   runs in bfloat16 under :func:`torch.autocast` while the
   parameters, optimiser state and loss stay in float32; it also
   runs on recent CPUs. ``amp="fp16"`` needs CUDA and uses gradient
   scaling. :func:`predict` takes the same ``amp`` option and
   returns float32.

.. dropdown:: Reproducibility

   ``seed`` fixes the batch order and dropout. The initial
   parameters depend on the global PyTorch generator, so call
   :func:`torch.manual_seed` before building the model. GPU
   kernels may still introduce small nondeterministic
   differences.

Updating after deployment
-------------------------

A model trained once can lose accuracy when the series change.
The benchmarks compare three ways of updating it during the test
period every ``refit_every`` steps, each on a rolling window of
recent data: training a new model (``retrain``), adding a short
fine-tuning run to the current weights (``finetune``), and an
anchored update that starts again from the originally trained
weights, stops early on the latest observations and keeps the
update only if it beats the original model there
(``finetune_es``). These schemes live in the benchmark adapter;
see :doc:`benchmarks`.

See also
--------

* :class:`TrainConfig`, :func:`fit`, :func:`predict` in the
  :doc:`API reference </api/index>`.
