Backends
========

The models of :mod:`xtft.nn` are resolved for the active backend
when they are first accessed:

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Backend
     - Module
     - Notes
   * - ``"torch"`` (default)
     - :mod:`xtft.nn.torch`
     - Reference implementation, used in the benchmarks, with the
       trainer of :mod:`xtft.nn.torch.trainer`.
   * - ``"tensorflow"``
     - ``xtft.nn.tf``
     - Keras XTFT and TFT; needs the
       ``tf`` extra.

Choose the backend before importing a model, in code or with the
``XTFT_BACKEND`` environment variable:

.. code-block:: python

   import xtft

   xtft.set_backend("tensorflow")
   from xtft.nn import XTFT, TFT      # Keras classes

   with xtft.use_backend("torch"):    # temporarily
       from xtft.nn import XTFT       # PyTorch class

.. code-block:: bash

   XTFT_BACKEND=tensorflow python train.py

``TFT`` exists only for the TensorFlow backend. The PyTorch
benchmarks use the independent TFT of neuralforecast as the
baseline, so that the comparison does not rest on our own
re-implementation of the competitor.

Importing the PyTorch path never imports TensorFlow or Keras, and
:mod:`xtft.evaluation` and :mod:`xtft.data` import neither
framework.
