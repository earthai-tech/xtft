Backends
========

One import serves both implementations: ``from xtft.nn import
XTFT`` returns the PyTorch class under the ``"torch"`` backend
(``xtft.nn.torch.models.XTFT``) and the Keras class under the
``"tensorflow"`` backend. The class is looked up when it is
imported, from the active backend:

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

The active backend is, in this order:

1. the one chosen with :func:`xtft.set_backend` (or temporarily
   with :func:`xtft.use_backend`);
2. the one named by the ``XTFT_BACKEND`` environment variable
   (``torch``, ``pytorch``, ``tensorflow``, ``tf`` or ``keras``);
3. otherwise the installed framework: PyTorch if it is installed,
   TensorFlow if only TensorFlow is.

So ``pip install "xtft[torch]"`` gives the PyTorch model and
``pip install "xtft[tf]"`` the Keras one, with no configuration.
With both installed, PyTorch is the default. Choose before
importing a model:

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
