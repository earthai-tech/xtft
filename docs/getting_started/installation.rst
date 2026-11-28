Installation
============

xtft needs Python 3.10 or later. Install it with the PyTorch
backend, which is the reference implementation:

.. code-block:: bash

   pip install "xtft[torch]"

.. note::

   Until the first release is on PyPI, install from the
   repository (see :ref:`install-source`).

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Extra
     - Installs
     - For
   * - ``torch``
     - PyTorch 2.1 or later
     - the XTFT model, training and inference
   * - ``tf``
     - TensorFlow 2.20 and Keras 3
     - the Keras XTFT and TFT ported from fusionlab-learn
   * - ``bench``
     - neuralforecast, statsforecast, datasetsforecast, LightGBM,
       xgboost
     - the benchmark scripts and baselines
   * - ``test``
     - pytest, pytest-cov
     - the test suite
   * - ``docs``
     - Sphinx, the pydata theme, sphinx-gallery and extensions
     - this documentation

The calibration and evaluation modules (:mod:`xtft.evaluation`)
and the data utilities (:mod:`xtft.data`) need only NumPy, pandas
and SciPy, so they can score the forecasts of any model without a
deep-learning framework.

GPUs and mixed precision
------------------------

Training runs on the CPU, a CUDA GPU or Apple ``mps``;
``TrainConfig(device="auto")`` picks CUDA when it is available.
Install the PyTorch build that matches your CUDA driver from
`pytorch.org <https://pytorch.org/get-started/locally/>`_ before
installing xtft, for example:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install "xtft[torch]"

Mixed precision is chosen with ``TrainConfig(amp="bf16")`` (CPU
and recent GPUs) or ``amp="fp16"`` (CUDA, with gradient scaling).

.. _install-source:

From source
-----------

.. code-block:: bash

   git clone https://github.com/earthai-tech/xtft.git
   cd xtft
   pip install -e ".[torch,dev,test]"
   pytest

Check the installation:

.. code-block:: python

   >>> import xtft
   >>> xtft.get_backend()
   'torch'
