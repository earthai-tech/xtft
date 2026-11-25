Installation
============

From PyPI, with the PyTorch backend:

.. code-block:: bash

   pip install "xtft[torch]"

From source, for development:

.. code-block:: bash

   git clone https://github.com/earthai-tech/xtft.git
   cd xtft
   pip install -e ".[torch,dev]"

Optional extras: ``tf`` (Keras backend), ``bench`` (benchmark
baselines), ``test`` (coverage) and ``docs`` (this documentation).
Python 3.10 or later is required.
