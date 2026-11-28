Contributing
============

Bug reports, questions and pull requests are welcome on
`GitHub <https://github.com/earthai-tech/xtft>`_. Work on the
``dev`` branch; ``master`` receives tested releases.

Set up
------

.. code-block:: bash

   git clone https://github.com/earthai-tech/xtft.git
   cd xtft
   git switch dev
   pip install -e ".[torch,dev,test,docs]"

Checks
------

Run them before opening a pull request; continuous integration
runs the same commands.

.. code-block:: bash

   ruff check .
   ruff format --check src tests tools benchmarks examples
   pytest --cov=xtft
   sphinx-build -b html -W --keep-going docs docs/_build/html

Code follows the numpydoc convention, with lines of at most 71
characters. Public functions and classes have a docstring with
parameters, returns, raised exceptions, the equations they
implement (in ``Notes``), references and runnable examples. New
behaviour comes with tests, including regression tests for fixed
bugs.

Documentation
-------------

The pages are reStructuredText in ``docs/``; the API pages are
generated from the docstrings, and the gallery from the scripts in
``examples/``, which run during the build. A new example is a file
``examples/plot_<name>.py`` with a docstring title, cells separated
by ``# %%`` comments, a fixed seed, and a run time of at most two
minutes on a CPU. To build without running the examples:

.. code-block:: bash

   XTFT_DOCS_NO_EXAMPLES=1 sphinx-build -b html docs docs/_build/html

Read the Docs builds the site from ``.readthedocs.yaml``; the
``docs.yml`` workflow checks it on every push to ``master``.

Continuous integration
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Workflow
     - What it does
   * - ``tests.yml``
     - pytest with coverage on Python 3.10-3.12 (Linux) and 3.12
       (Windows, macOS), CPU PyTorch; Keras backend tests on
       ``master``, weekly and on demand; coverage to Codecov; every
       gallery example run as a script
   * - ``lint.yml``
     - ``ruff check`` and ``ruff format --check``
   * - ``docs.yml``
     - on every push to ``master``: Sphinx build with warnings as
       errors, gallery executed; optionally triggers Read the Docs
   * - ``release.yml``
     - only when a ``v*`` tag is pushed: build, check the tag
       against ``pyproject.toml``, run the tests against the wheel,
       upload to TestPyPI (``TEST_PYPI_API_TOKEN``), then to PyPI
       for a final version (``PYPI_API_TOKEN``), and create the
       GitHub release

Releasing
---------

1. Set ``version`` in ``pyproject.toml``, update
   :doc:`/changelog`, and merge ``dev`` into ``master``.
2. Tag and push: ``git tag vX.Y.Z && git push origin vX.Y.Z``.
