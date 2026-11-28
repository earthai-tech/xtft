"""xtft: covariate-aware probabilistic forecasting.

xtft provides the XTFT model family, a temporal fusion network
robust to level shift and missing inputs, with interchangeable
PyTorch (default) and TensorFlow/Keras implementations, together
with probabilistic metrics, conformal calibration (per-step and
whole-path), data utilities and benchmark tooling.

The public API is reachable in two levels:

* :mod:`xtft.nn`: the model (``XTFT``) and its training loop
  (``TrainConfig``, ``fit``, ``predict``);
* :mod:`xtft.data`: panels, splits, scaling and windows;
* :mod:`xtft.evaluation`: metrics, conformal calibration and
  statistical tests.

Choose the framework with :func:`xtft.set_backend` or the
``XTFT_BACKEND`` environment variable before importing models from
:mod:`xtft.nn`.

Examples
--------
>>> import xtft
>>> xtft.get_backend() in xtft.backends.SUPPORTED_BACKENDS
True
"""

from __future__ import annotations

import importlib
import logging

from . import backends
from .backends import (
    available_backends,
    get_backend,
    set_backend,
    use_backend,
)

try:
    from ._version import version as __version__
except ImportError:  # editable install without a build step
    __version__ = "0.1.0.dev0"

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Subpackages imported on first attribute access, so that
# ``import xtft`` stays light and loads no deep-learning framework.
_LAZY_SUBMODULES = (
    "api",
    "compat",
    "core",
    "data",
    "evaluation",
    "metrics",
    "nn",
    "utils",
)

__all__ = [
    "__version__",
    "available_backends",
    "backends",
    "get_backend",
    "set_backend",
    "use_backend",
    *_LAZY_SUBMODULES,
]


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
