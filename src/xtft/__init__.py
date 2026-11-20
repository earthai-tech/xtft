"""xtft: cross-scale Temporal Fusion Transformer for forecasting.

xtft provides the XTFT model family with interchangeable PyTorch
(default) and TensorFlow/Keras implementations, together with
probabilistic metrics, data utilities and benchmark tooling.

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

_LAZY_SUBMODULES = (
    "api",
    "compat",
    "core",
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
