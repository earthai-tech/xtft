"""Backend selection for xtft.

Two kinds of backend live here:

* the deep-learning framework (PyTorch or TensorFlow) that
  :mod:`xtft.nn` resolves models from, see :func:`set_backend`;
* array-computation backends (NumPy, SciPy, CuPy, Dask) inherited
  from fusionlab-learn, in :mod:`xtft.backends.array`.
"""

from ._framework import (
    DEFAULT_BACKEND,
    ENV_VAR,
    SUPPORTED_BACKENDS,
    available_backends,
    get_backend,
    is_available,
    set_backend,
    use_backend,
)

__all__ = [
    "DEFAULT_BACKEND",
    "ENV_VAR",
    "SUPPORTED_BACKENDS",
    "available_backends",
    "get_backend",
    "is_available",
    "set_backend",
    "use_backend",
]
