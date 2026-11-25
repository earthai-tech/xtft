"""Compatibility shims for optional dependencies and library versions.

The Keras helpers re-exported here (model saving and loading,
prediction-output normalisation) need TensorFlow. They are imported
lazily on first access (PEP 562), so ``import xtft.compat`` and the
backend-neutral submodules, such as :mod:`xtft.compat.torch`, work
when TensorFlow is not installed.
"""

from __future__ import annotations

import importlib
from typing import Any

# Public name -> submodule that defines it.
_LAZY = {
    "load_inference_model": "keras",
    "load_model_from_tfv2": "keras",
    "save_manifest": "keras",
    "save_model": "keras",
    "normalize_predict_output": "keras_fit",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    """Import a Keras helper on first access."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    submodule = importlib.import_module(f".{module}", __name__)
    value = getattr(submodule, name)
    globals()[name] = value  # cache for later lookups
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])
