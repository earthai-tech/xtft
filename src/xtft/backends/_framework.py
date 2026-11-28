"""Deep-learning framework selection for xtft.

xtft ships one model family with two implementations: a PyTorch
implementation (the default and the one used for benchmarking) and
a TensorFlow/Keras implementation.
This module decides which implementation :mod:`xtft.nn` resolves.

The framework is chosen, in decreasing priority, by

1. an explicit call to :func:`set_backend`,
2. the ``XTFT_BACKEND`` environment variable,
3. the installed frameworks: ``"torch"`` if PyTorch is installed,
   otherwise ``"tensorflow"`` if TensorFlow is, otherwise
   ``"torch"`` (the default; importing a model then reports the
   missing package).

The frameworks are located with :func:`importlib.util.find_spec`,
without being imported.

The choice should be made before the first model is imported from
:mod:`xtft.nn`; classes already imported keep the implementation
they were resolved with.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

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

logger = logging.getLogger(__name__)

ENV_VAR = "XTFT_BACKEND"
DEFAULT_BACKEND = "torch"
SUPPORTED_BACKENDS = ("torch", "tensorflow")

# Distribution module that must be importable for each backend.
_REQUIRED_MODULE = {"torch": "torch", "tensorflow": "tensorflow"}
_ALIASES = {
    "torch": "torch",
    "pytorch": "torch",
    "pt": "torch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "keras": "tensorflow",
}

_state: dict[str, str | None] = {"backend": None}


def _normalize(name: str) -> str:
    """Map a user-facing backend name to its canonical name.

    Parameters
    ----------
    name : str
        Backend name or alias, case-insensitive. Accepted values are
        ``"torch"``, ``"pytorch"``, ``"pt"``, ``"tensorflow"``,
        ``"tf"`` and ``"keras"``.

    Returns
    -------
    str
        ``"torch"`` or ``"tensorflow"``.

    Raises
    ------
    TypeError
        If `name` is not a string.
    ValueError
        If `name` is not a known backend or alias.
    """
    if not isinstance(name, str):
        raise TypeError(
            f"Backend name must be a str, got {type(name).__name__}."
        )
    key = name.strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown backend {name!r}. Expected one of "
            f"{sorted(_ALIASES)}."
        )
    return _ALIASES[key]


def is_available(name: str) -> bool:
    """Tell whether a backend's framework is installed.

    The check uses :func:`importlib.util.find_spec`, so the framework
    is located without being imported.

    Parameters
    ----------
    name : str
        Backend name or alias (see :func:`set_backend`).

    Returns
    -------
    bool
        ``True`` if the framework package can be imported.

    Examples
    --------
    >>> from xtft.backends import is_available
    >>> isinstance(is_available("torch"), bool)
    True
    """
    module = _REQUIRED_MODULE[_normalize(name)]
    return importlib.util.find_spec(module) is not None


def available_backends() -> tuple[str, ...]:
    """Return the supported backends whose framework is installed.

    Returns
    -------
    tuple of str
        Canonical names, in the order of
        :data:`SUPPORTED_BACKENDS`.

    See Also
    --------
    is_available : Check a single backend.
    """
    return tuple(b for b in SUPPORTED_BACKENDS if is_available(b))


def get_backend() -> str:
    """Return the active deep-learning backend.

    The explicit choice of :func:`set_backend` comes first, then
    the ``XTFT_BACKEND`` environment variable. Without either, the
    backend follows the installed frameworks: PyTorch when it is
    installed, else TensorFlow when it is, else ``"torch"``.

    Returns
    -------
    str
        ``"torch"`` or ``"tensorflow"``.

    Raises
    ------
    ValueError
        If ``XTFT_BACKEND`` holds an unknown name.

    See Also
    --------
    set_backend : Choose the backend explicitly.
    available_backends : The backends whose framework is installed.

    Examples
    --------
    >>> from xtft.backends import get_backend, use_backend
    >>> with use_backend("torch"):
    ...     get_backend()
    'torch'
    """
    if _state["backend"] is not None:
        return _state["backend"]
    env = os.environ.get(ENV_VAR)
    if env:
        return _normalize(env)
    if not is_available(DEFAULT_BACKEND):
        # A TensorFlow-only installation works out of the box.
        for name in SUPPORTED_BACKENDS:
            if is_available(name):
                return name
    return DEFAULT_BACKEND


def set_backend(name: str, *, strict: bool = True) -> str:
    """Select the deep-learning backend used by :mod:`xtft.nn`.

    Parameters
    ----------
    name : str
        Backend name or alias: ``"torch"`` (aliases ``"pytorch"``,
        ``"pt"``) or ``"tensorflow"`` (aliases ``"tf"``,
        ``"keras"``).
    strict : bool, default=True
        If ``True``, refuse a backend whose framework is not
        installed. If ``False``, accept it and log a warning; the
        import error is then raised when a model is resolved.

    Returns
    -------
    str
        The canonical name of the selected backend.

    Raises
    ------
    ValueError
        If `name` is unknown.
    ImportError
        If `strict` is ``True`` and the framework is missing.

    Notes
    -----
    The selection is process-wide and is not thread-safe. Classes
    already imported from :mod:`xtft.nn` are not re-resolved.

    Examples
    --------
    >>> from xtft.backends import get_backend, set_backend
    >>> previous = get_backend()
    >>> set_backend("pytorch", strict=False)
    'torch'
    >>> _ = set_backend(previous, strict=False)
    """
    canonical = _normalize(name)
    if not is_available(canonical):
        msg = (
            f"Backend {canonical!r} requires the "
            f"{_REQUIRED_MODULE[canonical]!r} package, which is not "
            "installed."
        )
        if strict:
            raise ImportError(msg)
        logger.warning(msg)
    _state["backend"] = canonical
    return canonical


@contextmanager
def use_backend(name: str) -> Iterator[str]:
    """Temporarily select a backend inside a ``with`` block.

    Parameters
    ----------
    name : str
        Backend name or alias (see :func:`set_backend`). The
        framework does not need to be installed.

    Yields
    ------
    str
        The canonical name of the temporary backend.

    Examples
    --------
    >>> from xtft.backends import get_backend, use_backend
    >>> with use_backend("tf"):
    ...     get_backend()
    'tensorflow'
    """
    previous = _state["backend"]
    try:
        yield set_backend(name, strict=False)
    finally:
        _state["backend"] = previous
