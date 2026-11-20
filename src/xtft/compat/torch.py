"""PyTorch compatibility helpers.

Centralises version checks and device selection so that the PyTorch
implementation runs on recent releases and on the older wheels that
glibc-2.17 clusters can install (torch <= 2.5).
"""

from __future__ import annotations

import importlib.util

from packaging.version import Version

__all__ = [
    "HAS_TORCH",
    "TORCH_MIN_VERSION",
    "require_torch",
    "resolve_device",
    "torch_version",
]

TORCH_MIN_VERSION = Version("2.1")
HAS_TORCH = importlib.util.find_spec("torch") is not None


def torch_version() -> Version | None:
    """Return the installed PyTorch version.

    Returns
    -------
    packaging.version.Version or None
        The release version without local tags (``2.5.1+cpu``
        gives ``2.5.1``), or ``None`` if PyTorch is missing.
    """
    if not HAS_TORCH:
        return None
    import torch

    return Version(Version(torch.__version__).base_version)


def require_torch(min_version: Version = TORCH_MIN_VERSION):
    """Import PyTorch, checking its version.

    Parameters
    ----------
    min_version : packaging.version.Version, optional
        Oldest accepted release. Default is
        :data:`TORCH_MIN_VERSION`.

    Returns
    -------
    module
        The imported ``torch`` module.

    Raises
    ------
    ImportError
        If PyTorch is missing or older than `min_version`.
    """
    if not HAS_TORCH:
        raise ImportError(
            "The 'torch' backend needs PyTorch. Install it with "
            "`pip install xtft[torch]`."
        )
    import torch

    found = torch_version()
    if found < min_version:
        raise ImportError(
            f"xtft needs torch>={min_version}, found {found}."
        )
    return torch


def resolve_device(device: str | None = None):
    """Return a ``torch.device``, defaulting to CUDA when present.

    Parameters
    ----------
    device : str or None, optional
        Device string such as ``"cpu"``, ``"cuda"`` or
        ``"cuda:1"``. ``None`` or ``"auto"`` picks ``"cuda"`` if a
        GPU is visible and ``"cpu"`` otherwise.

    Returns
    -------
    torch.device
        The resolved device.

    Raises
    ------
    ImportError
        If PyTorch is missing.
    """
    torch = require_torch()
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)
