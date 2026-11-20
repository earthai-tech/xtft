"""TensorFlow/Keras implementation of XTFT, ported from fusionlab.

This subpackage keeps the Keras code of fusionlab-learn (XTFT,
SuperXTFT, TFT and their layers) with imports rewritten to
:mod:`xtft`. It exposes ``KERAS_DEPS``, a lazy loader of Keras and
TensorFlow symbols, and ``KERAS_BACKEND``, which is falsy when
TensorFlow is missing so that the modules still import.

Select it with ``xtft.backends.set_backend("tensorflow")`` and
import models from :mod:`xtft.nn`.
"""

import os

# Silence TensorFlow C++ INFO/WARNING logs and oneDNN notices.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from ...compat._config import (  # noqa: E402
    Config as config,
)
from ...compat._config import (  # noqa: E402
    check_keras_backend,
    import_keras_dependencies,
)

config.INSTALL_DEPS = False
config.WARN_STATUS = "warn"

_EXTRA_MSG = (
    "The 'tensorflow' backend of xtft needs the `tensorflow` "
    "package."
)

KERAS_DEPS = import_keras_dependencies(
    extra_msg=_EXTRA_MSG, error="ignore"
)
KERAS_BACKEND = check_keras_backend(error="ignore")


def dependency_message(module_name: str) -> str:
    """Build the error text shown when TensorFlow is missing.

    Parameters
    ----------
    module_name : str
        Name of the object that needs TensorFlow.

    Returns
    -------
    str
        Human-readable installation hint.
    """
    return (
        f"`{module_name}` needs the `tensorflow` package. Install "
        "it with `pip install xtft[tf]`."
    )


__all__ = ["KERAS_BACKEND", "KERAS_DEPS", "dependency_message"]
