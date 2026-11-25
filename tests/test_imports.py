"""The PyTorch path must not import TensorFlow.

Regression test for a CI failure: ``xtft.compat`` imported its Keras
helpers eagerly, so the PyTorch trainer could not be imported where
TensorFlow is not installed. Each check runs in a fresh interpreter
so that modules imported by other tests do not hide the problem.
"""

import subprocess
import sys

import pytest

pytest.importorskip("torch")

MODULES = [
    "xtft.compat",
    "xtft.compat.torch",
    "xtft.nn.torch.trainer",
    "xtft.nn.torch.models",
    "xtft.evaluation",
    "xtft.data",
]


@pytest.mark.parametrize("module", MODULES)
def test_torch_path_does_not_import_tensorflow(module):
    code = (
        f"import sys, {module}; "
        "bad = sorted(m for m in ('tensorflow', 'keras') "
        "if m in sys.modules); "
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "", f"importing {module} imported {out}"


def test_compat_lazy_names_resolve():
    import xtft.compat as compat

    assert "normalize_predict_output" in dir(compat)
    with pytest.raises(AttributeError):
        _ = compat.not_a_name
