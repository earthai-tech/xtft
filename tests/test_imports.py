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


# Two-level public API: ``from xtft.nn import XTFT, fit`` and the
# like, resolved lazily without circular imports.
SHORT_NAMES = {
    "XTFT": "xtft.nn.torch.models.xtft",
    "TrainConfig": "xtft.nn.torch.trainer",
    "fit": "xtft.nn.torch.trainer",
    "predict": "xtft.nn.torch.trainer",
    "QuantileLoss": "xtft.nn.torch.probabilistic.losses",
    "pinball_loss": "xtft.nn.torch.probabilistic.losses",
    "NonCrossingQuantileHead": (
        "xtft.nn.torch.probabilistic.quantile_head"
    ),
}


@pytest.mark.parametrize("name", sorted(SHORT_NAMES))
def test_short_import_in_fresh_interpreter(name):
    # A fresh interpreter: a circular import would fail here even if
    # other tests had already imported the modules involved.
    code = f"from xtft.nn import {name} as obj; print(obj.__module__)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == SHORT_NAMES[name]


def test_short_names_are_the_canonical_objects():
    from xtft.nn import XTFT, TrainConfig, fit, predict
    from xtft.nn.torch.models import XTFT as XTFT_long
    from xtft.nn.torch.trainer import TrainConfig as Cfg_long
    from xtft.nn.torch.trainer import fit as fit_long
    from xtft.nn.torch.trainer import predict as predict_long

    assert XTFT is XTFT_long
    assert TrainConfig is Cfg_long
    assert fit is fit_long
    assert predict is predict_long


def test_package_imports_load_no_framework():
    code = (
        "import sys, xtft, xtft.nn, xtft.data, xtft.evaluation; "
        "print(','.join(sorted(m for m in "
        "('torch', 'tensorflow', 'keras') if m in sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "", f"plain imports loaded {out}"


def test_top_level_lazy_subpackages():
    import xtft

    assert callable(xtft.evaluation.coverage)
    assert xtft.data.PanelSpec().target == ("y",)
    assert "evaluation" in dir(xtft)
