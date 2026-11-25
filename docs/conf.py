"""Sphinx configuration for the xtft documentation."""

from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../src"))

import xtft  # noqa: E402

project = "xtft"
author = "earthai-tech"
copyright = f"2025-{date.today().year}, {author}"
release = xtft.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
]

autosummary_generate = True
autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

templates_path = []
exclude_patterns = ["_build"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "xtft"
html_logo = "_static/logo/xtft-symbol.svg"
html_theme_options = {
    "logo": {
        "image_light": "_static/logo/xtft-logo.svg",
        "image_dark": "_static/logo/xtft-logo-dark.svg",
    },
    "github_url": "https://github.com/earthai-tech/xtft",
    "navigation_with_keys": False,
}
