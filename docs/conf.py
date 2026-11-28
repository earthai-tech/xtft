"""Sphinx configuration of the xtft documentation.

Build locally with::

    pip install -e ".[torch,docs]"
    sphinx-build -b html -W docs docs/_build/html

Read the Docs builds the same sources from ``.readthedocs.yaml``.
The example gallery is executed during the build (a few minutes on
a CPU); set ``XTFT_DOCS_NO_EXAMPLES=1`` to skip running it.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Import the checkout, so that the pages describe the code they
# sit next to.
sys.path.insert(0, str(ROOT / "src"))

import xtft  # noqa: E402

# ------------------------------------------------------------ project
project = "xtft"
author = "Kouao Laurent Kouadio and contributors"
copyright = f"2025-{date.today().year}, earthai-tech"
release = xtft.__version__
version = ".".join(release.split(".")[:2])

# ------------------------------------------------------------ general
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
rst_epilog = f"""
.. |release| replace:: {release}
"""

# ------------------------------------------------------- autodoc, API
autosummary_generate = True
autodoc_typehints = "none"  # numpydoc sections give the types
autodoc_member_order = "bysource"
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}
# A build without network still succeeds, with fewer links.
intersphinx_timeout = 10

# ------------------------------------------------------------ gallery
sphinx_gallery_conf = {
    "examples_dirs": str(ROOT / "examples"),
    "gallery_dirs": "auto_examples",
    # Run every plot_*.py; the separator differs on Windows.
    "filename_pattern": re.escape(os.sep) + "plot_",
    "plot_gallery": (
        "False" if os.environ.get("XTFT_DOCS_NO_EXAMPLES") else "True"
    ),
    "download_all_examples": False,
    "remove_config_comments": True,
    # Files are numbered in reading order.
    "within_subsection_order": "FileNameSortKey",
    "matplotlib_animations": False,
    # Back-references feed the "Examples using ..." lists of the
    # API pages.
    "doc_module": ("xtft",),
    "backreferences_dir": "api/generated/backreferences",
    "show_memory": False,
}
# Matplotlib warns when an example runs without a display.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="FigureCanvasAgg is non-interactive",
)

# -------------------------------------------------------------- HTML
html_theme = "pydata_sphinx_theme"
html_title = f"xtft {release}"
html_logo = "_static/logo/xtft-symbol.svg"
html_favicon = "_static/logo/xtft-symbol.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "logo": {
        "text": "XTFT",
        "image_light": "_static/logo/xtft-symbol.svg",
        "image_dark": "_static/logo/xtft-symbol-dark.svg",
    },
    "github_url": "https://github.com/earthai-tech/xtft",
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "use_edit_page_button": True,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
html_context = {
    "github_user": "earthai-tech",
    "github_repo": "xtft",
    "github_version": "dev",
    "doc_path": "docs",
}
html_sidebars = {"index": []}
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
