# Contributing

## Development setup

```bash
git clone https://github.com/earthai-tech/xtft.git
cd xtft
pip install -e ".[torch,dev,test,docs]"
```

Work on the `dev` branch; `master` receives tested releases.

## Checks

```bash
ruff check .
ruff format --check src tests tools benchmarks examples
pytest --cov=xtft
sphinx-build -b html -W docs docs/_build/html   # runs the examples
XTFT_DOCS_NO_EXAMPLES=1 sphinx-build -b html -W docs docs/_build/html
```

Code follows the numpydoc convention, with lines of at most 71
characters. New behaviour comes with tests.

## Continuous integration

| Workflow | What it does |
|---|---|
| `tests.yml` | pytest with coverage on Python 3.10-3.12 (Linux) and 3.12 (Windows, macOS), CPU PyTorch, coverage to Codecov; Keras backend tests on `master`, weekly and on demand; every gallery example run as a script |
| `lint.yml` | `ruff check` and `ruff format --check` |
| `docs.yml` | on every push to `master`: Sphinx build with warnings as errors and the gallery executed; then asks Read the Docs to rebuild when `READTHEDOCS_TOKEN` is set |
| `release.yml` | only when a `v*` tag is pushed: build, check the tag against `pyproject.toml`, run the tests against the installed wheel, upload to TestPyPI, then to PyPI for a final version (pre-releases stay on TestPyPI), and create the GitHub release |

Repository secrets (Settings > Secrets and variables > Actions):

| Secret | Used by | Purpose |
|---|---|---|
| `PYPI_API_TOKEN` | `release.yml` | upload to pypi.org |
| `TEST_PYPI_API_TOKEN` | `release.yml` | upload to test.pypi.org |
| `CODECOV_TOKEN` | `tests.yml` | coverage upload to Codecov |
| `READTHEDOCS_TOKEN` (optional) | `docs.yml` | trigger Read the Docs builds when its webhook is not set up |

## Releasing

1. Set `version` in `pyproject.toml`, update `docs/changelog.rst`
   and merge `dev` into `master`.
2. Optionally rehearse with a pre-release: set the version to, for
   example, `0.1.0rc1` and push the tag `v0.1.0rc1`; it goes to
   TestPyPI only.
3. Tag the final version: `git tag vX.Y.Z && git push origin vX.Y.Z`.
   It goes to TestPyPI, then PyPI.
4. Documentation versions: Read the Docs builds `latest` (master),
   `dev` and every tag. For the first release, activate the `stable`
   version on Read the Docs and add it to
   `docs/_static/switcher.json` as the preferred entry, e.g.
   `{"name": "0.1.0 (stable)", "version": "stable",
   "url": "https://xtft.readthedocs.io/en/stable/", "preferred": true}`
   (and remove `"preferred"` from `latest`). The switcher of every
   published version reads this list from the `latest` build.
