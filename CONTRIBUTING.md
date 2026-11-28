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
| `docs.yml` | Sphinx build with warnings as errors (gallery executed on `master` and tags); after a push, asks Read the Docs to rebuild when `READTHEDOCS_TOKEN` is set |
| `release.yml` | on a `v*` tag: build, check the tag against `pyproject.toml`, run the tests against the installed wheel, publish (final versions to PyPI, pre-releases to TestPyPI) and create the GitHub release; a manual run publishes the current commit to TestPyPI or PyPI |

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
2. Try the package on TestPyPI first: run the *Release* workflow
   by hand with target `testpypi`, or push a pre-release tag such
   as `v0.1.0rc1` (with the same version in `pyproject.toml`).
3. Tag the final version: `git tag vX.Y.Z && git push origin vX.Y.Z`.
