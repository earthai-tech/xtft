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
ruff format --check src tests tools benchmarks
pytest --cov=xtft
sphinx-build -b html -W docs docs/_build/html
```

Code follows the numpydoc convention, with lines of at most 71
characters. New behaviour comes with tests.

## Continuous integration

| Workflow | What it does |
|---|---|
| `tests.yml` | pytest with coverage on Python 3.10-3.12 (Linux) and 3.12 (Windows, macOS), CPU PyTorch; Keras backend tests on `master`, weekly and on demand; coverage to Codecov |
| `lint.yml` | `ruff check` and `ruff format --check` |
| `docs.yml` | Sphinx build with warnings as errors; deployed to GitHub Pages from `master` and version tags |
| `release.yml` | on a `vX.Y.Z` tag: build, check the tag against `pyproject.toml`, test the wheel, publish (final versions to PyPI, pre-releases to TestPyPI), create the GitHub release |

## Releasing

1. Set `version` in `pyproject.toml` and merge `dev` into `master`.
2. `git tag vX.Y.Z && git push origin vX.Y.Z`.
