# Benchmarks

| File | Role |
|---|---|
| `datasets.py` | dataset registry (long-horizon sets from datasetsforecast, `Synthetic` for smoke tests); conventional validation/test lengths |
| `models.py` | adapters with one interface: `xtft`, `nf:<Model>` (neuralforecast), `sf:<Model>` (statsforecast) |
| `run.py` | one (dataset, model, horizon, seed) cell: fit, forecast at shared rolling origins, conformal calibration, metrics; writes a JSON file and per-window losses |
| `make_tasks.py` | task list for a SLURM array (`jobs/bench_array.sbatch`) |
| `aggregate.py` | tables over runs, average ranks, Friedman/Nemenyi, Diebold-Mariano tests against a reference model |
| `speed/` | training and inference speed, PyTorch vs Keras XTFT |

Every model gets the same input length, horizon, rolling origins and
optimisation budget (steps and early-stopping patience); all metrics
and calibration come from `xtft.evaluation`.
