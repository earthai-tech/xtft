# Benchmarks

Scripts that compare XTFT with other forecasters under one protocol:
the same input length, horizon, rolling origins and optimisation
budget (steps and early-stopping patience) for every learned model,
chronological train/validation/test periods, several seeds, and
metrics and conformal calibration from `xtft.evaluation`.
Install the baselines with `pip install -e ".[torch,bench]"`.

```bash
python benchmarks/run.py --dataset ETTh1 --model xtft --horizon 96
python benchmarks/run.py --dataset ETTh1 --model nf:PatchTST --horizon 96
python benchmarks/aggregate.py results --metric crps_q --reference xtft
```

| File | Role |
|---|---|
| `datasets.py` | dataset registry; data are downloaded on first use into `data/raw` |
| `models.py` | adapters with one interface: `xtft`, `nf:<Model>` (neuralforecast), `sf:<Model>` (statsforecast), `lgb`, `lear` |
| `baselines_ml.py` | LightGBM global quantile model and LEAR (LASSO-estimated autoregressive model for day-ahead prices) |
| `run.py` | one (dataset, model, horizon, seed) cell: fit, forecast at shared rolling origins, calibrate (split-conformal and adaptive), score; writes a JSON file, per-window losses and forecasts |
| `aggregate.py` | tables over runs, average ranks, Friedman/Nemenyi, Diebold-Mariano tests against a reference model |
| `recalibrate.py` | re-scores saved forecasts with other calibration settings or whole-path bands, without refitting |
| `make_tasks.py` | task list for a SLURM array (`jobs/bench_array.sbatch`, see `jobs/README.md`) |
| `speed/` | training and inference speed, PyTorch vs Keras XTFT |

## Data sets

* Long horizon: ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic,
  Exchange, ILI, Weather (via datasetsforecast).
* Day-ahead electricity prices: `EPF-NP`, `EPF-PJM`, `EPF-BE`,
  `EPF-FR`, `EPF-DE` (open benchmark of Lago et al., 2021), with two
  day-ahead load or generation forecasts as known covariates.
* Retail: `M5-StoreDept`, M5 sales aggregated to 70 store x
  department series, with price index, SNAP days, events and static
  store, department and state attributes.
* `Synthetic`: small series for smoke tests.

## Options

* Missing inputs: `--missing-rate 0.2 --missing-mode random|blocks`
  removes target values; models train on the gapped series and are
  scored on the clean one.
* Updating during the test period:
  `--model-kw '{"refit_every": 672, "refit_mode": "finetune_es"}'`
  updates XTFT every 672 steps (`finetune`, `finetune_es` or
  `retrain`); LEAR takes `recalibrate_every` in days (0: once).
* Model settings: `--model-kw` passes a JSON dict to the model
  (e.g. `'{"hidden_units": 128}'`); `--tag` labels the variant.
