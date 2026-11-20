# XTFT rebuild plan

Status: 2026-11-20. Origin: TPAMI-2025-04-1034 rejected on
2026-07-30 (four reviewers). Old manuscript and reviews:
`E:\_REPOSITORIES\TPAMI`.

## Decisions taken

| Topic | Decision |
|---|---|
| Framework | PyTorch is the reference implementation and the one benchmarked; the Keras code from fusionlab-learn is kept under `xtft.nn.tf` and selected with `set_backend("tensorflow")` |
| Name | XTFT stays; "X" means *cross-scale*, not "Extreme" |
| Target venue | International Journal of Forecasting (first), Engineering Applications of AI (alternative); software paper (JMLR MLOSS or JOSS) for the library |
| Role of Nansha | Application case study (with Zhongshan), not the main evidence: 8 annual steps (2015-2022) cannot support a multiscale-temporal claim |

## Reviewer issues and how each is answered

| Issue | Answer |
|---|---|
| Bag of existing components; "Extreme" unjustified | One thesis (cross-scale fusion with calibrated quantiles); four principled changes below |
| One dataset, only TFT as baseline, no ablation | 8-10 public datasets, about 15 baselines, full ablations |
| q0.1-q0.9 called a 90% interval (it is 80%); coverage only | CRPS, pinball, Winkler/interval score, PICP and PINAW at the correct nominal level, reliability diagrams, MSIS |
| Possible random 70/30 leakage, no test set | Chronological train/val/test, rolling origin, spatial blocks for Nansha |
| Stride downsampling aliasing; stride choice | Learnable low-pass filter before decimation; strides from periodicity; LSTM vs TCN vs patch encoders |
| "Hierarchical" attention is a sum | Local (within scale) then global (across scale) attention |
| Fixed window W | Soft, differentiable learned window |
| Complexity formulas wrong | Corrected asymptotics plus measured params, FLOPs, time, memory |
| Interpretability not shown | Variable-selection weights, per-scale attribution, attention analysis, expert check on Nansha |
| Capacity confound | Parameter-matched TFT and XTFT |

## Method changes (PyTorch implementation)

1. Anti-aliased multi-resolution encoder.
2. Cross-scale hierarchical attention with a learned window.
3. Non-crossing quantile head plus conformalized quantile
   regression.
4. Measurable interpretability (per-variable, per-scale).

Memory attention and anomaly scoring remain library features, not
paper claims.

## Benchmarks (availability and licences to verify)

* Long-horizon: ETTh1/h2/m1/m2, Weather, Electricity, Traffic,
  Exchange, ILI.
* Covariate-rich and probabilistic: Electricity and Traffic in the
  TFT setting, EPF electricity prices, GEFCom2014, M5 or Favorita
  (Kaggle terms: author downloads).
* Application: Nansha and Zhongshan subsidence (processed with
  `D:\projects\geoprior-v3`; Stage-1 data in
  `E:\nature\nature\results\nansha_GeoPriorSubsNet_stage1`).

Baselines: naive and seasonal naive; ETS, ARIMA, Theta; LightGBM
and XGBoost on lags; DeepAR, N-BEATS, N-HiTS, TFT, Informer,
Autoformer, FEDformer, PatchTST, iTransformer, Crossformer, TimesNet,
DLinear, TiDE, TSMixer; optional zero-shot Chronos and TimesFM.

Protocol: five seeds, equal tuning budget, Diebold-Mariano tests,
Friedman/Nemenyi critical-difference diagrams. Heavy runs on the
HPC CPU nodes (env `sci` or a project env under
`/SCRATCH/kouakloMI/envs/xtft`), never on the login node.

## Work phases

1. Repository bootstrap and port from fusionlab-learn (done).
2. PyTorch TFT baseline, then PyTorch XTFT with the four changes;
   parity tests against the Keras version where designs coincide.
3. Probabilistic metrics module and data loaders with checksums.
4. Benchmark harness, baseline adapters, SLURM job templates.
5. Runs, ablations, statistics.
6. Nansha/Zhongshan case study.
7. Manuscript and figures (editable figure folders).
8. Public GitHub repository `earthai-tech/xtft`, documentation,
   PyPI release, software paper.
