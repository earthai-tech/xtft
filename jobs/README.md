# HPC jobs

SLURM batch scripts used for the tests, the speed comparison and the
benchmark arrays. Submit them from the repository root; logs go to
`logs/`.

1. Copy `site.env.example` to `site.env` (untracked) and set the
   conda hook, the environment and a scratch folder.
2. Pick the partition with `sbatch -p <partition>` if the default in
   the script header does not exist on your cluster.

| Script | Purpose |
|---|---|
| `tests_cpu.sbatch` | test suite on one CPU node |
| `speed_cpu.sbatch` | PyTorch vs Keras XTFT speed (`benchmarks/speed`) |
| `bench_array.sbatch` | one benchmark cell per array task (`benchmarks/make_tasks.py`) |

The scripts avoid `%x` in `--output` (unsupported by old SLURM
releases, e.g. 15.08), so job names are written literally.
