#!/bin/bash
# Cost scaling of XTFT with input length, horizon and width
# (tools/cost_scaling.py) on one exclusive CPU node, 8 threads as in
# the benchmark runs. Submit from the repository root:
#   sbatch jobs/cost_scaling.sbatch
#SBATCH --job-name=xtft-cost
#SBATCH --partition=B720-ibe-compute_28_64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/xtft-cost-%j.out
#SBATCH --error=logs/xtft-cost-%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/jobs/site.env"
: "${XTFT_CONDA_SH:?set in jobs/site.env}"
: "${XTFT_ENV:?set in jobs/site.env}"
: "${XTFT_SCRATCH:?set in jobs/site.env}"
set +u
source "${XTFT_CONDA_SH}"
conda activate "${XTFT_ENV}"
set -u

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
OUT=${XTFT_SCRATCH}/cost
mkdir -p "${OUT}"
cd "${SLURM_SUBMIT_DIR}"
echo "node=$(hostname) job=${SLURM_JOB_ID}"
lscpu | grep -E "Model name|Socket|Core|Thread"

python tools/cost_scaling.py --threads 8 --batch 128 --steps 10 \
    --out "${OUT}/cost_scaling_${SLURM_JOB_ID}.csv"
