#!/bin/bash
# One benchmark cell per array task; line N of the task file.
#   python benchmarks/make_tasks.py ... --out benchmarks/tasks.txt
#   sbatch --array=1-$(wc -l < benchmarks/tasks.txt)%20 \
#          jobs/bench_array.sbatch benchmarks/tasks.txt
# %20 caps concurrent tasks (fair use). Each task is small and
# latency-bound, so 8 cores per task beat whole nodes.
#SBATCH --job-name=xtft-bench
#SBATCH --partition=B720-ibe-compute_28_64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/xtft-bench-%A_%a.out
#SBATCH --error=logs/xtft-bench-%A_%a.err

set -euo pipefail

TASKS="${1:?task file required}"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${TASKS}")
if [ -z "${LINE}" ]; then
    echo "no task on line ${SLURM_ARRAY_TASK_ID}"; exit 1
fi

# Site settings (conda, env, scratch): jobs/site.env, untracked;
# copy jobs/site.env.example and edit it.
source "${SLURM_SUBMIT_DIR}/jobs/site.env"
: "${XTFT_CONDA_SH:?set in jobs/site.env}"
: "${XTFT_ENV:?set in jobs/site.env}"
: "${XTFT_SCRATCH:?set in jobs/site.env}"
set +u
source "${XTFT_CONDA_SH}"
conda activate "${XTFT_ENV}"
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export XTFT_DATA=${XTFT_SCRATCH}/data
OUT=${XTFT_SCRATCH}/results/${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUT}" "${XTFT_DATA}"

cd "${SLURM_SUBMIT_DIR}"
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID} commit=$(git rev-parse --short HEAD 2>/dev/null || echo none)"
echo "args: ${LINE}"
# shellcheck disable=SC2086
python benchmarks/run.py ${LINE} --threads "${SLURM_CPUS_PER_TASK}" \
    --out-dir "${OUT}"
