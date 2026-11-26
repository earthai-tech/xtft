#!/bin/bash
# CAMELS-US flood forecasting study, one run per array task (line N
# of the task file). Daily data for 614 basins need more memory and
# time than the other benchmarks (TFT runs up to about a day).
#   python jobs/camels_tasks.py --out benchmarks/tasks_camels.txt
#   sbatch --array=1-$(wc -l < benchmarks/tasks_camels.txt)%12 \
#          jobs/camels_array.sbatch benchmarks/tasks_camels.txt
# %12 caps concurrent tasks (fair use).
#SBATCH --job-name=xtft-camels
#SBATCH --partition=B720-ibe-compute_28_64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=54G
#SBATCH --time=48:00:00
#SBATCH --output=logs/xtft-camels-%A_%a.out
#SBATCH --error=logs/xtft-camels-%A_%a.err

set -euo pipefail

TASKS="${1:?task file required}"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${TASKS}")
if [ -z "${LINE}" ]; then
    echo "no task on line ${SLURM_ARRAY_TASK_ID}"; exit 1
fi

# Site settings (conda, env, scratch): jobs/site.env, untracked.
source "${SLURM_SUBMIT_DIR}/jobs/site.env"
: "${XTFT_CONDA_SH:?set in jobs/site.env}"
: "${XTFT_ENV:?set in jobs/site.env}"
: "${XTFT_SCRATCH:?set in jobs/site.env}"
set +u
source "${XTFT_CONDA_SH}"
conda activate "${XTFT_ENV}"
set -u

# Use the source tree of this checkout, not an installed copy.
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export XTFT_DATA=${XTFT_SCRATCH}/data
OUT=${XTFT_SCRATCH}/results/${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUT}"

cd "${SLURM_SUBMIT_DIR}"
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID}"
echo "args: ${LINE}"
# Each line is split on whitespace (no shell quoting); see
# jobs/camels_tasks.py.
# shellcheck disable=SC2086
python benchmarks/run.py ${LINE} --threads "${SLURM_CPUS_PER_TASK}" \
    --out-dir "${OUT}"
