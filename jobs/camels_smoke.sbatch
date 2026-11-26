#!/bin/bash
# Smoke test of the CAMELS-US study: builds the data cache, prints
# the dataset summary and runs every model briefly on 20 basins.
# Submit from the repository root: sbatch jobs/camels_smoke.sbatch
#SBATCH --job-name=xtft-camels-smoke
#SBATCH --partition=B720-ibe-compute_28_64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/xtft-camels-smoke-%j.out
#SBATCH --error=logs/xtft-camels-smoke-%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/jobs/site.env"
set +u
source "${XTFT_CONDA_SH}"
conda activate "${XTFT_ENV}"
set -u

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
export XTFT_DATA=${XTFT_SCRATCH}/data
export OMP_NUM_THREADS=8
OUT=${XTFT_SCRATCH}/results/camels_smoke_${SLURM_JOB_ID}
mkdir -p "${OUT}"
cd "${SLURM_SUBMIT_DIR}"

python -c "
import sys; sys.path.insert(0, 'benchmarks')
import datasets
d = datasets.load('CAMELS-US')
print('basins', d.df.unique_id.nunique(), 'rows', len(d.df))
print('missing y share', round(float(d.df.y.isna().mean()), 4))
print('static', d.static)
print('past', d.past)
print('known', d.known)
"

COMMON="--dataset CAMELS-US --horizon 7 --input-size 365 \
--max-series 20 --max-epochs 1 --steps-per-epoch 30 \
--max-steps 30 --patience 1 --batch-size 64 --threads 8 \
--out-dir ${OUT}"
for m in xtft nf:LSTM lgb sf:SeasonalNaive sf:Naive; do
    echo "== ${m}"
    # shellcheck disable=SC2086
    python benchmarks/run.py --model "${m}" --seed 0 ${COMMON} \
        2>&1 | tail -4
done
echo "== nf:TFT width 64"
# shellcheck disable=SC2086
python benchmarks/run.py --model nf:TFT --seed 0 ${COMMON} \
    --model-kw '{"hidden_size":64}' --tag w64 2>&1 | tail -4
ls "${OUT}"
