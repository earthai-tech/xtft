#!/bin/bash
# Speed and size: PyTorch XTFT vs Keras (fusionlab) XTFT on one
# full CPU node. Submit from ~/projects/xtft:
#   sbatch jobs/speed_cpu.sbatch
#SBATCH --job-name=xtft-speed
#SBATCH --partition=B720-ibe-compute_28_64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/xtft-speed-%j.out
#SBATCH --error=logs/xtft-speed-%j.err

set -euo pipefail

set +u
source /SCRATCH/kouakloMI/miniforge3/etc/profile.d/conda.sh
conda activate /SCRATCH/kouakloMI/envs/xtft
set -u

export TF_CPP_MIN_LOG_LEVEL=2
OUT=/SCRATCH/kouakloMI/xtft/speed
mkdir -p "${OUT}"
cd "${SLURM_SUBMIT_DIR}"
echo "node=$(hostname) job=${SLURM_JOB_ID} commit=$(git rev-parse --short HEAD)"

# 8 threads (a typical shared allocation) and the full node.
for threads in 8 28; do
    export OMP_NUM_THREADS="${threads}"
    export MKL_NUM_THREADS="${threads}"
    python benchmarks/speed/compare_xtft.py \
        --steps 16 64 168 --batch 64 --threads "${threads}" \
        --warmup 3 --repeats 10 \
        --out "${OUT}/speed_t${threads}_${SLURM_JOB_ID}.json"
done
