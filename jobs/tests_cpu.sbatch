#!/bin/bash
# Run the xtft test suite on one CPU node of wombele17.
# Submit from ~/projects/xtft:  sbatch jobs/tests_cpu.sbatch
# SLURM 15.08: %x is unsupported, so the job name is written
# literally in the log paths.
#SBATCH --job-name=xtft-tests
#SBATCH --partition=B720-ibe-compute_28_64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/xtft-tests-%j.out
#SBATCH --error=logs/xtft-tests-%j.err

set -euo pipefail

# conda's activate scripts reference unset variables.
set +u
source /SCRATCH/kouakloMI/miniforge3/etc/profile.d/conda.sh
conda activate /SCRATCH/kouakloMI/envs/xtft
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${SLURM_SUBMIT_DIR}"
echo "node=$(hostname) job=${SLURM_JOB_ID} commit=$(git rev-parse --short HEAD)"
python -c "import torch, xtft; print('torch', torch.__version__, 'backends', xtft.available_backends())"
python -m pytest -q -p no:cacheprovider
