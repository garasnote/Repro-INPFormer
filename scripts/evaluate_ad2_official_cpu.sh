#!/bin/bash
#SBATCH --job-name=ad2-official-cpu
#SBATCH --output=logs/ad2-official-cpu-%j.out
#SBATCH --error=logs/ad2-official-cpu-%j.err
#SBATCH --partition=frida
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository directory}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
manifest="${1:?Manifest required}"
output="${2:?Metrics output required}"
srun --container-image="$PWD/containers/inpformer_env.sqfs" \
    --container-mounts=/shared:/shared --container-workdir="$PWD" \
    python evaluate_mvtec_official.py --manifest "$manifest" --output "$output" \
    --fpr-limits 0.05 0.30 --max-memory-gib 100
