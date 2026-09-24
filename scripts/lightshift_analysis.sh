#!/bin/bash
#SBATCH --job-name=lightshift
#SBATCH --output=logs/lightshift-%j.out
#SBATCH --error=logs/lightshift-%j.err
#SBATCH --partition=frida
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Light shift robustness analysis on MVTec AD 2
# Uses already-trained multi-class model, evaluates per lighting condition
# Usage: sbatch lightshift_analysis.sh

BASE_DIR="${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR}}"
CONTAINER="${BASE_DIR}/containers/inpformer_env.sqfs"

srun \
    --container-image="${CONTAINER}" \
    --container-mounts=/shared:/shared \
    --container-workdir="${BASE_DIR}" \
    python lightshift_analysis.py \
        --data_path data/mvtec_ad2 \
        --batch_size 16
