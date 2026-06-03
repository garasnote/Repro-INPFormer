#!/bin/bash
#SBATCH --job-name=lightshift
#SBATCH --output=/shared/home/juan.osorio/ml/logs/lightshift-%j.out
#SBATCH --error=/shared/home/juan.osorio/ml/logs/lightshift-%j.err
#SBATCH --partition=frida
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Light shift robustness analysis on MVTec AD 2
# Uses already-trained multi-class model, evaluates per lighting condition
# Usage: sbatch lightshift_analysis.sh

CONTAINER="/shared/workspace/lkm/juan.osorio/container/inpformer_env.sqfs"
BASE_DIR="/shared/home/juan.osorio/ml"

srun \
    --container-image="${CONTAINER}" \
    --container-mounts=/shared:/shared \
    --container-workdir="${BASE_DIR}/INP-Former" \
    python lightshift_analysis.py \
        --data_path ../data/mvtec_ad_2 \
        --batch_size 16
