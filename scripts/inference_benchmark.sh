#!/bin/bash
#SBATCH --job-name=infer-bench
#SBATCH --output=logs/infer-bench-%j.out
#SBATCH --error=logs/infer-bench-%j.out
#SBATCH --partition=frida
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Inference speed benchmark: INP count, resolution, backbone sweeps.
# Pure forward-pass timing — no training, no data loading.
# Usage: sbatch inference_benchmark.sh

BASE_DIR="${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR}}"
CONTAINER="${BASE_DIR}/containers/inpformer_env.sqfs"

srun \
    --container-image="${CONTAINER}" \
    --container-mounts=/shared:/shared \
    --container-workdir="${BASE_DIR}" \
    bash -c '

sleep 10
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "Started: $(date)"
echo ""

python inference_benchmark.py --warmup 50 --runs 200 2>&1

echo ""
echo ">>> Inference benchmark complete | $(date)"
'
