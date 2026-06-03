#!/bin/bash
#SBATCH --job-name=infer-bench
#SBATCH --output=/shared/home/juan.osorio/ml/logs/infer-bench-%j.out
#SBATCH --error=/shared/home/juan.osorio/ml/logs/infer-bench-%j.out
#SBATCH --partition=frida
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Inference speed benchmark: INP count, resolution, backbone sweeps.
# Pure forward-pass timing — no training, no data loading.
# Usage: sbatch inference_benchmark.sh

CONTAINER="/shared/workspace/lkm/juan.osorio/container/inpformer_env.sqfs"
BASE_DIR="/shared/home/juan.osorio/ml"

srun \
    --container-image="${CONTAINER}" \
    --container-mounts=/shared:/shared \
    --container-workdir="${BASE_DIR}/INP-Former" \
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
