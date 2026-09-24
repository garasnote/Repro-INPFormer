#!/bin/bash
#SBATCH --job-name=inp-smoke
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# One-GPU setup smoke test. This may download the DINOv2 backbone once.

set -Eeuo pipefail

PROJECT_ROOT="${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
CONTAINER="${PROJECT_ROOT}/containers/inpformer_env.sqfs"
LOG="${PROJECT_ROOT}/logs/setup-smoke-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-manual}.log"

mkdir -p "${PROJECT_ROOT}/logs"
exec > >(tee -a "${LOG}") 2>&1
trap 'status=$?; echo "SETUP FAILED: smoke test failed (exit ${status})"; exit "${status}"' ERR
exec 9>"${PROJECT_ROOT}/logs/.setup-smoke.lock"
flock --nonblock 9 || { echo "Another smoke-test job is already running." >&2; false; }

[[ -r "${CONTAINER}" ]]
[[ -d "${PROJECT_ROOT}/data/mvtec_ad/bottle/train/good" ]]
[[ -d "${PROJECT_ROOT}/data/visa/1cls/candle/train/good" ]]

srun \
    --container-image="${CONTAINER}" \
    --container-mounts=/shared:/shared \
    --container-workdir="${PROJECT_ROOT}" \
    python scripts/smoke_test.py

echo "SETUP SUCCESS"
