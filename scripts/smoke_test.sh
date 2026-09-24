#!/bin/bash
#SBATCH --job-name=inp-smoke
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# One-GPU setup smoke test. This may download the DINOv2 backbone once.

set -Eeuo pipefail

source "${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}/scripts/env.sh"
PROJECT_ROOT="${INPFORMER_ROOT}"
LOG="${PROJECT_ROOT}/logs/setup-smoke-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-manual}.log"

mkdir -p "${PROJECT_ROOT}/logs"
exec > >(tee -a "${LOG}") 2>&1
trap 'status=$?; echo "SETUP FAILED: smoke test failed (exit ${status})"; exit "${status}"' ERR
exec 9>"${PROJECT_ROOT}/logs/.setup-smoke.lock"
flock --nonblock 9 || { echo "Another smoke-test job is already running." >&2; false; }

[[ -d "${PROJECT_ROOT}/data/mvtec_ad/bottle/train/good" ]]
[[ -d "${PROJECT_ROOT}/data/visa/1cls/candle/train/good" ]]

inp_run "${PROJECT_ROOT}" \
    python scripts/smoke_test.py

echo "SETUP SUCCESS"
