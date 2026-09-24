#!/bin/bash
#SBATCH --job-name=dl-mvtec
#SBATCH --output=logs/dl-mvtec-%j.out
#SBATCH --error=logs/dl-mvtec-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00

# Download MVTec AD through the project container.
# Submit from the repository root after the container job succeeds.

set -Eeuo pipefail

source "${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}/scripts/env.sh"
PROJECT_ROOT="${INPFORMER_ROOT}"
LOG="${PROJECT_ROOT}/logs/setup-mvtec-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-manual}.log"

mkdir -p "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/logs"
exec > >(tee -a "${LOG}") 2>&1
trap 'status=$?; echo "SETUP FAILED: MVTec AD download failed (exit ${status})"; exit "${status}"' ERR
exec 9>"${PROJECT_ROOT}/logs/.setup-mvtec.lock"
flock --nonblock 9 || { echo "Another MVTec AD setup job is already running." >&2; false; }

export NVIDIA_VISIBLE_DEVICES=void
inp_run "${PROJECT_ROOT}" \
    python scripts/download_mvtec_hf.py

[[ -d "${PROJECT_ROOT}/data/mvtec_ad/bottle/train/good" ]]
[[ -d "${PROJECT_ROOT}/data/mvtec_ad/bottle/test" ]]
[[ -d "${PROJECT_ROOT}/data/mvtec_ad/bottle/ground_truth" ]]
echo "SETUP SUCCESS"
