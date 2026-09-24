#!/bin/bash
#SBATCH --job-name=setup-inpformer
#SBATCH --output=logs/setup-container-%j.out
#SBATCH --error=logs/setup-container-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Build the INP-Former Enroot container on a Slurm cluster with the Pyxis plugin.
# Without Pyxis, skip this and install requirements.txt into a Python env instead.
# Submit from the repository root: sbatch scripts/setup_inpformer_env.sh

set -Eeuo pipefail

source "${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}/scripts/env.sh"
PROJECT_ROOT="${INPFORMER_ROOT}"
CONTAINER_OUT="${INPFORMER_CONTAINER}"
LOG="${PROJECT_ROOT}/logs/setup-container-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-manual}.log"

mkdir -p "$(dirname -- "${CONTAINER_OUT}")" "${PROJECT_ROOT}/logs"
exec > >(tee -a "${LOG}") 2>&1
trap 'status=$?; echo "SETUP FAILED: container build failed (exit ${status})"; exit "${status}"' ERR
exec 9>"${PROJECT_ROOT}/logs/.setup-container.lock"
flock --nonblock 9 || { echo "Another container setup job is already running." >&2; false; }

if [[ -e "${CONTAINER_OUT}" ]]; then
    echo "Container already exists: ${CONTAINER_OUT}" >&2
    echo "Move it aside before rebuilding." >&2
    false
fi

inp_has_pyxis || { echo "srun has no --container-image (Pyxis); use a native Python env instead." >&2; false; }

echo ">>> Building container: ${CONTAINER_OUT}"
srun \
  --container-image=nvcr.io#nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04 \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  --container-save="${CONTAINER_OUT}" \
  bash -c '
    set -Eeuo pipefail

    apt-get update
    apt-get install -y --no-install-recommends \
        ca-certificates git libgl1 libglib2.0-0 python3 python3-dev python3-pip wget
    rm -rf /var/lib/apt/lists/*
    ln -sf /usr/bin/python3 /usr/bin/python

    pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121
    pip install --no-cache-dir \
        "numpy>=1.24,<2" \
        adeval \
        gdown \
        huggingface_hub \
        kornia \
        matplotlib \
        "opencv-python-headless>=4.8,<4.10" \
        pandas \
        Pillow \
        scikit-image \
        scikit-learn \
        scipy \
        tabulate \
        timm==0.9.12 \
        tqdm

    typing_file=$(python -c "import cv2, os; print(os.path.join(os.path.dirname(cv2.__file__), '\''typing'\'', '\''__init__.py'\''))" 2>/dev/null)
    if [[ -n "${typing_file}" ]] && grep -q "DictValue" "${typing_file}" 2>/dev/null; then
        sed -i "s/LayerId = cv2.dnn.DictValue/LayerId = int/" "${typing_file}"
    fi

    python -c "import adeval, cv2, huggingface_hub, kornia, numpy, scipy, sklearn, timm, torch, torchvision"
    python -c "import torch; assert torch.cuda.is_available(), '\''CUDA is not visible inside the container'\''; print(torch.cuda.get_device_name(0))"
  '

[[ -r "${CONTAINER_OUT}" ]]
echo ">>> Container verified: ${CONTAINER_OUT}"
echo "SETUP SUCCESS"
