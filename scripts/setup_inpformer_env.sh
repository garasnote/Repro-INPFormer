#!/bin/bash
# Build INP-Former container on FRIDA — from scratch
# Usage: bash setup_inpformer_env.sh

set -e
CONTAINER_OUT="/shared/workspace/lkm/juan.osorio/container/inpformer_env.sqfs"

echo "========================================="
echo "Building INP-Former Environment (fresh)"
echo "========================================="

srun \
  --partition=dev \
  --time=01:00:00 \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --container-image=nvcr.io#nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04 \
  --container-mounts=/shared:/shared \
  --container-workdir=/shared/home/juan.osorio/ml \
  --container-save="${CONTAINER_OUT}" \
  bash -c '
    set -e

    echo "Installing system deps..."
    apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
    ln -sf /usr/bin/python3 /usr/bin/python

    echo ""
    echo "Installing PyTorch (CUDA 12.1)..."
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

    echo ""
    echo "Installing INP-Former deps..."
    pip install --no-cache-dir \
        "numpy>=1.24,<2" \
        timm==0.9.12 \
        kornia \
        adeval \
        gdown \
        scikit-learn \
        scikit-image \
        "opencv-python-headless>=4.8,<4.10" \
        pandas \
        tabulate \
        matplotlib \
        tqdm \
        Pillow

    # Patch DictValue stub if present
    TYPING_FILE=$(python -c "import cv2, os; print(os.path.join(os.path.dirname(cv2.__file__), 'typing', '__init__.py'))" 2>/dev/null)
    if [ -n "${TYPING_FILE}" ] && grep -q "DictValue" "${TYPING_FILE}" 2>/dev/null; then
        sed -i "s/LayerId = cv2.dnn.DictValue/LayerId = int/" "${TYPING_FILE}"
        echo "  Patched cv2 DictValue stub"
    fi

    echo ""
    echo "Verification:"
    python -c "import numpy; print(f\"  numpy: {numpy.__version__}\")"
    python -c "import timm; print(f\"  timm: {timm.__version__}\")"
    python -c "import kornia; print(f\"  kornia: {kornia.__version__}\")"
    python -c "import cv2; print(f\"  opencv: {cv2.__version__}\")"
    python -c "from adeval import EvalAccumulatorCuda; print(f\"  adeval: OK\")"
    python -c "import torch; print(f\"  torch: {torch.__version__}\")"
    python -c "import torch; assert torch.cuda.is_available(), \"NO CUDA\""
    echo ""
    echo "========================================="
    echo "INP-Former environment ready."
    echo "========================================="
  '

echo ""
echo "Container saved to: ${CONTAINER_OUT}"
