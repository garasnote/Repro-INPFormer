#!/bin/bash
#SBATCH --job-name=lora-ft
#SBATCH --output=/shared/home/juan.osorio/ml/logs/lora-ft-%j.out
#SBATCH --error=/shared/home/juan.osorio/ml/logs/lora-ft-%j.out
#SBATCH --partition=frida
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# LoRA finetuning: Real-IAD pretrained → MVTec-AD and VisA
# Tests parameter-efficient transfer across datasets.
# Compares: zero-shot (no finetune) vs LoRA-finetuned (50 epochs)
# Also sweeps rank = 2, 4, 8 to find sweet spot.
# Usage: sbatch lora_finetune.sh

CONTAINER="/shared/workspace/lkm/juan.osorio/container/inpformer_env.sqfs"
BASE_DIR="/shared/home/juan.osorio/ml"

srun \
    --container-image="${CONTAINER}" \
    --container-mounts=/shared:/shared \
    --container-workdir="${BASE_DIR}/INP-Former" \
    bash -c '

sleep 10
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Started: $(date)"

# Fix Real-IAD json symlink (needed if source weights require loading Real-IAD data)
if [ ! -e "../data/Real-IAD/realiad_jsons/realiad_jsons" ]; then
    ln -s . "../data/Real-IAD/realiad_jsons/realiad_jsons"
fi

# Download Real-IAD multi-class weights if missing
REALIAD_DIR="saved_results/INP-Former-Multi-Class_dataset=Real-IAD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6"
mkdir -p "${REALIAD_DIR}"
if [ ! -s "${REALIAD_DIR}/model.pth" ]; then
    echo ">>> Downloading Real-IAD pretrained weights..."
    pip install --no-cache-dir gdown 2>&1 | tail -1
    gdown "https://drive.google.com/uc?id=1iwMvUMyslacXgApOJ6DK0A5WXYRCH5NH" \
        -O "${REALIAD_DIR}/model.pth" 2>&1
fi
SOURCE="${REALIAD_DIR}/model.pth"

COMMON="--source_weights ${SOURCE} --encoder dinov2reg_vit_base_14 --INP_num 6 --batch_size 16 --total_epochs 50"

for RANK in ${LORA_RANKS:-2 4 8}; do
    echo ""
    echo "============================================"
    echo ">>> Real-IAD → MVTec-AD | LoRA rank=${RANK} | $(date)"
    echo "============================================"
    python lora_finetune.py $COMMON \
        --dataset MVTec-AD \
        --data_path ../data/mvtec_anomaly_detection \
        --lora_rank ${RANK} 2>&1
    echo ">>> Finished: $(date)"

    echo ""
    echo "============================================"
    echo ">>> Real-IAD → VisA | LoRA rank=${RANK} | $(date)"
    echo "============================================"
    python lora_finetune.py $COMMON \
        --dataset VisA \
        --data_path ../data/VisA_pytorch/1cls \
        --lora_rank ${RANK} 2>&1
    echo ">>> Finished: $(date)"
done

echo ""
echo "============================================"
echo ">>> LoRA finetuning complete | $(date)"
echo "============================================"
'
