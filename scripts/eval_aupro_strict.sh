#!/bin/bash
#SBATCH --job-name=aupro-strict
#SBATCH --output=/shared/home/juan.osorio/ml/logs/aupro-strict-%j.out
#SBATCH --error=/shared/home/juan.osorio/ml/logs/aupro-strict-%j.out
#SBATCH --partition=frida
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Evaluate AU-PRO at FPR limits 0.30 and 0.05.
# CPU-based compute_pro is slow but gives both metrics.
# Usage: sbatch eval_aupro_strict.sh

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

AD2_WEIGHTS="saved_results/INP-Former-Multi-Class_dataset=MVTec-AD2_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6_y=3_lambda=0.2/model.pth"
MVTEC_WEIGHTS="saved_results/INP-Former-Multi-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth"

echo ""
echo "============================================"
echo ">>> MVTec-AD2: AU-PRO 0.30 + 0.05 | $(date)"
echo "============================================"
python eval_aupro_strict.py \
    --dataset MVTec-AD2 \
    --data_path ../data/mvtec_ad_2 \
    --weights "${AD2_WEIGHTS}" \
    --num_th 300 2>&1

echo ""
echo "============================================"
echo ">>> MVTec-AD: AU-PRO 0.30 + 0.05 | $(date)"
echo "============================================"
python eval_aupro_strict.py \
    --dataset MVTec-AD \
    --data_path ../data/mvtec_anomaly_detection \
    --weights "${MVTEC_WEIGHTS}" \
    --num_th 300 2>&1

echo ""
echo ">>> AU-PRO evaluation complete | $(date)"
'
