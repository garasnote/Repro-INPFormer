#!/bin/bash
#SBATCH --job-name=epoch-eff
#SBATCH --output=/shared/home/juan.osorio/ml/logs/epoch-eff-%j.out
#SBATCH --error=/shared/home/juan.osorio/ml/logs/epoch-eff-%j.out
#SBATCH --partition=frida
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Epoch efficiency: how quickly does INP-Former converge?
# Trains full config (y=3, λ=0.2) for 200 epochs on MVTec-AD,
# evaluating at epochs 25, 50, 100, 150, 200.
# Produces convergence data for the paper (Table/Figure).
# Usage: sbatch --gres=gpu:<TYPE>:1 epoch_efficiency.sh

CONTAINER="/shared/workspace/lkm/juan.osorio/container/inpformer_env.sqfs"
BASE_DIR="/shared/home/juan.osorio/ml"

srun \
    --container-image="${CONTAINER}" \
    --container-mounts=/shared:/shared \
    --container-workdir="${BASE_DIR}/INP-Former" \
    bash -c '

sleep 10
echo "============================================"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Started: $(date)"
echo "============================================"

SAVE_DIR="saved_results/EpochEff_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6_y=3_lambda=0.2_seed=42_INP"

if [ -f "${SAVE_DIR}/model.pth" ]; then
    echo ">>> SKIP: epoch efficiency already completed"
    exit 0
fi

python INP_Former_Multi_Class.py \
    --encoder dinov2reg_vit_base_14 \
    --input_size 448 --crop_size 392 \
    --INP_num 6 --batch_size 16 \
    --total_epochs 200 \
    --eval_epochs 25,50,100,150 \
    --loss_y 3 --loss_lambda 0.2 \
    --seed 42 \
    --save_name EpochEff \
    --phase train \
    --dataset MVTec-AD \
    --data_path ../data/mvtec_anomaly_detection 2>&1

echo ""
echo "============================================"
echo ">>> Epoch efficiency complete | $(date)"
echo "============================================"
'
