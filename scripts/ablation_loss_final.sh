#!/bin/bash
#SBATCH --job-name=abl-final
#SBATCH --output=/shared/home/juan.osorio/ml/logs/abl-final-%j.out
#SBATCH --error=/shared/home/juan.osorio/ml/logs/abl-final-%j.out
#SBATCH --partition=frida
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00

# Finish remaining seed ablation on MVTec-AD only:
#   1. no_inp seed=123       (1 run)
#   2. full (y=3,λ=0.2) ×5  (5 runs)
# 6 runs × ~15h = ~90h. Skip-if-exists for safe resubmission.

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

COMMON="--encoder dinov2reg_vit_base_14 --input_size 448 --crop_size 392 --INP_num 6 --batch_size 16 --total_epochs 200 --phase train --dataset MVTec-AD --data_path ../data/mvtec_anomaly_detection"

run() {
    local DESC="$1"; shift
    local SAVE_DIR="$1"; shift

    if [ -f "${SAVE_DIR}/model.pth" ]; then
        echo ">>> SKIP: ${DESC} (already done)"
        return
    fi

    echo ""
    echo "============================================"
    echo ">>> ${DESC} | $(date)"
    echo "============================================"
    python INP_Former_Multi_Class.py "$@" 2>&1
    echo ">>> Finished: $(date)"
}

# 1. no_inp seed=123
run "no_inp seed=123" \
    "saved_results/INP-Former-Multi-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6_y=0_lambda=0.0_seed=123_noINP" \
    $COMMON --loss_y 0 --loss_lambda 0.0 --seed 123 --no_inp

# 2. full (y=3, lambda=0.2) × 5 seeds
for SEED in 1 2 3 42 123; do
    run "full seed=${SEED}" \
        "saved_results/INP-Former-Multi-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6_y=3_lambda=0.2_seed=${SEED}_INP" \
        $COMMON --loss_y 3 --loss_lambda 0.2 --seed ${SEED}
done

echo ""
echo "============================================"
echo ">>> Final seed ablation complete | $(date)"
echo "============================================"
'
