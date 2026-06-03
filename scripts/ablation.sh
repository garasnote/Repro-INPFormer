#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --output=/shared/home/juan.osorio/ml/logs/ablation-%j.out
#SBATCH --error=/shared/home/juan.osorio/ml/logs/ablation-%j.out
#SBATCH --partition=frida
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00

# Ablation experiments (loss ablation moved to ablation_loss_seeds.sh):
#   1. INP count: M = 2, 4, 8, 12, 16 (6 is baseline from loss seeds)
#   2. Backbone: ViT-Small (ViT-Base is baseline)
#   3. Resolution ablation: 224/196, 336/294 (448/392 is baseline)
#   4. Epoch efficiency: train 200 epochs with evals at 25, 50, 100, 150
# Usage: sbatch ablation.sh

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

COMMON_MVTEC="--dataset MVTec-AD --data_path ../data/mvtec_anomaly_detection --input_size 448 --crop_size 392 --batch_size 16 --total_epochs 200 --phase train"
COMMON_AD2="--dataset MVTec-AD2 --data_path ../data/mvtec_ad_2 --input_size 448 --crop_size 392 --batch_size 16 --total_epochs 200 --phase train"

run() {
    echo ""
    echo "============================================"
    echo ">>> $1 | $(date)"
    echo "============================================"
    shift
    python INP_Former_Multi_Class.py "$@" 2>&1
    echo ">>> Finished: $(date)"
}

# --- INP count ablation on MVTec-AD2 (paper already covers MVTec/VisA) ---
run "INP COUNT AD2: M=1"  $COMMON_AD2 --encoder dinov2reg_vit_base_14 --INP_num 1
run "INP COUNT AD2: M=2"  $COMMON_AD2 --encoder dinov2reg_vit_base_14 --INP_num 2
run "INP COUNT AD2: M=4"  $COMMON_AD2 --encoder dinov2reg_vit_base_14 --INP_num 4
run "INP COUNT AD2: M=6"  $COMMON_AD2 --encoder dinov2reg_vit_base_14 --INP_num 6
run "INP COUNT AD2: M=8"  $COMMON_AD2 --encoder dinov2reg_vit_base_14 --INP_num 8
run "INP COUNT AD2: M=12" $COMMON_AD2 --encoder dinov2reg_vit_base_14 --INP_num 12
run "INP COUNT AD2: M=16" $COMMON_AD2 --encoder dinov2reg_vit_base_14 --INP_num 16

# --- Backbone ablation (skip base, covered by loss seeds baseline) ---
run "BACKBONE: ViT-Small" $COMMON_MVTEC --encoder dinov2reg_vit_small_14 --INP_num 6

# --- Resolution ablation (skip 448/392, covered by loss seeds baseline) ---
run "RESOLUTION: 224/196" $COMMON_MVTEC --encoder dinov2reg_vit_base_14 --INP_num 6 --input_size 224 --crop_size 196
run "RESOLUTION: 336/294" $COMMON_MVTEC --encoder dinov2reg_vit_base_14 --INP_num 6 --input_size 336 --crop_size 294

# --- Epoch efficiency (full 200 with intermediate evals) ---
run "EPOCH EFFICIENCY: MVTec-AD" $COMMON_MVTEC --encoder dinov2reg_vit_base_14 --INP_num 6 --eval_epochs 25,50,100,150

echo ""
echo "============================================"
echo ">>> ALL ABLATIONS COMPLETE | $(date)"
echo "============================================"
'
