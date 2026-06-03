#!/bin/bash
#SBATCH --job-name=abl-loss-seed
#SBATCH --output=/shared/home/juan.osorio/ml/logs/abl-loss-seed-%j.out
#SBATCH --error=/shared/home/juan.osorio/ml/logs/abl-loss-seed-%j.out
#SBATCH --partition=frida
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00

# Multi-seed module/loss ablation matching paper Table 5
# 5 configs × 5 seeds × 2 datasets = 50 training runs (200 epochs each)
# ~5h per run on GPU → ~125h per dataset → split across jobs.
#
# Configs (matching paper Table 5 rows + extra):
#   1. no_inp:         ✗ INP  ✗ ℒc  ✗ ℒsm  — bottleneck + self-attn decoder
#   2. inp_only:       ✓ INP  ✗ ℒc  ✗ ℒsm  — INP mechanism, plain cosine loss
#   3. inp_lc:         ✓ INP  ✓ ℒc  ✗ ℒsm  — INP + coherence
#   4. inp_lsm:        ✓ INP  ✗ ℒc  ✓ ℒsm  — INP + soft mining (NOT in paper)
#   5. full:           ✓ INP  ✓ ℒc  ✓ ℒsm  — full model
#
# Set DATASET_IDX (0=MVTec-AD, 1=VisA) and CONFIG_RANGE (e.g. "0 1 2" or "3 4")
# to split across multiple jobs. Defaults to all.
#
# Usage:
#   sbatch --export=DATASET_IDX=0,CONFIG_RANGE="0 1 2" ablation_loss_seeds.sh   # MVTec configs 0-2
#   sbatch --export=DATASET_IDX=0,CONFIG_RANGE="3 4" ablation_loss_seeds.sh     # MVTec configs 3-4
#   sbatch --export=DATASET_IDX=1,CONFIG_RANGE="0 1 2" ablation_loss_seeds.sh   # VisA configs 0-2
#   sbatch --export=DATASET_IDX=1,CONFIG_RANGE="3 4" ablation_loss_seeds.sh     # VisA configs 3-4

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

COMMON="--encoder dinov2reg_vit_base_14 --input_size 448 --crop_size 392 --INP_num 6 --batch_size 16 --total_epochs 200 --phase train"

declare -a NAMES=(  "no_inp"    "inp_only"  "inp_lc"    "inp_lsm"   "full")
declare -a NO_INP=( "--no_inp"  ""          ""          ""          "")
declare -a Y_VALS=( 0           0           0           3           3)
declare -a L_VALS=( 0.0         0.0         0.2         0.0         0.2)
declare -a SEEDS=(1 2 3 42 123)

declare -a DATASETS=("MVTec-AD" "VisA")
declare -a DATAPATHS=("../data/mvtec_anomaly_detection" "../data/VisA_pytorch/1cls")

# Use env vars to select subset, default to all
D_LIST="${DATASET_IDX:-0 1}"
C_LIST="${CONFIG_RANGE:-0 1 2 3 4}"
echo "Running datasets: ${D_LIST}, configs: ${C_LIST}"

for d in ${D_LIST}; do
    DS="${DATASETS[$d]}"
    DP="${DATAPATHS[$d]}"
    for i in ${C_LIST}; do
        for SEED in "${SEEDS[@]}"; do
            NAME="${NAMES[$i]}"
            Y="${Y_VALS[$i]}"
            L="${L_VALS[$i]}"
            INP_FLAG="${NO_INP[$i]}"

            INP_TAG="INP"
            if [ -n "${INP_FLAG}" ]; then INP_TAG="noINP"; fi
            SAVE_DIR="saved_results/INP-Former-Multi-Class_dataset=${DS}_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6_y=${Y}_lambda=${L}_seed=${SEED}_${INP_TAG}"

            if [ -f "${SAVE_DIR}/model.pth" ]; then
                echo ">>> SKIP: ${DS} | ${NAME} | seed=${SEED} (already done)"
                continue
            fi

            echo ""
            echo "============================================"
            echo ">>> ${DS} | ${NAME} | seed=${SEED} | y=${Y} lambda=${L} ${INP_FLAG}"
            echo ">>> $(date)"
            echo "============================================"

            python INP_Former_Multi_Class.py $COMMON \
                --dataset ${DS} --data_path ${DP} \
                --loss_y ${Y} --loss_lambda ${L} --seed ${SEED} ${INP_FLAG} 2>&1

            echo ">>> Finished: $(date)"
        done
    done
done

echo ""
echo "============================================"
echo ">>> Multi-seed ablation complete (5 configs × 5 seeds × 2 datasets)."
echo ">>> $(date)"
echo "============================================"
'
