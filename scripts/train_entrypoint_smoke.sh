#!/bin/bash
#SBATCH --job-name=inp-train-smoke
#SBATCH --output=logs/inp-train-smoke-%j.out
#SBATCH --error=logs/inp-train-smoke-%j.err
#SBATCH --partition=frida
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=aga
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -Eeuo pipefail

BASE_DIR="${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
CONTAINER="${BASE_DIR}/containers/inpformer_env.sqfs"
SOURCE_DATA="${BASE_DIR}/data/mvtec_ad"
SMOKE_DATA="${BASE_DIR}/data/.training_smoke/mvtec_ad_full_test"
SAVE_ROOT="${BASE_DIR}/saved_results/training_smoke"
SAVE_NAME="INP-Former-Training-Smoke-v2"
RUN_NAME="${SAVE_NAME}_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=224_Crop=196_INP_num=6_y=0_lambda=0.0_seed=0_INP"
CHECKPOINT="${SAVE_ROOT}/${RUN_NAME}/model.pth"

items=(carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper)

cd "${BASE_DIR}"
mkdir -p logs "${SAVE_ROOT}"

if [[ ! -r "${CONTAINER}" ]]; then
    echo "TRAINING SMOKE FAILED: unreadable container: ${CONTAINER}" >&2
    exit 1
fi

# Build a non-destructive, symlink-only smoke dataset. One normal train image per
# category gives one optimizer step. The full test split is linked read-only so
# the normal AU-PRO evaluation and checkpoint-save branch have valid statistics.
for item in "${items[@]}"; do
    train_image="$(find "${SOURCE_DATA}/${item}/train/good" -maxdepth 1 -type f -print -quit)"
    if [[ ! -f "${train_image}" || ! -d "${SOURCE_DATA}/${item}/test" || ! -d "${SOURCE_DATA}/${item}/ground_truth" ]]; then
        echo "TRAINING SMOKE FAILED: incomplete source category: ${item}" >&2
        exit 1
    fi

    mkdir -p "${SMOKE_DATA}/${item}/train/good"
    ln -sfn "${train_image}" "${SMOKE_DATA}/${item}/train/good/000.png"
    ln -sfn "${SOURCE_DATA}/${item}/test" "${SMOKE_DATA}/${item}/test"
    ln -sfn "${SOURCE_DATA}/${item}/ground_truth" "${SMOKE_DATA}/${item}/ground_truth"
done

if [[ -e "${CHECKPOINT}" ]]; then
    echo "TRAINING SMOKE FAILED: refusing to overwrite existing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

echo "TRAINING SMOKE: actual INP_Former_Multi_Class.py entry point"
echo "WORKLOAD: 15 train images, one batch, one epoch; full read-only test split"
echo "STARTED: $(date --iso-8601=seconds)"

srun \
    --container-image="${CONTAINER}" \
    --container-mounts=/shared:/shared \
    --container-workdir="${BASE_DIR}" \
    python INP_Former_Multi_Class.py \
        --dataset MVTec-AD \
        --data_path "${SMOKE_DATA}" \
        --save_dir "${SAVE_ROOT}" \
        --save_name "${SAVE_NAME}" \
        --encoder dinov2reg_vit_base_14 \
        --input_size 224 \
        --crop_size 196 \
        --INP_num 6 \
        --batch_size 15 \
        --total_epochs 1 \
        --phase train \
        --loss_y 0 \
        --loss_lambda 0.0 \
        --seed 0

if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "TRAINING SMOKE FAILED: checkpoint was not created: ${CHECKPOINT}" >&2
    exit 1
fi

echo "TRAINING SMOKE SUCCESS"
echo "BACKWARD/OPTIMIZER: completed by the training entry point"
echo "CHECKPOINT: ${CHECKPOINT}"
echo "FINISHED: $(date --iso-8601=seconds)"
