#!/bin/bash
#SBATCH --job-name=mvtec-inp-abl
#SBATCH --output=logs/mvtec-inp-abl-%A_%a.out
#SBATCH --error=logs/mvtec-inp-abl-%A_%a.err
#SBATCH --partition=frida
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=aga
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-9%4

set -Eeuo pipefail

BASE_DIR="${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
CONTAINER="${BASE_DIR}/containers/inpformer_env.sqfs"
DATA_PATH="${BASE_DIR}/data/mvtec_ad"
SAVE_ROOT="${BASE_DIR}/saved_results"

cd "${BASE_DIR}"
mkdir -p logs "${SAVE_ROOT}"

if [[ ! -r "${CONTAINER}" ]]; then
    echo "TRAINING FAILED: unreadable container: ${CONTAINER}" >&2
    exit 1
fi
if [[ ! -d "${DATA_PATH}/bottle/train/good" ]]; then
    echo "TRAINING FAILED: MVTec AD is missing or incomplete: ${DATA_PATH}" >&2
    exit 1
fi

task_id="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array}"
if (( task_id < 5 )); then
    config="inp_only"
    seed="${task_id}"
    loss_lambda="0.0"
else
    config="inp_lc"
    seed="$((task_id - 5))"
    loss_lambda="0.2"
fi

loss_y=0
run_name="INP-Former-Multi-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6_y=${loss_y}_lambda=${loss_lambda}_seed=${seed}_INP"
run_dir="${SAVE_ROOT}/${run_name}"
checkpoint="${run_dir}/model.pth"

echo "RUN CONFIG: ${config}"
echo "SEED: ${seed}"
echo "INP: on"
echo "Lc weight: ${loss_lambda}"
echo "Lsm exponent: ${loss_y}"
echo "OUTPUT: ${run_dir}"
echo "STARTED: $(date --iso-8601=seconds)"

if [[ -e "${checkpoint}" ]]; then
    echo "TRAINING FAILED: refusing to overwrite existing checkpoint: ${checkpoint}" >&2
    exit 1
fi

srun \
    --container-image="${CONTAINER}" \
    --container-mounts=/shared:/shared \
    --container-workdir="${BASE_DIR}" \
    python INP_Former_Multi_Class.py \
        --dataset MVTec-AD \
        --data_path "${DATA_PATH}" \
        --save_dir "${SAVE_ROOT}" \
        --save_name INP-Former-Multi-Class \
        --encoder dinov2reg_vit_base_14 \
        --input_size 448 \
        --crop_size 392 \
        --INP_num 6 \
        --batch_size 16 \
        --total_epochs 200 \
        --phase train \
        --loss_y "${loss_y}" \
        --loss_lambda "${loss_lambda}" \
        --seed "${seed}"

if [[ ! -s "${checkpoint}" ]]; then
    echo "TRAINING FAILED: checkpoint was not created: ${checkpoint}" >&2
    exit 1
fi

echo "TRAINING SUCCESS: ${config}, seed=${seed}"
echo "CHECKPOINT: ${checkpoint}"
echo "FINISHED: $(date --iso-8601=seconds)"
