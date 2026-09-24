#!/bin/bash
#SBATCH --job-name=mvtec-inp-corrective
#SBATCH --output=logs/mvtec-inp-corrective-%A_%a.out
#SBATCH --error=logs/mvtec-inp-corrective-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4

set -Eeuo pipefail

source "${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}/scripts/env.sh"
BASE_DIR="${INPFORMER_ROOT}"
DATA_PATH="${BASE_DIR}/data/mvtec_ad"
SAVE_ROOT="${BASE_DIR}/saved_results"

cd "${BASE_DIR}"
mkdir -p logs "${SAVE_ROOT}"

if [[ ! -d "${DATA_PATH}/bottle/train/good" ]]; then
    echo "TRAINING FAILED: MVTec AD is missing or incomplete: ${DATA_PATH}" >&2
    exit 1
fi

task_id="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array}"
case "${task_id}" in
    0) config="inp_only"; seed=42;  loss_lambda="0.0" ;;
    1) config="inp_only"; seed=123; loss_lambda="0.0" ;;
    2) config="inp_lc";   seed=42;  loss_lambda="0.2" ;;
    3) config="inp_lc";   seed=123; loss_lambda="0.2" ;;
    *) echo "TRAINING FAILED: unexpected array task ${task_id}" >&2; exit 1 ;;
esac

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

inp_run "${BASE_DIR}" \
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
