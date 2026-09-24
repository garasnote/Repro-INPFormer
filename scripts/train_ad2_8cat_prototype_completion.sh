#!/bin/bash
#SBATCH --job-name=ad2-8cat-proto
#SBATCH --output=logs/ad2-8cat-proto-%j.out
#SBATCH --error=logs/ad2-8cat-proto-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(4|8|12|16)$ ]]; then
    echo "Usage: sbatch $0 {4|8|12|16}" >&2
    exit 2
fi

prototype_count="$1"
source "${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}/scripts/env.sh"
base_dir="${INPFORMER_ROOT}"
run_name="INP-Former-Multi-Class-8cat-sweep_dataset=MVTec-AD2_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=${prototype_count}_y=3_lambda=0.2_seed=1_INP"
run_dir="${base_dir}/saved_results/${run_name}"

# Claim this result directory atomically. A duplicate job exits before training.
if ! mkdir "${run_dir}"; then
    echo "Refusing to overwrite existing output: ${run_dir}" >&2
    exit 3
fi

echo "M=${prototype_count}"
echo "Output=${run_dir}"
echo "Started=$(date --iso-8601=seconds)"

inp_run "${base_dir}" \
    python INP_Former_Multi_Class.py \
        --dataset MVTec-AD2 \
        --data_path data/mvtec_ad2 \
        --save_dir saved_results \
        --save_name INP-Former-Multi-Class-8cat-sweep \
        --encoder dinov2reg_vit_base_14 \
        --input_size 448 \
        --crop_size 392 \
        --INP_num "${prototype_count}" \
        --loss_y 3 \
        --loss_lambda 0.2 \
        --seed 1 \
        --batch_size 16 \
        --total_epochs 200 \
        --phase train

echo "Finished=$(date --iso-8601=seconds)"
