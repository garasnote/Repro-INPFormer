#!/bin/bash
#SBATCH --job-name=ad2-export
#SBATCH --output=logs/ad2-export-%j.out
#SBATCH --error=logs/ad2-export-%j.err
#SBATCH --partition=frida
#SBATCH --gres=gpu:L4:1
#SBATCH --exclude=aga
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository directory}"
m="${1:?M required}"
checkpoint="${2:?Checkpoint required}"
output="${3:?New output directory required}"
geometry="${4:?Explicit geometry required}"
split="${5:-test_public}"
srun --container-image="$PWD/containers/inpformer_env.sqfs" \
    --container-mounts=/shared:/shared --container-workdir="$PWD" \
    python export_mvtec_predictions.py --dataset AD2 --data-root "$PWD/data/mvtec_ad2" \
    --checkpoint "$checkpoint" --output "$output" --split "$split" \
    --m "$m" --geometry "$geometry" --require-epoch200
