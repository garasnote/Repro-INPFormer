#!/bin/bash
#SBATCH --job-name=official-export-smoke
#SBATCH --output=logs/official-export-smoke-%j.out
#SBATCH --error=logs/official-export-smoke-%j.err
#SBATCH --partition=frida
#SBATCH --exclude=aga
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -Eeuo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "${SLURM_SUBMIT_DIR:?}"
checkpoint="$PWD/saved_results/INP-Former-Multi-Class-8cat-sweep_dataset=MVTec-AD2_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6_y=3_lambda=0.2_seed=1_INP/model.pth"
for geometry in full-frame legacy-crop256; do
    srun --container-image="$PWD/containers/inpformer_env.sqfs" \
        --container-mounts=/shared:/shared --container-workdir="$PWD" \
        python export_mvtec_predictions.py --dataset AD2 --data-root "$PWD/data/mvtec_ad2" \
        --checkpoint "$checkpoint" --output "$PWD/evaluation_results/smoke-${SLURM_JOB_ID}/$geometry" \
        --split test_public --m 6 --geometry "$geometry" --categories can --smoke-images 2 --batch-size 2 --device cpu
done
srun --container-image="$PWD/containers/inpformer_env.sqfs" \
    --container-mounts=/shared:/shared --container-workdir="$PWD" \
    python scripts/test_mvtec_official_adapter.py
