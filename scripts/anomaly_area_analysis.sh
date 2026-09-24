#!/bin/bash
#SBATCH --job-name=area-analysis
#SBATCH --output=logs/area-analysis-%j.out
#SBATCH --error=logs/area-analysis-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Anomaly area vs detection performance analysis.
# Tests hypothesis: large anomalies degrade INP extraction → worse detection.
# Usage: sbatch anomaly_area_analysis.sh

source "${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}/scripts/env.sh"
BASE_DIR="${INPFORMER_ROOT}"

inp_run "${BASE_DIR}" \
    bash -c '

sleep 10
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Started: $(date)"

MVTEC_WEIGHTS="saved_results/INP-Former-Multi-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth"
VISA_WEIGHTS="saved_results/INP-Former-Multi-Class_dataset=VisA_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth"

echo ""
echo "============================================"
echo ">>> MVTec-AD area analysis | $(date)"
echo "============================================"
python anomaly_area_analysis.py \
    --dataset MVTec-AD \
    --data_path data/mvtec_ad \
    --weights "${MVTEC_WEIGHTS}" 2>&1

echo ""
echo "============================================"
echo ">>> VisA area analysis | $(date)"
echo "============================================"
python anomaly_area_analysis.py \
    --dataset VisA \
    --data_path data/visa/1cls \
    --weights "${VISA_WEIGHTS}" 2>&1

echo ""
echo ">>> Area analysis complete | $(date)"
'
