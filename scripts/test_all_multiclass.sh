#!/bin/bash
#SBATCH --job-name=test-mc-all
#SBATCH --output=/shared/home/juan.osorio/ml/logs/test-mc-all-%j.out
#SBATCH --error=/shared/home/juan.osorio/ml/logs/test-mc-all-%j.err
#SBATCH --partition=frida
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Test pretrained INP-Former Multi-Class on all datasets
# Downloads missing weights automatically. Trains MVTec-AD2 (no pretrained available).
# Usage: sbatch test_all_multiclass.sh

CONTAINER="/shared/workspace/lkm/juan.osorio/container/inpformer_env.sqfs"
BASE_DIR="/shared/home/juan.osorio/ml"

srun \
    --container-image="${CONTAINER}" \
    --container-mounts=/shared:/shared \
    --container-workdir="${BASE_DIR}/INP-Former" \
    bash -c '
set -e

# Fix Real-IAD json double-nesting
if [ ! -e "../data/Real-IAD/realiad_jsons/realiad_jsons" ]; then
    ln -s . "../data/Real-IAD/realiad_jsons/realiad_jsons"
fi

echo "============================================"
echo ">>> Testing MVTec-AD Multi-Class"
echo "============================================"
MVTEC1_WEIGHTS="saved_results/INP-Former-Multi-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth"
python INP_Former_Multi_Class.py \
    --dataset MVTec-AD \
    --data_path ../data/mvtec_anomaly_detection \
    --phase test \
    --batch_size 16 \
    --load_from "${MVTEC1_WEIGHTS}"

echo ""
echo "============================================"
echo ">>> Testing VisA Multi-Class"
echo "============================================"
VISA_WEIGHTS="saved_results/INP-Former-Multi-Class_dataset=VisA_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth"
python INP_Former_Multi_Class.py \
    --dataset VisA \
    --data_path ../data/VisA_pytorch/1cls \
    --phase test \
    --batch_size 16 \
    --load_from "${VISA_WEIGHTS}"

echo ""
echo "============================================"
echo ">>> Testing Real-IAD Multi-Class"
echo "============================================"
REALIAD_DIR="saved_results/INP-Former-Multi-Class_dataset=Real-IAD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6"
mkdir -p "${REALIAD_DIR}"
if [ ! -s "${REALIAD_DIR}/model.pth" ]; then
    echo ">>> Downloading Real-IAD weights..."
    gdown "https://drive.google.com/uc?id=1iwMvUMyslacXgApOJ6DK0A5WXYRCH5NH" \
        -O "${REALIAD_DIR}/model.pth"
fi

python INP_Former_Multi_Class.py \
    --dataset Real-IAD \
    --data_path ../data/Real-IAD \
    --phase test \
    --batch_size 16 \
    --load_from "${REALIAD_DIR}/model.pth"

echo ""
echo "============================================"
echo ">>> MVTec-AD2: Cross-test with MVTec-AD1 weights"
echo "============================================"
MVTEC1_WEIGHTS="saved_results/INP-Former-Multi-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth"
python INP_Former_Multi_Class.py \
    --dataset MVTec-AD2 \
    --data_path ../data/mvtec_ad_2 \
    --phase test \
    --batch_size 16 \
    --load_from "${MVTEC1_WEIGHTS}"

echo ""
echo "============================================"
echo ">>> MVTec-AD2: Train from scratch + test"
echo "============================================"
python INP_Former_Multi_Class.py \
    --dataset MVTec-AD2 \
    --data_path ../data/mvtec_ad_2 \
    --phase train \
    --batch_size 16 \
    --total_epochs 200

echo ""
echo "============================================"
echo ">>> MVTec-AD2: Single-Class train + test (7 categories)"
echo "============================================"
python INP_Former_Single_Class.py \
    --dataset MVTec-AD2 \
    --data_path ../data/mvtec_ad_2 \
    --phase train \
    --batch_size 16 \
    --total_epochs 200

echo ""
echo "============================================"
echo ">>> All tests complete."
echo "============================================"
'
