# Reproducibility Study: INP-Former (CVPR 2025)

Reproduction and extended analysis of [INP-Former: Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection](https://arxiv.org/abs/2503.02424) (Luo et al., CVPR 2025).

**Course:** MLDS2, University of Ljubljana  
**Paper:** See `paper/main.tex`

## Repository Structure

```
├── INP_Former_Multi_Class.py       # Multi-class training/eval (main)
├── INP_Former_Single_Class.py      # Single-class training/eval
├── INP_Former_Few_Shot.py          # Few-shot training/eval
├── INP_Former_Super_Multi_Class.py # Super multi-class training/eval
├── INP_Former_Zero_Shot.py         # Zero-shot evaluation
├── dataset.py                      # Dataset loading
├── utils.py                        # Utilities, evaluation, losses
├── lora.py                         # [OURS] LoRA adapter implementation
├── lora_finetune.py                # [OURS] LoRA cross-dataset transfer
├── anomaly_area_analysis.py        # [OURS] Anomaly area vs performance
├── eval_aupro_strict.py            # [OURS] AU-PRO at strict FPR
├── lightshift_analysis.py          # [OURS] Lighting robustness eval
├── inference_benchmark.py          # [OURS] Throughput and latency
├── aggregate_results.py            # [OURS] Collect results into tables
├── models/               # Model architecture
├── backbones/            # Encoder weights (auto-downloaded)
├── dinov2/               # DINOv2 backbone (default)
├── optimizers/           # StableAdamW
├── scripts/              # SLURM job scripts (FRIDA cluster)
├── paper/                # Reproducibility report (TMLR format)
├── assets/               # Figures from the original paper
└── EXPERIMENTS.md        # Experiment tracker with all results
```

Files marked `[OURS]` are our additions; everything else is from the [official INP-Former repo](https://github.com/luow23/INP-Former).

## Setup

### Environment

```bash
conda create -n INP python=3.8.12
conda activate INP
pip install -r requirements.txt
```

Tested on NVIDIA A100 (80GB) and L4 (24GB).

### Backbone Weights

The DINOv2 Register ViT-Base/14 backbone is downloaded automatically on first run via `torch.hub`. Weights are cached in `backbones/`. No manual download needed.

### Datasets

Download and place in the **parent directory** of this repo (i.e., `../`):

| Dataset | Source | Target path |
|---------|--------|-------------|
| MVTec-AD | [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad) | `../mvtec_anomaly_detection/` |
| VisA | [amazon-science/spot-diff](https://github.com/amazon-science/spot-diff) | `../VisA_pytorch/1cls/` |
| Real-IAD | [realiad4ad.github.io](https://realiad4ad.github.io/Real-IAD/) | `../Real-IAD/` |
| MVTec-AD2 | [mvtec.com](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) | `../mvtec_anomaly_detection/` (merged) |

For MVTec-AD2, use `scripts/extract_mvtecad2.sh` or `scripts/download_mvtec_hf.py`.
For VisA, preprocess into 1-class format using the [official splitting code](https://github.com/amazon-science/spot-diff).

## Reproducing Results

### 1. Main Reproduction (Multi-Class)

```bash
# MVTec-AD
python INP_Former_Multi_Class.py --dataset MVTec-AD \
    --data_path ../mvtec_anomaly_detection --phase train

# VisA
python INP_Former_Multi_Class.py --dataset VisA \
    --data_path ../VisA_pytorch/1cls --phase train
```

Expected results (I-AUROC / P-AUROC / P-AUPRO):
- MVTec-AD: 99.67 / 98.45 / 94.95
- VisA: 98.90 / 98.97 / 95.27

### 2. Loss/Module Ablation (5 seeds)

Ablation on MVTec-AD with seeds {1, 2, 3, 42, 123}:

```bash
# No INP baseline (y=0, lambda=0)
python INP_Former_Multi_Class.py --dataset MVTec-AD \
    --data_path ../mvtec_anomaly_detection --phase train \
    --INP_num 0 --sm_gamma 0 --seed 1

# INP + Lsm only (y=3, lambda=0)
python INP_Former_Multi_Class.py --dataset MVTec-AD \
    --data_path ../mvtec_anomaly_detection --phase train \
    --sm_gamma 3 --coherence_lambda 0 --seed 1

# Full model (y=3, lambda=0.2) — default
python INP_Former_Multi_Class.py --dataset MVTec-AD \
    --data_path ../mvtec_anomaly_detection --phase train --seed 1
```

Or via SLURM: `sbatch scripts/ablation_loss_final.sh`

### 3. LoRA Transfer Learning

```bash
python lora_finetune.py --dataset MVTec-AD \
    --data_path ../mvtec_anomaly_detection \
    --pretrained_weights saved_results/.../model.pth \
    --lora_rank 4
```

### 4. Analysis Scripts

```bash
# Anomaly area vs detection performance
python anomaly_area_analysis.py --dataset MVTec-AD \
    --data_path ../mvtec_anomaly_detection \
    --weights saved_results/.../model.pth

# AU-PRO at strict FPR (0.05)
python eval_aupro_strict.py --dataset MVTec-AD \
    --data_path ../mvtec_anomaly_detection \
    --weights saved_results/.../model.pth

# Lighting robustness (requires MVTec-AD2)
python lightshift_analysis.py --dataset MVTec-AD \
    --data_path ../mvtec_anomaly_detection \
    --weights saved_results/.../model.pth

# Inference throughput
python inference_benchmark.py --weights saved_results/.../model.pth
```

## SLURM (FRIDA Cluster)

All SLURM scripts are in `scripts/` and use Enroot containers on FRIDA.

```bash
sbatch scripts/setup_inpformer_env.sh       # Build container
sbatch scripts/test_all_multiclass.sh        # Main reproduction
sbatch scripts/ablation_loss_final.sh        # Ablation with seeds
sbatch scripts/lora_finetune.sh              # LoRA experiments
sbatch scripts/epoch_efficiency.sh           # Convergence analysis
```

See `EXPERIMENTS.md` for full experiment status and results.

## Our Contributions

1. **Loss/module ablation with statistical robustness** — 5 seeds per config, including a novel INP+Lsm configuration not tested in the original paper
2. **LoRA transfer learning** — parameter-efficient cross-dataset adaptation at ranks 2, 4, 8
3. **Epoch efficiency analysis** — convergence profiling showing 99.6% of final performance at epoch 50/200
4. **Anomaly area analysis** — per-bin I-AUROC and P-AUPRO revealing opposite trends: detection improves with anomaly size but localization degrades
5. **Lighting robustness** — evaluation under MVTec-AD2 lighting shifts with per-category breakdown
6. **AU-PRO strict** — evaluation at FPR ≤ 0.05 threshold

## Citation

Original paper:
```bibtex
@InProceedings{luo2025INP-Former,
    author    = {Luo, Wei and Cao, Yunkang and Yao, Haiming and Zhang, Xiaotian and Lou, Jianan and Cheng, Yuqi and Shen, Weiming and Yu, Wenyong},
    title     = {Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {9974-9983}
}
```

## Acknowledgments

Based on the [official INP-Former implementation](https://github.com/luow23/INP-Former) by Wei Luo.
