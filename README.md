# Reproducibility Study: INP-Former (CVPR 2025)

Reproduction and extended analysis of [INP-Former: Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection](https://arxiv.org/abs/2503.02424) (Luo et al., CVPR 2025).

**Course:** MLDS2  
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
├── scripts/              # SLURM job scripts (also runnable with bash)
├── data/                 # Local datasets (ignored by Git)
├── containers/           # Local Enroot image (ignored by Git)
├── logs/                 # Setup and SLURM logs (ignored by Git)
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

Run all setup and experiment commands from the repository root. Datasets and the
container stay inside the checkout and are ignored by Git:

| Dataset | Source | Target path |
|---------|--------|-------------|
| MVTec-AD | [Voxel51 mirror](https://huggingface.co/datasets/Voxel51/mvtec-ad) | `data/mvtec_ad/` |
| VisA | [amazon-science/spot-diff](https://github.com/amazon-science/spot-diff) | `data/visa/1cls/` |
| Real-IAD | [realiad4ad.github.io](https://realiad4ad.github.io/Real-IAD/) | `data/Real-IAD/` |
| MVTec-AD2 | [mvtec.com](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) | `data/mvtec_ad2/` |

On a Slurm cluster, submit the setup pipeline from the repository root. If the
cluster has the Pyxis/Enroot plugin, a container is built first and MVTec AD and
VisA wait for it; otherwise the container step is skipped and every job uses the
active Python environment. MVTec AD 2 downloads independently, and the smoke test
waits for MVTec AD and VisA:

```bash
cd /path/to/Repro-INPFormer
bash scripts/submit_setup.sh
```

To resume after only some jobs were submitted, pass their active job IDs back to
the helper, for example:
`CONTAINER_JOB=12345 AD2_JOB=12346 bash scripts/submit_setup.sh`.

MVTec AD 2 download links can expire. Obtain the eight current category links
from the [official MVTec AD 2 page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2),
then provide them as `MVTECAD2_<CATEGORY>_URL` environment variables. For example:

```bash
export MVTECAD2_CAN_URL='<current can URL>'
export MVTECAD2_FABRIC_URL='<current fabric URL>'
# Also set FRUIT_JELLY, RICE, SHEET_METAL, VIAL, WALLPLUGS, and WALNUTS.
sbatch --export=ALL scripts/extract_mvtecad2.sh
```

All eight variables are needed for a new download. Categories already present
and complete under `data/mvtec_ad2/` are skipped.

Each setup job stops on the first error, writes a timestamped log under `logs/`,
and ends with `SETUP SUCCESS` or `SETUP FAILED: <reason>`. The VisA job checks
the official archive SHA-256 and applies Amazon's official `1cls.csv` split.

## Reproducing Results

### 1. Main Reproduction (Multi-Class)

```bash
# MVTec-AD
python INP_Former_Multi_Class.py --dataset MVTec-AD \
    --data_path data/mvtec_ad --phase train

# VisA
python INP_Former_Multi_Class.py --dataset VisA \
    --data_path data/visa/1cls --phase train
```

Expected results (I-AUROC / P-AUROC / P-AUPRO):
- MVTec-AD: 99.67 / 98.45 / 94.95
- VisA: 98.90 / 98.97 / 95.27

### 2. Loss/Module Ablation (5 seeds)

Ablation on MVTec-AD with seeds {1, 2, 3, 42, 123}:

```bash
# No INP baseline (y=0, lambda=0)
python INP_Former_Multi_Class.py --dataset MVTec-AD \
    --data_path data/mvtec_ad --phase train \
    --INP_num 0 --sm_gamma 0 --seed 1

# INP + Lsm only (y=3, lambda=0)
python INP_Former_Multi_Class.py --dataset MVTec-AD \
    --data_path data/mvtec_ad --phase train \
    --sm_gamma 3 --coherence_lambda 0 --seed 1

# Full model (y=3, lambda=0.2) — default
python INP_Former_Multi_Class.py --dataset MVTec-AD \
    --data_path data/mvtec_ad --phase train --seed 1
```

Or via SLURM: `sbatch scripts/ablation_loss_final.sh`

### 3. LoRA Transfer Learning

```bash
python lora_finetune.py --dataset MVTec-AD \
    --data_path data/mvtec_ad \
    --pretrained_weights saved_results/.../model.pth \
    --lora_rank 4
```

### 4. Analysis Scripts

```bash
# Anomaly area vs detection performance
python anomaly_area_analysis.py --dataset MVTec-AD \
    --data_path data/mvtec_ad \
    --weights saved_results/.../model.pth

# AU-PRO at strict FPR (0.05)
python eval_aupro_strict.py --dataset MVTec-AD \
    --data_path data/mvtec_ad \
    --weights saved_results/.../model.pth

# Lighting robustness (requires MVTec-AD2)
python lightshift_analysis.py \
    --data_path data/mvtec_ad2 \
    --load_from saved_results/.../model.pth

# Inference throughput
python inference_benchmark.py --weights saved_results/.../model.pth
```

## SLURM

All job scripts are in `scripts/` and must be submitted from the repository root
(or with `INPFORMER_ROOT` set). They request generic resources only; add your
cluster's partition, account, or GPU type on the command line, e.g.
`sbatch -p gpu --gres=gpu:a100:1 scripts/test_all_multiclass.sh`.

Each script runs its workload through `scripts/env.sh`: inside the Enroot image
`containers/inpformer_env.sqfs` when Pyxis is available and the image exists, and in
the current Python environment otherwise. Set `INPFORMER_CONTAINER=none` to force the
native environment, or `INPFORMER_CONTAINER_MOUNTS=/data:/data` to expose datasets
stored outside the checkout to the container. Most scripts also run without Slurm,
e.g. `bash scripts/test_all_multiclass.sh`.

```bash
sbatch scripts/setup_inpformer_env.sh        # Build container (Pyxis clusters only)
sbatch scripts/download_mvtec.sh             # Download MVTec AD
sbatch scripts/download_visa.sh              # Download and prepare VisA
sbatch scripts/extract_mvtecad2.sh           # Download MVTec AD 2
sbatch scripts/smoke_test.sh                  # One-GPU setup smoke test
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
