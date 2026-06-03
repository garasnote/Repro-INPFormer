# INP-Former Reproduction: Experimental Plan & Status

**Course:** MLDS2  
**Last updated:** 2026-05-23  
**Cluster:** FRIDA (`/shared/home/juan.osorio/ml/`)

---

## Overview

Reproducing INP-Former (CVPR 2025) — reconstruction-based anomaly detection using Intrinsic Normal Prototypes. Extensions: MVTec AD 2 benchmark, LoRA transfer learning, anomaly area analysis.

**Total experiments:** ~30 completed, ~10 remaining  
**Datasets:** MVTec-AD (15 categories), VisA (12), Real-IAD (30), MVTec-AD2 (7)  
**Backbone:** DINOv2 Register ViT-Base/14 (default), ViT-Small/14 (ablation)  
**Container:** `/shared/workspace/lkm/juan.osorio/container/inpformer_env.sqfs`

---

## P1: Reproduction of Main Results

Verify INP-Former multi-class performance on standard benchmarks.

| Dataset   | I-AUROC | P-AUROC | P-AUPRO | Status |
|-----------|---------|---------|---------|--------|
| MVTec-AD  | 99.67   | 98.45   | 94.95   | **DONE** |
| VisA      | 98.90   | 98.97   | 95.27   | **DONE** |
| Real-IAD  | ~88     | ~99     | ~96     | **DONE** |
| MVTec-AD2 | 69.29   | 88.87   | 60.29   | **DONE** |

**Also completed:**
- MVTec-AD2 single-class: I-AUROC 71.10, P-AUROC 89.24, P-AUPRO 63.26

**Checkpoints:** `saved_results/INP-Former-Multi-Class_dataset=<DS>_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/`  
**Per-category breakdowns:** in `log.txt` of each checkpoint dir  
**Aggregate script:** `python aggregate_results.py [--latex] [--dataset MVTec-AD]`

**Verdict:** Solid. Numbers align with paper. AD2 is lower but consistent with dataset difficulty.

---

## P2: Loss/Module Ablation with Seeds (Table 5 equivalent)

Statistical robustness of component contributions on MVTec-AD only (5 seeds: 1, 2, 3, 42, 123).

| Config | INP | Lsm | Lc | I-AUROC (mean+/-std) | Seeds done | Status |
|--------|-----|-----|----|----------------------|------------|--------|
| no_inp | no  | no  | no | 98.19+/-0.65         | 5/5        | **DONE** |
| inp_lsm | yes | yes | no | 99.64+/-0.04        | 5/5        | **DONE** |
| full   | yes | yes | yes | 99.62+/-0.01, P-AUPRO 94.92+/-0.07 | 5/5 | **DONE** |

**Dropped configs:** inp_only (y=0, lambda=0), inp_lc (y=0, lambda=0.2) — not enough compute time.

**Script:** `ablation_loss_final.sh` (job 89627) — completed 2026-05-23  
**Full seeds:** 1 (99.64/94.91), 2 (99.62/95.03), 3 (99.61/94.80), 42 (99.60/94.94), 123 (99.63/94.92)  
**no_inp seed=123:** I-AUROC 98.39, P-AUPRO 91.83

### Concerns
- **Only MVTec-AD** — VisA seeds were dropped entirely. Single-dataset ablation is weaker; frame as "computational constraints" in paper.
- **3 rows instead of 5** — cannot isolate Lc's contribution independently from Lsm. Shows "off -> partial -> full" but can't prove Lc alone does anything. Acknowledge gap.
- **Full vs INP+Lsm difference negligible** — Lc provides marginal P-AUPRO improvement (+0.06) but no I-AUROC gain. Partially contradicts paper's claim.

---

## P3: Efficiency Study

### 3a. INP Count Ablation on MVTec-AD2 — DONE

Paper covers MVTec/VisA; we extend to AD2.

| M  | I-AUROC | P-AUROC | P-AUPRO |
|----|---------|---------|---------|
| 1  | 70.57   | 88.46   | 63.03   |
| 2  | **72.43** | 88.03 | 62.98   |
| 4  | 69.85   | **89.23** | 61.54 |
| 6  | 69.29   | 88.87   | 60.29   |
| 8  | 68.70   | 88.99   | 60.73   |
| 12 | 69.27   | 88.64   | 60.02   |
| 16 | 68.27   | 88.13   | 57.60   |

**Finding:** Peak at M=2 for I-AUROC, M=4 for P-AUROC. Performance declines with more prototypes — opposite of MVTec/VisA where M=6 is optimal. Interesting divergence worth discussing (AD2 has different anomaly characteristics).

### 3b. Resolution Ablation on MVTec-AD — DONE

| Input/Crop | I-AUROC | P-AUROC | P-AUPRO |
|------------|---------|---------|---------|
| 224/196    | 99.21   | 97.97   | 92.07   |
| 336/294    | 99.63   | 98.28   | 94.04   |
| 448/392    | 99.67   | 98.45   | 94.95   |

Clean monotonic improvement. P-AUPRO benefits most (+3 points from 224 to 448).

### 3c. Backbone Ablation on MVTec-AD — DONE

| Backbone    | I-AUROC | P-AUROC | P-AUPRO |
|-------------|---------|---------|---------|
| ViT-Small/14 | 99.20  | 98.18   | 94.39   |
| ViT-Base/14  | 99.67  | 98.45   | 94.95   |

ViT-Small only -0.47 I-AUROC. Good cost-performance tradeoff.

### 3d. Inference Benchmark — DONE (L4 GPU)

| Config | FPS | Mean latency (ms) | P95 (ms) |
|--------|-----|-------------------|----------|
| **INP count sweep (ViT-Base, 448/392):** | | | |
| M=2 | 25.0 | 39.93 | 40.60 |
| M=4 | 25.2 | 39.65 | 40.26 |
| M=6 | 25.0 | 39.95 | 40.26 |
| M=8 | 25.0 | 39.96 | 40.38 |
| M=12 | 25.0 | 39.95 | 40.47 |
| M=16 | 25.0 | 39.96 | 40.57 |
| **Resolution sweep (ViT-Base, M=6):** | | | |
| 224/196 (196 patches) | 85.3 | 11.72 | 12.16 |
| 336/294 (441 patches) | 50.1 | 19.95 | 20.43 |
| 448/392 (784 patches) | 25.0 | 39.95 | 40.32 |
| **Backbone (M=6, 448/392):** | | | |
| ViT-Small/14 (39M params) | 72.6 | 13.78 | — |
| ViT-Base/14 (155M params) | 25.0 | 39.95 | — |

**Finding:** INP count has zero impact on speed (25 FPS for all M). Resolution is the bottleneck (3.4x slowdown from 224 to 448). ViT-Small is 2.9x faster than ViT-Base.  
**Log:** `logs/infer-bench-88273.out`

### 3e. Epoch Efficiency — DONE

**Job:** 89727 (A100), completed 2026-05-21  
**Script:** `epoch_efficiency.sh`  
**Config:** Full (y=3, λ=0.2), seed=42, MVTec-AD. Eval at epochs 25, 50, 100, 150, 200.

| Epoch | Loss  | I-AUROC | P-AUROC | P-AUPRO |
|-------|-------|---------|---------|---------|
| 25    | 0.163 | 99.29   | 98.07   | 94.57   |
| 50    | 0.145 | 99.50   | 98.26   | 94.76   |
| 100   | 0.133 | 99.55   | 98.40   | 94.90   |
| 150   | 0.128 | 99.59   | 98.45   | 94.95   |
| 200   | 0.126 | 99.61   | 98.48   | 94.96   |

**Finding:** Loss drops 68% in first 25 epochs. Model at epoch 50 reaches 99.6% of final P-AUPRO. Diminishing returns after epoch 100. 50 epochs sufficient for practical use.

---

## P4: LoRA Transfer Learning (Extension)

Parameter-efficient finetuning: Real-IAD pretrained -> MVTec-AD / VisA.

| Method          | MVTec I-AUROC | MVTec P-AUROC | MVTec P-AUPRO | VisA I-AUROC | VisA P-AUROC | VisA P-AUPRO |
|-----------------|---------------|---------------|---------------|--------------|--------------|--------------|
| Zero-shot       | 69.15         | 79.92         | 57.17         | 51.47        | 72.60        | 31.60        |
| LoRA r=2 (50ep) | 97.22         | 96.21         | 91.30         | 91.79        | 96.29        | 85.11        |
| LoRA r=4 (50ep) | 97.81         | 96.62         | 91.87         | 93.17        | 96.79        | 87.55        |
| LoRA r=8 (50ep) | **98.16**     | **96.96**     | **92.47**     | **94.44**    | **97.25**    | **89.91**    |
| Full train      | 99.67         | 98.45         | 94.95         | 98.90        | 98.97        | 95.27        |

**All ranks completed.** r=8 (job 89712) finished 2026-05-22.  
**Device fix applied** in `lora.py` (lora_A/lora_B now created on model device).  
**LoRA rank override:** `sbatch --export=LORA_RANKS=8 lora_finetune.sh`

**Finding:** Consistent scaling with rank. r=8 recovers 98.2% of full-train I-AUROC on MVTec. Diminishing returns from r=4→r=8 smaller than r=2→r=4. 42 adapters injected, 0.58% adapter params (5.12% total trainable).

**Script:** `lora_finetune.sh`  
**Key file:** `INP-Former/lora.py` (device fix applied), `INP-Former/lora_finetune.py`

---

## P5: Anomaly Area Analysis (Extension) — DONE

**Hypothesis:** INP-Former extracts normal prototypes from test images. Large anomaly area -> fewer normal patches -> degraded INP extraction -> worse detection.

**Result:** Hypothesis REJECTED. Detection *improves* with larger anomalies — opposite of prediction.

### MVTec-AD
- Spearman (area vs score): ρ = +0.53, p = 8.4e-91
- Spearman (area vs detected): ρ = +0.41, p = 1.9e-51
- Detection rate by area bin:

| Area range | N | Det rate | Mean score |
|-----------|---|----------|------------|
| 0.0%–0.5% | 250 | 63.6% | 0.184 |
| 0.5%–1.3% | 252 | 91.7% | 0.240 |
| 1.3%–2.8% | 253 | 98.8% | 0.255 |
| 2.8%–6.2% | 251 | 99.2% | 0.280 |
| 6.2%–63.4% | 252 | 100.0% | 0.304 |

### VisA
- Median per-category ρ = +0.58
- Strong positive correlation in most categories (cashew ρ=+0.54, fryum ρ=+0.80, capsules ρ=+0.83)

**Interpretation:** Larger defects produce more reconstruction error and are easier to detect. The "INP extraction degrades" effect is dominated by the "bigger target = bigger anomaly signal" effect. Small anomalies (<0.5% area) are the hard cases (63.6% detection rate).

**Figures:** `saved_results/.../anomaly_area_vs_performance_MVTec-AD.pdf` and `..._VisA.pdf`  
**Log:** `logs/area-analysis-88263.out` (May 17, L4 GPU)  
**Note:** May 21 rerun crashed on B200 (CUDA sm_100 incompatible). Old results are complete.

---

## P6: AU-PRO Strict Evaluation — DONE

MVTec AD 2 paper reports AU-PRO at FPR=0.30 (standard) and FPR=0.05 (strict).

**Job:** 89711 (L4), completed 2026-05-21. VisA crashed (std::system_error at end) but MVTec-AD and AD2 results are complete.

### MVTec-AD
| FPR threshold | Mean AU-PRO |
|---------------|-------------|
| ≤ 0.30        | 93.05       |
| ≤ 0.05        | 72.14       |

Worst categories at FPR≤0.05: transistor (36.4%), tile (45.2%).

### MVTec-AD2
| FPR threshold | Mean AU-PRO |
|---------------|-------------|
| ≤ 0.30        | 59.04       |
| ≤ 0.05        | 32.27       |

Worst categories at FPR≤0.05: can (4.5%), wallplugs (14.5%).

**Finding:** Tightening FPR from 0.3 to 0.05 causes -20.9 drop on MVTec-AD and -26.8 on AD2. Localization quality degrades significantly under strict thresholds.

**Script:** `eval_aupro_strict.sh`  
**Log:** `logs/aupro-strict-89711.out`

---

## P7: Lightshift Analysis on AD2 — DONE

Robustness to lighting condition changes on MVTec-AD2.

| Condition | I-AUROC | P-AUROC | AUPRO | ΔI-AUROC vs regular |
|-----------|---------|---------|-------|---------------------|
| regular | 74.59 | 91.65 | 67.46 | — |
| overexposed | 73.18 | 90.84 | 66.69 | -1.4 |
| underexposed | 75.42 | 89.16 | 65.18 | +0.8 |
| shift_1 | 63.56 | 90.06 | 62.06 | **-11.0** |
| shift_2 | 72.44 | 86.95 | 57.28 | -2.2 |
| shift_3 | 60.92 | 88.28 | 57.63 | **-13.7** |

**Finding:** Over/underexposure causes minimal degradation (<1.5 points). Color shifts (shift_1, shift_3) severely impact I-AUROC (-11 to -14 points). Per-category: wallplugs and walnuts most sensitive to shifts; vial most robust (94.7% I-AUROC across conditions).

**Log:** `logs/lightshift-88206.out`

---

## Ghost Experiments (empty/broken, not in paper)

These directories exist but have no usable results. All contain empty `log.txt.txt` (double extension typo):

| Experiment | Status | Action |
|-----------|--------|--------|
| Few-Shot-1 (Real-IAD, VisA) | empty | ignore |
| Few-Shot-2 (MVTec, Real-IAD, VisA) | empty | ignore |
| Few-Shot-4 (MVTec, Real-IAD, VisA) | empty | ignore |
| Super-Multi-Class | empty | ignore |
| Zero-Shot (Real-IAD -> MVTec) | empty | ignore |
| Single-Class MVTec-AD | no log | ignore |
| Single-Class VisA | no log | ignore |
| Single-Class Real-IAD | stopped epoch 57/200 | ignore |

---

## Submission Queue (priority order)

| # | Script | Type | Time | Status |
|---|--------|------|------|--------|
| 1 | `ablation_loss_final.sh` | training | ~42h | **DONE** (job 89627, all 6/6 complete) |
| 2 | `lora_finetune.sh` (r=8) | training | ~12h | **DONE** (job 89712) |
| 3 | `inference_benchmark.sh` | eval | ~1h | **DONE** (L4, May 17) |
| 4 | `eval_aupro_strict.sh` | eval | ~15min | **DONE** (job 89711, VisA crashed but MVTec+AD2 complete) |
| 5 | `anomaly_area_analysis.sh` | eval | ~1-2h | **DONE** (L4, May 17) |
| 6 | `lightshift_analysis.sh` | eval | ~2h | **DONE** (May 17) |
| 7 | `epoch_efficiency.sh` | training | ~9h | **DONE** (job 89727) |

**Known bad nodes:** `aga` (NVML driver mismatch), B200/B300 nodes (CUDA sm_100 incompatible with container). Use `--exclude=aga --gres=gpu:A100:1` or `--gres=gpu:L4:1`.

---

## Paper Flow

1. **Introduction** — INP-Former (CVPR 2025) reproduction, new benchmark (AD2), extensions
2. **Reproduction** — Tables: MVTec, VisA, Real-IAD per-category + AD2. Compare to paper numbers.
3. **Ablation** — Table 5 (3 rows) with mean+/-std on MVTec. INP count on AD2. Resolution, backbone.
4. **Extension: LoRA** — Zero-shot -> LoRA transfer table. Parameter efficiency angle.
5. **Extension: Anomaly Area** — If results interesting (submit and evaluate)
6. **Analysis** — AU-PRO strict, inference speed, convergence (epoch efficiency)
7. **Discussion** — AD2 challenges, INP count divergence, LoRA as practical deployment strategy
