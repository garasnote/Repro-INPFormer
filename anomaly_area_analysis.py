"""
Anomaly area vs detection performance analysis.

Hypothesis: INP-Former extracts normal prototypes from test images.
When anomaly area is small, most patches are normal → good INP extraction → good reconstruction.
When anomaly area is large, fewer normal patches → degraded INPs → worse detection.
Prediction: per-image anomaly score accuracy should decrease monotonically with anomaly area.

Outputs:
  - Scatter plot: anomaly area ratio vs anomaly score
  - Binned performance: area bins → mean detection accuracy
  - Spearman correlation test
  - Per-category breakdown
  - Saved figure as anomaly_area_vs_performance.pdf

Usage:
  python anomaly_area_analysis.py --dataset MVTec-AD --weights saved_results/.../model.pth
  python anomaly_area_analysis.py --dataset VisA --weights saved_results/.../model.pth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from functools import partial
from tqdm import tqdm
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score
from PIL import Image

from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from utils import setup_seed, cal_anomaly_maps, get_gaussian_kernel, compute_pro
from dataset import MVTecDataset, get_data_transforms


MVTEC_ITEMS = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
               'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
VISA_ITEMS = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
              'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']


def build_model(args, device):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]

    Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
    INP = nn.ParameterList([nn.Parameter(torch.randn(args.INP_num, embed_dim)) for _ in range(1)])
    INP_Extractor = nn.ModuleList([
        Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        for _ in range(1)
    ])
    INP_Guided_Decoder = nn.ModuleList([
        Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        for _ in range(8)
    ])

    model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor,
                       decoder=INP_Guided_Decoder, target_layers=target_layers,
                       remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                       fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
    return model.to(device)


def compute_per_image_data(model, dataloader, device, max_ratio=0.01):
    """Run inference, return per-image: anomaly_score, anomaly_map, gt_mask, label."""
    model.eval()
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    results = []

    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80, desc='Inference'):
            img = img.to(device)
            en, de = model(img)[:2]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = F.interpolate(anomaly_map, size=256, mode='bilinear', align_corners=False)
            gt = F.interpolate(gt, size=256, mode='nearest')
            anomaly_map = gaussian_kernel(anomaly_map)

            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            anomaly_map_flat = anomaly_map.flatten(1)
            sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][:, :int(anomaly_map_flat.shape[1] * max_ratio)]
            sp_score = sp_score.mean(dim=1)

            for i in range(img.shape[0]):
                results.append({
                    'score': sp_score[i].cpu().item(),
                    'anomaly_map': anomaly_map[i, 0].cpu().numpy(),
                    'gt_mask': gt[i, 0].cpu().numpy(),
                    'label': label[i].item(),
                    'path': img_path[i],
                })

    return results


def analyze_area_vs_performance(all_data, save_dir, dataset_name):
    """Core analysis: anomaly area ratio vs detection score."""
    anomalous = [d for d in all_data if d['label'] == 1]
    normal = [d for d in all_data if d['label'] == 0]

    if not anomalous:
        print("No anomalous images found!")
        return

    areas = np.array([d['gt_mask'].mean() for d in anomalous])
    scores = np.array([d['score'] for d in anomalous])
    normal_scores = np.array([d['score'] for d in normal])
    categories = np.array([d['category'] for d in anomalous])

    # Overall I-AUROC for reference
    all_scores = np.concatenate([normal_scores, scores])
    all_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(scores))])
    overall_auroc = roc_auc_score(all_labels, all_scores) * 100

    # Spearman correlation: area vs score
    rho_score, p_score = scipy_stats.spearmanr(areas, scores)

    # Threshold for scatter plot only
    threshold = normal_scores.mean() + 2 * normal_scores.std()

    print(f"\n{'='*70}")
    print(f"ANOMALY AREA ANALYSIS — {dataset_name}")
    print(f"{'='*70}")
    print(f"Anomalous images: {len(anomalous)}, Normal images: {len(normal)}")
    print(f"Overall I-AUROC: {overall_auroc:.2f}%")
    print(f"\nSpearman (area vs score): ρ = {rho_score:+.4f}, p = {p_score:.2e}")

    # Binned analysis with per-bin I-AUROC and P-AUPRO
    n_bins = 5
    bin_edges = np.percentile(areas, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-6

    anomalous_amaps = np.array([d['anomaly_map'] for d in anomalous])
    anomalous_masks = np.array([d['gt_mask'] for d in anomalous])
    normal_amaps = np.array([d['anomaly_map'] for d in normal])
    normal_masks = np.zeros_like(normal_amaps)

    print(f"\n{'Area range':>20s} | {'N':>4s} | {'I-AUROC':>8s} | {'P-AUPRO':>8s}")
    print("-" * 60)

    bin_centers = []
    bin_aurocs = []
    bin_aupros = []

    for i in range(n_bins):
        mask = (areas >= bin_edges[i]) & (areas < bin_edges[i+1])
        if mask.sum() == 0:
            continue
        n = mask.sum()
        mean_area = areas[mask].mean() * 100
        area_lo = areas[mask].min() * 100
        area_hi = areas[mask].max() * 100

        bin_scores = scores[mask]
        all_scores_bin = np.concatenate([normal_scores, bin_scores])
        all_labels_bin = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(bin_scores))])
        auroc_bin = roc_auc_score(all_labels_bin, all_scores_bin) * 100

        bin_amaps = anomalous_amaps[mask]
        bin_masks = anomalous_masks[mask]
        aupro_bin = compute_pro(bin_masks, bin_amaps, num_th=200, fpr_limit=0.3) * 100

        bin_centers.append(mean_area)
        bin_aurocs.append(auroc_bin)
        bin_aupros.append(aupro_bin)

        print(f"  {area_lo:5.1f}% — {area_hi:5.1f}% | {n:4d} | {auroc_bin:7.1f}% | {aupro_bin:7.1f}%")

    # Per-category breakdown
    unique_cats = sorted(set(categories))
    print(f"\nPer-category Spearman (area vs score):")
    print(f"  {'Category':<16s} | {'N':>4s} | {'ρ':>7s} | {'p':>10s} | {'Area range':>15s}")
    print("-" * 70)
    cat_rhos = []
    for cat in unique_cats:
        cat_mask = categories == cat
        if cat_mask.sum() < 5:
            continue
        cat_areas = areas[cat_mask]
        cat_scores = scores[cat_mask]
        if cat_areas.std() < 1e-8:
            continue
        rho_cat, p_cat = scipy_stats.spearmanr(cat_areas, cat_scores)
        cat_rhos.append(rho_cat)
        print(f"  {cat:<16s} | {cat_mask.sum():4d} | {rho_cat:+7.4f} | {p_cat:10.2e} | {cat_areas.min()*100:5.1f}%—{cat_areas.max()*100:5.1f}%")

    if cat_rhos:
        print(f"  {'MEDIAN ρ':<16s} | {'':4s} | {np.median(cat_rhos):+7.4f}")

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1) Scatter: area vs score, colored by category
    ax = axes[0]
    cmap = plt.cm.tab20
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    colors = [cmap(cat_to_idx[c] % 20) for c in categories]
    ax.scatter(areas * 100, scores, c=colors, alpha=0.5, s=15, edgecolors='none')
    ax.axhline(threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold:.3f})')
    ax.set_xlabel('Anomaly Area (%)')
    ax.set_ylabel('Anomaly Score')
    ax.set_title(f'Per-Image Score vs Area\n(ρ={rho_score:+.3f}, p={p_score:.1e})')
    ax.legend(fontsize=8)

    # 2) Binned I-AUROC and P-AUPRO
    ax = axes[1]
    x = np.arange(len(bin_centers))
    w = 0.35
    ax.bar(x - w/2, bin_aurocs, w, color='steelblue', alpha=0.8, label='I-AUROC')
    ax.bar(x + w/2, bin_aupros, w, color='coral', alpha=0.8, label='P-AUPRO')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c:.1f}%' for c in bin_centers], fontsize=8)
    ax.set_xlabel('Mean Anomaly Area')
    ax.set_ylabel('Metric (%)')
    ax.set_title('I-AUROC & P-AUPRO vs Area')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    # 3) Per-category ρ distribution
    ax = axes[2]
    if cat_rhos:
        ax.barh(range(len(unique_cats[:15])), [cat_rhos[i] if i < len(cat_rhos) else 0 for i in range(min(15, len(unique_cats)))],
                color=['salmon' if r < 0 else 'steelblue' for r in cat_rhos[:15]], alpha=0.8)
        ax.set_yticks(range(min(15, len(unique_cats))))
        ax.set_yticklabels(unique_cats[:15], fontsize=8)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Spearman ρ (area vs score)')
        ax.set_title('Per-Category Correlation')

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f'anomaly_area_vs_performance_{dataset_name}.pdf')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {fig_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['MVTec-AD', 'VisA'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    item_list = MVTEC_ITEMS if args.dataset == 'MVTec-AD' else VISA_ITEMS

    model = build_model(args, device)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=True)
    model.eval()
    print(f"Loaded weights: {args.weights}")

    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    all_data = []
    for item in item_list:
        test_path = os.path.join(args.data_path, item)
        test_data = MVTecDataset(root=test_path, transform=data_transform,
                                 gt_transform=gt_transform, phase='test')
        test_dl = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                               shuffle=False, num_workers=4)
        results = compute_per_image_data(model, test_dl, device)
        for r in results:
            r['category'] = item
        all_data.extend(results)
        print(f"  {item}: {len(results)} images")

    save_dir = os.path.dirname(args.weights)
    analyze_area_vs_performance(all_data, save_dir, args.dataset)

    print("\nDone.")


if __name__ == '__main__':
    main()
