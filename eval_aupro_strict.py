"""
Evaluate AU-PRO at both FPR limits (0.30 and 0.05) for any dataset.

MVTec AD 2 paper reports both AU-PRO_0.30 and AU-PRO_0.05.
Standard INP-Former pipeline only computes AU-PRO_0.30 via adeval.
This script uses the CPU compute_pro with configurable fpr_limit.

Usage:
  python eval_aupro_strict.py --dataset MVTec-AD2 --data_path ../data/mvtec_anomaly_detection_2 \
      --weights saved_results/.../model.pth
  python eval_aupro_strict.py --dataset MVTec-AD --data_path ../data/mvtec_anomaly_detection \
      --weights saved_results/.../model.pth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from functools import partial
from tqdm import tqdm

from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from utils import setup_seed, cal_anomaly_maps, get_gaussian_kernel, compute_pro
from dataset import MVTecDataset, MVTecAD2Dataset, get_data_transforms


DATASET_ITEMS = {
    'MVTec-AD': ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper'],
    'VisA': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
             'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'],
    'MVTec-AD2': ['can', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts'],
}


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


def get_anomaly_maps(model, dataloader, device, resize_mask=256):
    """Run inference, collect anomaly maps and GT masks for anomalous images only."""
    model.eval()
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    all_amaps = []
    all_masks = []

    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            img = img.to(device)
            en, de = model(img)[:2]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
            gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            anomaly_map = gaussian_kernel(anomaly_map)

            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            for i in range(img.shape[0]):
                mask_np = gt[i, 0].cpu().numpy()
                if mask_np.max() > 0:
                    all_amaps.append(anomaly_map[i, 0].cpu().numpy())
                    all_masks.append(mask_np.astype(np.uint8))

    return np.array(all_amaps), np.array(all_masks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_th', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    item_list = DATASET_ITEMS[args.dataset]

    model = build_model(args, device)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=True)
    model.eval()
    print(f"Loaded: {args.weights}")

    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    fpr_limits = [0.30, 0.05]

    print(f"\n{'='*80}")
    print(f"AU-PRO EVALUATION — {args.dataset}")
    print(f"{'='*80}")

    header = f"  {'Category':<16s} | {'N_anom':>6s}"
    for lim in fpr_limits:
        header += f" | {'AUPRO_'+str(lim):>10s}"
    print(header)
    print("-" * 80)

    all_aupro = {lim: [] for lim in fpr_limits}

    for item in item_list:
        if args.dataset == 'MVTec-AD2':
            test_data = MVTecAD2Dataset(root=os.path.join(args.data_path, item),
                                         transform=data_transform, gt_transform=gt_transform, phase='test')
        else:
            test_data = MVTecDataset(root=os.path.join(args.data_path, item),
                                     transform=data_transform, gt_transform=gt_transform, phase='test')

        test_dl = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                               shuffle=False, num_workers=4)

        amaps, masks = get_anomaly_maps(model, test_dl, device)

        if len(amaps) == 0:
            print(f"  {item:<16s} | {0:6d} | {'N/A':>10s} | {'N/A':>10s}")
            continue

        row = f"  {item:<16s} | {len(amaps):6d}"
        for lim in fpr_limits:
            pro = compute_pro(masks, amaps, num_th=args.num_th, fpr_limit=lim)
            all_aupro[lim].append(pro)
            row += f" | {pro*100:10.2f}"
        print(row)

    print("-" * 80)
    row = f"  {'MEAN':<16s} | {'':6s}"
    for lim in fpr_limits:
        if all_aupro[lim]:
            row += f" | {np.mean(all_aupro[lim])*100:10.2f}"
        else:
            row += f" | {'N/A':>10s}"
    print(row)

    print("\nDone.")


if __name__ == '__main__':
    main()
