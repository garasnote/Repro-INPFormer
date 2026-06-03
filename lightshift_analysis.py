"""
Light Shift Robustness Analysis for MVTec AD 2.

Trains on regular lighting, evaluates separately on each lighting condition:
  regular, overexposed, underexposed, shift_1, shift_2, shift_3

Reports per-condition I-AUROC, P-AUROC, P-AUPRO to show degradation.
"""
import torch
import torch.nn as nn
import numpy as np
import os
import glob
import argparse
from functools import partial
from tqdm import tqdm
from torch.nn.init import trunc_normal_
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

from optimizers import StableAdamW
from utils import evaluation_batch, WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger
from dataset import get_data_transforms
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

LIGHT_CONDITIONS = ['regular', 'overexposed', 'underexposed', 'shift_1', 'shift_2', 'shift_3']


class MVTecAD2LightDataset(torch.utils.data.Dataset):
    """MVTec AD 2 test dataset filtered by lighting condition."""

    def __init__(self, root, transform, gt_transform, condition):
        self.img_path = os.path.join(root, 'test_public')
        self.gt_path = os.path.join(root, 'test_public', 'ground_truth', 'bad')
        self.transform = transform
        self.gt_transform = gt_transform
        self.condition = condition
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()
        self.cls_idx = 0

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        suffix = f"_{self.condition}.png"

        good_paths = sorted([p for p in glob.glob(os.path.join(self.img_path, 'good', '*.png'))
                            if p.endswith(suffix)])
        img_tot_paths.extend(good_paths)
        gt_tot_paths.extend([0] * len(good_paths))
        tot_labels.extend([0] * len(good_paths))
        tot_types.extend(['good'] * len(good_paths))

        bad_paths = sorted([p for p in glob.glob(os.path.join(self.img_path, 'bad', '*.png'))
                           if p.endswith(suffix)])
        mask_suffix = f"_{self.condition}_mask.png"
        gt_paths = sorted([p for p in glob.glob(self.gt_path + '/*.png')
                          if p.endswith(mask_suffix)])

        img_tot_paths.extend(bad_paths)
        gt_tot_paths.extend(gt_paths)
        tot_labels.extend([1] * len(bad_paths))
        tot_types.extend(['bad'] * len(bad_paths))

        assert len(bad_paths) == len(gt_paths), \
            f"Mismatch for {self.condition}: {len(bad_paths)} images vs {len(gt_paths)} masks"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        return img, gt, label, img_path


def main():
    parser = argparse.ArgumentParser(description='Light Shift Robustness Analysis')
    parser.add_argument('--data_path', type=str, default='../data/mvtec_ad_2')
    parser.add_argument('--load_from', type=str,
                        default='saved_results/INP-Former-Multi-Class_dataset=MVTec-AD2_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6_y=3_lambda=0.2/model.pth',
                        help='Path to pretrained model.pth')
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--total_epochs', type=int, default=200)
    args = parser.parse_args()

    setup_seed(1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    item_list = ['can', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']

    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    # Build model
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
    model = model.to(device)

    save_dir = f'saved_results/lightshift_INP_num={args.INP_num}'
    os.makedirs(save_dir, exist_ok=True)

    if args.load_from:
        print(f"Loading weights from {args.load_from}")
        model.load_state_dict(torch.load(args.load_from), strict=True)
    else:
        # Train on regular lighting (all categories multi-class)
        print("Training on regular lighting (multi-class)...")
        train_data_list = []
        for i, item in enumerate(item_list):
            train_path = os.path.join(args.data_path, item, 'train')
            train_data = ImageFolder(root=train_path, transform=data_transform)
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            train_data.samples = [(s[0], i) for s in train_data.samples]
            train_data_list.append(train_data)

        train_data = ConcatDataset(train_data_list)
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                      num_workers=4, drop_last=True)

        trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        optimizer = StableAdamW([{'params': trainable.parameters()}],
                                lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4,
                                           total_iters=args.total_epochs * len(train_dataloader),
                                           warmup_iters=100)

        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            for img, _ in tqdm(train_dataloader, ncols=80, desc=f'Epoch {epoch+1}/{args.total_epochs}'):
                img = img.to(device)
                en, de, g_loss = model(img)
                loss = global_cosine_hm_adaptive(en, de, y=3)
                loss = loss + 0.2 * g_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                lr_scheduler.step()
            print(f'Epoch [{epoch+1}/{args.total_epochs}], loss: {np.mean(loss_list):.4f}')

        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        print(f"Model saved to {save_dir}/model.pth")

    # Evaluate per lighting condition
    model.eval()
    print("\n" + "=" * 70)
    print("LIGHT SHIFT ROBUSTNESS ANALYSIS")
    print("=" * 70)

    # category_results[item][condition] = (i_auroc, p_auroc, aupro)
    category_results = {item: {} for item in item_list}
    condition_means = []

    for condition in LIGHT_CONDITIONS:
        auroc_sp_list, auroc_px_list, aupro_px_list = [], [], []

        for item in item_list:
            item_path = os.path.join(args.data_path, item)
            test_data = MVTecAD2LightDataset(root=item_path, transform=data_transform,
                                             gt_transform=gt_transform, condition=condition)
            if len(test_data) == 0:
                continue
            test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
            results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
            category_results[item][condition] = (auroc_sp, auroc_px, aupro_px)
            auroc_sp_list.append(auroc_sp)
            auroc_px_list.append(auroc_px)
            aupro_px_list.append(aupro_px)

        mean_i_auroc = np.mean(auroc_sp_list) if auroc_sp_list else 0
        mean_p_auroc = np.mean(auroc_px_list) if auroc_px_list else 0
        mean_aupro = np.mean(aupro_px_list) if aupro_px_list else 0
        condition_means.append([condition, mean_i_auroc, mean_p_auroc, mean_aupro])

    # --- Per-condition summary ---
    print("\n" + "=" * 70)
    print("PER-CONDITION SUMMARY (mean across categories)")
    print("=" * 70)
    for row in condition_means:
        print(f"  {row[0]:15s} | I-AUROC: {row[1]:.4f} | P-AUROC: {row[2]:.4f} | AUPRO: {row[3]:.4f}")
    print("-" * 70)
    print(f"  {'MEAN':15s} | I-AUROC: {np.mean([r[1] for r in condition_means]):.4f} | "
          f"P-AUROC: {np.mean([r[2] for r in condition_means]):.4f} | "
          f"AUPRO: {np.mean([r[3] for r in condition_means]):.4f}")

    # --- Degradation vs regular ---
    regular_row = [r for r in condition_means if r[0] == 'regular']
    if regular_row:
        reg = regular_row[0]
        print("\nDegradation vs regular (AUPRO is primary metric for localization):")
        for row in condition_means:
            if row[0] == 'regular':
                continue
            print(f"  {row[0]:15s} | ΔAUPRO: {row[3] - reg[3]:+.4f} | ΔP-AUROC: {row[2] - reg[2]:+.4f} | ΔI-AUROC: {row[1] - reg[1]:+.4f}")

    # --- Per-category breakdown ---
    print("\n" + "=" * 70)
    print("PER-CATEGORY BREAKDOWN (I-AUROC)")
    print("=" * 70)
    header = f"  {'Category':15s}"
    for c in LIGHT_CONDITIONS:
        header += f" | {c:>12s}"
    print(header)
    print("-" * 70)
    for item in item_list:
        row = f"  {item:15s}"
        for c in LIGHT_CONDITIONS:
            if c in category_results[item]:
                row += f" | {category_results[item][c][0]:12.4f}"
            else:
                row += f" | {'N/A':>12s}"
        print(row)

    print("\n" + "=" * 70)
    print("PER-CATEGORY BREAKDOWN (AUPRO — primary localization metric)")
    print("=" * 70)
    header = f"  {'Category':15s}"
    for c in LIGHT_CONDITIONS:
        header += f" | {c:>12s}"
    print(header)
    print("-" * 70)
    for item in item_list:
        row = f"  {item:15s}"
        for c in LIGHT_CONDITIONS:
            if c in category_results[item]:
                row += f" | {category_results[item][c][2]:12.4f}"
            else:
                row += f" | {'N/A':>12s}"
        print(row)

    # --- Per-category degradation vs regular (AUPRO) ---
    print("\n" + "=" * 70)
    print("PER-CATEGORY DEGRADATION vs REGULAR (ΔAUPRO)")
    print("=" * 70)
    header = f"  {'Category':15s}"
    for c in LIGHT_CONDITIONS:
        if c == 'regular':
            continue
        header += f" | {c:>12s}"
    print(header)
    print("-" * 70)
    for item in item_list:
        row = f"  {item:15s}"
        reg_val = category_results[item].get('regular', (0, 0, 0))[2]
        for c in LIGHT_CONDITIONS:
            if c == 'regular':
                continue
            if c in category_results[item]:
                delta = category_results[item][c][2] - reg_val
                row += f" | {delta:+12.4f}"
            else:
                row += f" | {'N/A':>12s}"
        print(row)

    print("\nDone.")


if __name__ == '__main__':
    main()
