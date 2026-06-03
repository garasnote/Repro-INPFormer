"""
LoRA finetuning for INP-Former cross-dataset transfer.

Loads pretrained weights (e.g. from Real-IAD), injects LoRA adapters
into decoder/bottleneck, finetunes on target dataset, evaluates.
"""
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from functools import partial
from tqdm import tqdm
from torch.nn.init import trunc_normal_

from optimizers import StableAdamW
from utils import evaluation_batch, WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger
from dataset import MVTecDataset, RealIADDataset, get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from lora import apply_lora, get_lora_state_dict, merge_lora


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


def load_data(args, data_transform, gt_transform):
    if args.dataset in ('MVTec-AD', 'VisA'):
        item_list = {
            'MVTec-AD': ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                         'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper'],
            'VisA': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                     'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'],
        }[args.dataset]

        train_data_list, test_data_list = [], []
        for i, item in enumerate(item_list):
            train_path = os.path.join(args.data_path, item, 'train')
            test_path = os.path.join(args.data_path, item)
            train_data = ImageFolder(root=train_path, transform=data_transform)
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            train_data.samples = [(s[0], i) for s in train_data.samples]
            test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
            train_data_list.append(train_data)
            test_data_list.append(test_data)

        train_data = ConcatDataset(train_data_list)
        train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        return train_dl, test_data_list, item_list

    elif args.dataset == 'Real-IAD':
        item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                     'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                     'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                     'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                     'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']

        train_data_list, test_data_list = [], []
        for i, item in enumerate(item_list):
            train_data = RealIADDataset(root=args.data_path, category=item, transform=data_transform,
                                        gt_transform=gt_transform, phase='train')
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            test_data = RealIADDataset(root=args.data_path, category=item, transform=data_transform,
                                       gt_transform=gt_transform, phase='test')
            train_data_list.append(train_data)
            test_data_list.append(test_data)

        train_data = ConcatDataset(train_data_list)
        train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        return train_dl, test_data_list, item_list


def evaluate(model, test_data_list, item_list, device, batch_size, print_fn):
    model.eval()
    auroc_sp_list, auroc_px_list, aupro_px_list = [], [], []
    for item, test_data in zip(item_list, test_data_list):
        test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
        results = evaluation_batch(model, test_dl, device, max_ratio=0.01, resize_mask=256)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
        auroc_sp_list.append(auroc_sp)
        auroc_px_list.append(auroc_px)
        aupro_px_list.append(aupro_px)
        print_fn(f'{item}: I-AUROC:{auroc_sp:.4f}, P-AUROC:{auroc_px:.4f}, AUPRO:{aupro_px:.4f}')

    print_fn(f'Mean: I-AUROC:{np.mean(auroc_sp_list):.4f}, P-AUROC:{np.mean(auroc_px_list):.4f}, AUPRO:{np.mean(aupro_px_list):.4f}')
    return np.mean(auroc_sp_list), np.mean(auroc_px_list), np.mean(aupro_px_list)


def main():
    parser = argparse.ArgumentParser(description='LoRA finetuning for INP-Former')
    parser.add_argument('--source_weights', type=str, required=True, help='Path to pretrained model.pth')
    parser.add_argument('--dataset', type=str, required=True, help='Target dataset: MVTec-AD, VisA, Real-IAD')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--total_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--lora_alpha', type=float, default=1.0)
    parser.add_argument('--lora_targets', type=str, default='decoder,bottleneck',
                        help='Comma-separated module prefixes to apply LoRA')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    save_name = f'LoRA_r{args.lora_rank}_a{args.lora_alpha}_{args.dataset}_from_{os.path.basename(os.path.dirname(args.source_weights))}_epochs={args.total_epochs}_seed={args.seed}'
    save_dir = os.path.join('saved_results', save_name)
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(save_name, save_dir)
    print_fn = logger.info

    # Build model and load pretrained weights
    model = build_model(args, device)
    print_fn(f"Loading source weights from {args.source_weights}")
    model.load_state_dict(torch.load(args.source_weights, map_location=device), strict=True)

    # Evaluate before finetuning
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)
    train_dl, test_data_list, item_list = load_data(args, data_transform, gt_transform)

    print_fn("=== Before LoRA finetuning (zero-shot transfer) ===")
    evaluate(model, test_data_list, item_list, device, args.batch_size, print_fn)

    # Freeze everything, inject LoRA
    for param in model.parameters():
        param.requires_grad = False

    target_modules = [t.strip() for t in args.lora_targets.split(',')]
    lora_params = apply_lora(model, target_modules=target_modules,
                             rank=args.lora_rank, alpha=args.lora_alpha)
    print_fn(f"LoRA targets: {target_modules}")

    # Also unfreeze INP tokens and aggregation (small, important for adaptation)
    for param in model.prototype_token.parameters() if hasattr(model.prototype_token, 'parameters') else [model.prototype_token]:
        param.requires_grad = True
    for param in model.aggregation.parameters():
        param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_all = sum(p.numel() for p in model.parameters())
    print_fn(f"Trainable: {total_trainable:,} / {total_all:,} ({100*total_trainable/total_all:.2f}%)")

    # Train
    optimizer = StableAdamW([{'params': trainable_params}],
                            lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=args.lr, final_value=args.lr * 0.1,
                                       total_iters=args.total_epochs * len(train_dl), warmup_iters=50)

    print_fn(f"\n=== LoRA finetuning ({args.total_epochs} epochs) ===")
    for epoch in range(args.total_epochs):
        model.train()
        loss_list = []
        for img, _ in tqdm(train_dl, ncols=80, desc=f'Epoch {epoch+1}/{args.total_epochs}'):
            img = img.to(device)
            en, de, g_loss = model(img)
            loss = global_cosine_hm_adaptive(en, de, y=3)
            loss = loss + 0.2 * g_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()
        print_fn(f'Epoch [{epoch+1}/{args.total_epochs}], loss: {np.mean(loss_list):.4f}')

    # Evaluate after finetuning
    print_fn("\n=== After LoRA finetuning ===")
    evaluate(model, test_data_list, item_list, device, args.batch_size, print_fn)

    # Save LoRA weights
    lora_state = get_lora_state_dict(model)
    torch.save(lora_state, os.path.join(save_dir, 'lora_weights.pth'))

    # Merge and save full model
    merge_lora(model)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_merged.pth'))
    print_fn(f"Saved to {save_dir}")


if __name__ == '__main__':
    main()
