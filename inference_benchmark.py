"""
Inference speed benchmark for INP-Former.

Measures FPS and latency across different configs:
  - INP count (M=2,4,6,8,12,16)
  - Input resolution (224/196, 336/294, 448/392)
  - Backbone (ViT-Small vs ViT-Base)
"""
import torch
import torch.nn as nn
import time
import argparse
import numpy as np
from functools import partial

from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from utils import setup_seed


def build_model(encoder_name, inp_num, device):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)
    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]

    Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
    INP = nn.ParameterList([nn.Parameter(torch.randn(inp_num, embed_dim)) for _ in range(1)])
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
    return model.to(device).eval()


def benchmark(model, input_size, crop_size, device, warmup=20, runs=100):
    dummy = torch.randn(1, 3, crop_size, crop_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times = np.array(times)
    return {
        'mean_ms': times.mean() * 1000,
        'std_ms': times.std() * 1000,
        'fps': 1.0 / times.mean(),
        'p50_ms': np.percentile(times, 50) * 1000,
        'p95_ms': np.percentile(times, 95) * 1000,
    }


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = total - encoder_params
    return total, trainable, encoder_params, decoder_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=20)
    args = parser.parse_args()

    setup_seed(1)
    device = 'cuda:0'

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {args.warmup}, Runs: {args.runs}")

    # --- INP count sweep ---
    print("\n" + "=" * 80)
    print("INP COUNT SWEEP (ViT-Base, 448→392)")
    print("=" * 80)
    print(f"  {'M':>4s} | {'FPS':>8s} | {'Mean ms':>9s} | {'Std ms':>8s} | {'P95 ms':>8s} | {'Decoder params':>14s}")
    print("-" * 80)

    for m in [2, 4, 6, 8, 12, 16]:
        model = build_model('dinov2reg_vit_base_14', m, device)
        _, _, _, dec_params = count_params(model)
        stats = benchmark(model, 448, 392, device, args.warmup, args.runs)
        print(f"  {m:4d} | {stats['fps']:8.1f} | {stats['mean_ms']:9.2f} | {stats['std_ms']:8.2f} | {stats['p95_ms']:8.2f} | {dec_params:>14,}")
        del model
        torch.cuda.empty_cache()

    # --- Resolution sweep ---
    print("\n" + "=" * 80)
    print("RESOLUTION SWEEP (ViT-Base, M=6)")
    print("=" * 80)
    print(f"  {'Resize→Crop':>12s} | {'Patches':>7s} | {'FPS':>8s} | {'Mean ms':>9s} | {'P95 ms':>8s}")
    print("-" * 80)

    for input_size, crop_size in [(224, 196), (336, 294), (448, 392)]:
        model = build_model('dinov2reg_vit_base_14', 6, device)
        n_patches = (crop_size // 14) ** 2
        stats = benchmark(model, input_size, crop_size, device, args.warmup, args.runs)
        print(f"  {input_size}→{crop_size:3d}    | {n_patches:7d} | {stats['fps']:8.1f} | {stats['mean_ms']:9.2f} | {stats['p95_ms']:8.2f}")
        del model
        torch.cuda.empty_cache()

    # --- Backbone comparison ---
    print("\n" + "=" * 80)
    print("BACKBONE COMPARISON (M=6, 448→392)")
    print("=" * 80)
    print(f"  {'Backbone':>25s} | {'FPS':>8s} | {'Mean ms':>9s} | {'Total params':>13s} | {'Decoder params':>14s}")
    print("-" * 80)

    for enc in ['dinov2reg_vit_small_14', 'dinov2reg_vit_base_14']:
        model = build_model(enc, 6, device)
        total, _, enc_p, dec_p = count_params(model)
        stats = benchmark(model, 448, 392, device, args.warmup, args.runs)
        print(f"  {enc:>25s} | {stats['fps']:8.1f} | {stats['mean_ms']:9.2f} | {total:>13,} | {dec_p:>14,}")
        del model
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == '__main__':
    main()
