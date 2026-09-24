"""Inference only: continuous maps plus a complete, hashed evaluation manifest.

full-frame: resize the whole image to 392x392, then restore maps to native H/W.
legacy-crop256: historical 448 resize / 392 crop / 256 map; diagnostic only.
No thresholding, per-image normalization, retraining, or custom AU-PRO calls.
"""
import argparse
import json
from pathlib import Path
from functools import partial

from mvtec_official_adapter import CATEGORIES, ROOT, samples, sha256, write_json


def build_model(m, state, device, use_inp=True):
    import torch
    from torch import nn
    from dinov2.models import vision_transformer as dino
    from models.uad import INP_Former
    from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
    # Same encoder constructor as vit_encoder.load; full checkpoint supplies ALL
    # weights, including encoder. No download or randomly initialized fallback.
    encoder = dino.vit_base(patch_size=14, img_size=518, block_chunks=0,
                           init_values=1e-8, num_register_tokens=4,
                           interpolate_antialias=False, interpolate_offset=0.1)
    block_args = dict(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True,
                      norm_layer=partial(nn.LayerNorm, eps=1e-8))
    model = INP_Former(
        encoder=encoder,
        bottleneck=nn.ModuleList([Mlp(768, 3072, 768, drop=0.)]),
        aggregation=nn.ModuleList([Aggregation_Block(**block_args)]),
        decoder=nn.ModuleList([Prototype_Block(**block_args) for _ in range(8)]),
        target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
        remove_class_token=True,
        fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        prototype_token=nn.ParameterList([nn.Parameter(torch.empty(m, 768))]),
        use_inp=use_inp)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=list(CATEGORIES), required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--m', type=int, required=True)
    p.add_argument('--no-inp', action='store_true',
                   help='Legacy No-INP: cross-attention to pre-bottleneck encoder patches, not self-attention')
    p.add_argument('--geometry', choices=['full-frame', 'legacy-crop256'], required=True)
    p.add_argument('--categories', nargs='+')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--require-epoch200', action='store_true')
    p.add_argument('--smoke-images', type=int, default=0,
                   help='Nonzero creates an incomplete test manifest; cannot produce paper metrics')
    args = p.parse_args()
    if 'private' in args.split and args.geometry != 'full-frame':
        p.error('Private export requires a full-image geometry; crop256 is not a server prediction')
    if args.output.exists():
        p.error('Output already exists; use a new directory (never overwrite predictions)')
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size == 0:
        p.error(f'Checkpoint is not ready: {args.checkpoint}')
    cats = args.categories or CATEGORIES[args.dataset]
    rows = samples(args.data_root, args.dataset, args.split, cats)
    if args.require_epoch200:
        log = args.checkpoint.with_name('log.txt').read_text()
        if 'epoch [200/200]' not in log or 'Mean:' not in log.split('epoch [200/200]')[-1]:
            p.error('No completed epoch-200 evaluation in companion log')
        tail = log.split('epoch [200/200]')[-1]
        if any(f'{cat}: I-Auroc:' not in tail for cat in cats):
            p.error('Companion epoch-200 log lacks requested categories')
    if args.smoke_images:
        rows = rows[:args.smoke_images]
    import numpy as np
    import tifffile
    import torch
    from torch.nn import functional as F
    from torchvision import transforms
    from PIL import Image
    from dataset import get_data_transforms
    from utils import cal_anomaly_maps, get_gaussian_kernel

    torch.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA is unavailable; no silent CPU fallback')
    checkpoint_hash = sha256(args.checkpoint)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = build_model(args.m, state, args.device, use_inp=not args.no_inp)
    del state
    if sha256(args.checkpoint) != checkpoint_hash:
        raise RuntimeError('Checkpoint changed while being loaded')
    if args.geometry == 'legacy-crop256':
        transform = get_data_transforms(448, 392)[0]
        dtype = np.float32
    else:
        transform = transforms.Compose([
            transforms.Resize((392, 392)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        dtype = np.float16  # Exactly the dtype required by official AD2 checker.
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device).eval()
    args.output.mkdir(parents=True)
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            images = []
            for row in batch:
                path = args.data_root / row['image']
                row['image_sha256'] = sha256(path)
                row['mask_sha256'] = sha256(args.data_root / row['mask']) if row['mask'] else None
                with Image.open(path) as img:
                    row['native_hw'] = [img.height, img.width]
                    images.append(transform(img.convert('RGB')))
            x = torch.stack(images).to(args.device)
            en, de = model(x)[:2]
            amap = cal_anomaly_maps(en, de, 392)[0]
            amap = F.interpolate(amap, size=(256, 256), mode='bilinear', align_corners=False)
            amap = gaussian(amap)
            for row, one in zip(batch, amap):
                flat = one.flatten()
                row['image_score'] = float(torch.sort(flat, descending=True).values[:int(flat.numel() * .01)].mean().cpu())
                # Retain pre-native, float32 model map for exact provenance/debugging.
                small_path = Path('model_maps256') / Path(row['image']).with_suffix('.npy')
                (args.output / small_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(args.output / small_path, one[0].cpu().numpy())
                row['model_map256'] = small_path.as_posix()
                if args.geometry == 'full-frame':
                    one = F.interpolate(one[None], size=row['native_hw'], mode='bilinear', align_corners=False)[0]
                array = one[0].cpu().numpy().astype(dtype)
                if not np.isfinite(array).all():
                    raise ValueError(f'Nonfinite map: {row["image"]}')
                destination = args.output / row['prediction']
                destination.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(destination, array, photometric='minisblack', metadata=None)
                row['prediction_sha256'] = sha256(destination)
            print(f'{min(start + args.batch_size, len(rows))}/{len(rows)} maps', flush=True)
    manifest = {
        'complete': not bool(args.smoke_images), 'dataset': args.dataset,
        'data_root': str(args.data_root.resolve()), 'split': args.split, 'categories': cats,
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': checkpoint_hash,
        'geometry': args.geometry, 'dtype': np.dtype(dtype).name, 'm': args.m,
        'use_inp': not args.no_inp,
        'architecture': 'encoder-patch-memory-cross-attention' if args.no_inp else 'inp-guided-decoder',
        'image_score_definition': 'mean of largest floor(0.01*256*256) scores in smoothed float32 model map',
        'map_pipeline': 'cosine maps -> bilinear 392 align_corners=True -> bilinear 256 False -> Gaussian k5 sigma4 -> native bilinear False (full-frame only)',
        'input_pipeline': 'resize full image to 392' if args.geometry == 'full-frame' else 'resize448, centercrop392',
        'software': {'torch': torch.__version__, 'numpy': np.__version__},
        'source_hashes': {str(f.relative_to(ROOT)): sha256(f) for f in
                          [Path(__file__).resolve(), ROOT/'models/uad.py', ROOT/'models/vision_transformer.py', ROOT/'utils.py', ROOT/'dataset.py']},
        'samples': rows,
    }
    write_json(args.output / 'manifest.json', manifest)
    print(f'Saved {args.output / "manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
