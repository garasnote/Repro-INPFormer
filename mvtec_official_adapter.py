"""Dataset/prediction adapter. AU-PRO is delegated to unchanged MVTec code."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'third_party/mvtec/mvtec_ad_evaluation'
CATEGORIES = {
    'AD2': ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts'],
    'AD': ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
           'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
    'VisA': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
             'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'],
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def samples(root, dataset, split, categories):
    """Enumerate ALL test images; never filter using mask contents."""
    root = Path(root)
    if dataset != 'AD2' and split != 'test':
        raise ValueError('AD/VisA require split=test')
    if dataset == 'AD2' and split not in ('test_public', 'test_private', 'test_private_mixed'):
        raise ValueError('Unsupported AD2 split')
    if not categories or len(set(categories)) != len(categories):
        raise ValueError('Empty or duplicate category list')
    if set(categories) - set(CATEGORIES[dataset]):
        raise ValueError('Unknown category')
    result = []
    for cat in categories:
        base = root / cat / split
        private = 'private' in split
        paths = sorted(p for p in (base.glob('*') if private else base.glob('*/*'))
                       if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.bmp', '.jpeg'))
        # ground_truth/bad/*.png is deliberately not a test image.
        if not paths:
            raise FileNotFoundError(f'No images: {base}')
        for p in paths:
            label = None if private else int(p.parent.name != 'good')
            gt = None
            if label:
                gt = (base / 'ground_truth/bad' if dataset == 'AD2'
                      else root / cat / 'ground_truth' / p.parent.name) / (p.stem + '_mask.png')
                if not gt.is_file():
                    raise FileNotFoundError(gt)
            result.append({'category': cat, 'image': p.relative_to(root).as_posix(),
                           'mask': gt.relative_to(root).as_posix() if gt else None,
                           'label': label,
                           'prediction': (Path('anomaly_images') / p.relative_to(root)).with_suffix('.tiff').as_posix()})
    return result


def official_modules():
    provenance = json.loads((VENDOR.parent / 'SOURCE.json').read_text())
    modules = []
    for name in ('pro_curve_util', 'generic_util'):
        path = VENDOR / (name + '.py')
        expected = provenance['files'][path.relative_to(VENDOR.parent).as_posix()]
        if sha256(path) != expected:
            raise RuntimeError(f'Official source changed: {path}')
        spec = importlib.util.spec_from_file_location('official_mvtec_' + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def official_aupro(predictions, masks, limits=(0.05, 0.30)):
    """Exact calls/normalization from MVTec evaluate_experiment.py, no fallback."""
    import numpy as np
    if not predictions or len(predictions) != len(masks):
        raise ValueError('Empty/mismatched population')
    shape = predictions[0].shape
    for pred, mask in zip(predictions, masks):
        if pred.ndim != 2 or pred.shape != shape or mask.shape != shape:
            raise ValueError('Official implementation requires equal 2D shapes within each category')
        if not np.isfinite(pred).all() or not np.isin(mask, (0, 1)).all():
            raise ValueError('Nonfinite prediction or nonbinary mask')
    if not any(m.any() for m in masks) or not any((m == 0).any() for m in masks):
        raise ValueError('PRO requires positive regions and negative pixels')
    if any(not 0 < limit <= 1 for limit in limits):
        raise ValueError('Invalid FPR limit')
    pro, generic = official_modules()
    fprs, pros = pro.compute_pro(anomaly_maps=predictions, ground_truth_maps=masks)
    # This is the upstream evaluator's existing integration and normalization.
    return {f'AU-PRO_{limit:.2f}': float(generic.trapezoid(fprs, pros, x_max=limit) / limit)
            for limit in limits}


def load_mask(root, row, geometry):
    import numpy as np
    from PIL import Image
    if row['label'] is None:
        raise ValueError('Private labels unavailable; use the server')
    if row['mask'] is None:
        shape = (256, 256) if geometry == 'legacy-crop256' else tuple(row['native_hw'])
        return np.zeros(shape, dtype=np.uint8)
    with Image.open(Path(root) / row['mask']) as img:
        mask = img.convert('L')
        if geometry == 'legacy-crop256':
            # Historical GT geometry, explicitly a crop-domain diagnostic.
            mask = mask.resize((448, 448), Image.Resampling.BILINEAR).crop((28, 28, 420, 420))
            arr = np.asarray(mask)
            # Same index selection as torch interpolate(nearest), 392 -> 256.
            idx = np.floor(np.arange(256) * 392 / 256).astype(int)
            return (arr[np.ix_(idx, idx)] > 127.5).astype(np.uint8)
        return (np.asarray(mask) > 0).astype(np.uint8)
