"""CPU-only evaluation of continuous TIFFs with MVTec's unchanged AU-PRO code.

The manifest defines the exact population and geometry. Native/full-frame maps
are benchmark inputs; legacy-crop256 is an explicitly separate diagnostic.
"""
import argparse
import gc
import json
from pathlib import Path

from mvtec_official_adapter import (CATEGORIES, VENDOR, load_mask, official_aupro,
                                    samples, sha256, write_json)


def evaluate(manifest_path, output, limits, max_memory_gib):
    import numpy as np
    import tifffile
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if not manifest['complete'] or 'private' in manifest['split']:
        raise ValueError('Requires complete public/test predictions; private evaluation is server-only')
    root = Path(manifest['data_root'])
    rows = manifest['samples']
    expected = samples(root, manifest['dataset'], manifest['split'], manifest['categories'])
    if [r['image'] for r in rows] != [r['image'] for r in expected]:
        raise ValueError('Manifest is not the complete ordered test population')
    for row, ref in zip(rows, expected):
        for key in ('category', 'mask', 'label', 'prediction'):
            if row[key] != ref[key]:
                raise ValueError(f'Incorrect {key}: {row["image"]}')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = {'manifest_sha256': sha256(manifest_path), 'limits': limits,
              'geometry': manifest['geometry'], 'dtype': manifest['dtype'],
              'official_source_manifest_sha256': sha256(VENDOR.parent/'SOURCE.json'),
              'adapter_sha256': sha256(Path(__file__).with_name('mvtec_official_adapter.py')),
              'runner_sha256': sha256(__file__)}
    category_results = {}
    for cat in manifest['categories']:
        destination = output / (cat + '.json')
        if destination.exists():
            prior = json.loads(destination.read_text())
            if prior['source'] != source:
                raise ValueError(f'Refusing to overwrite a different evaluation: {destination}')
            category_results[cat] = prior['metrics']
            continue
        cat_rows = [r for r in rows if r['category'] == cat]
        total_pixels = sum(256**2 if manifest['geometry'] == 'legacy-crop256'
                           else r['native_hw'][0] * r['native_hw'][1] for r in cat_rows)
        # Conservative capacity guard, NOT a change to the official algorithm.
        estimated_gib = total_pixels * 80 / 2**30
        if estimated_gib > max_memory_gib or total_pixels >= 2**32 - 1:
            raise MemoryError(f'{cat}: estimated {estimated_gib:.1f} GiB; request more RAM, do not downsample silently')
        predictions, masks, scores, max_scores, labels = [], [], [], [], []
        for row in cat_rows:
            path = manifest_path.parent / row['prediction']
            if sha256(path) != row['prediction_sha256']:
                raise ValueError(f'Prediction changed: {path}')
            if sha256(root / row['image']) != row['image_sha256']:
                raise ValueError(f'Input image changed: {row["image"]}')
            if row['mask'] and sha256(root / row['mask']) != row['mask_sha256']:
                raise ValueError(f'GT changed: {row["mask"]}')
            with Image.open(root / row['image']) as image:
                if [image.height, image.width] != row['native_hw']:
                    raise ValueError('Incorrect native image shape')
            pred = tifffile.imread(path)
            mask = load_mask(root, row, manifest['geometry'])
            if (pred.shape != mask.shape or not np.isfinite(pred).all() or pred.dtype.kind != 'f'
                    or str(pred.dtype) != manifest['dtype'] or not np.isfinite(row['image_score'])):
                raise ValueError(f'Invalid map geometry/dtype/values: {path}')
            predictions.append(pred)
            masks.append(mask)
            scores.append(row['image_score'])
            max_scores.append(float(pred.max()))
            labels.append(row['label'])
        metrics = official_aupro(predictions, masks, limits)
        # AUROC has no dependency on either custom or official PRO code.
        metrics['I-AUROC'] = float(roc_auc_score(labels, scores))
        metrics['I-AUROC_max_map'] = float(roc_auc_score(labels, max_scores))
        metrics['P-AUROC'] = float(roc_auc_score(np.asarray(masks).ravel(), np.asarray(predictions).ravel()))
        counts = {'images': len(labels), 'normal': labels.count(0), 'anomalous': labels.count(1)}
        write_json(destination, {'source': source, 'metrics': metrics, 'counts': counts})
        category_results[cat] = metrics
        print(cat, metrics, flush=True)
        del predictions, masks
        gc.collect()
    names = next(iter(category_results.values())).keys()
    mean = {name: float(np.mean([row[name] for row in category_results.values()])) for name in names}
    result = {'source': source, 'dataset': manifest['dataset'], 'split': manifest['split'],
              'categories': manifest['categories'], 'per_category': category_results, 'mean': mean,
              'scale': '0..1', 'aggregation': 'unweighted category mean',
              'metric': 'MVTec AD evaluation v1.0, unchanged; recommended by AD2 README',
              'image_score': manifest['image_score_definition'],
              'full_category_set': manifest['categories'] == CATEGORIES[manifest['dataset']]}
    write_json(output / 'summary.json', result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--fpr-limits', nargs='+', type=float, default=[0.05, 0.30])
    p.add_argument('--max-memory-gib', type=float, default=100)
    a = p.parse_args()
    evaluate(a.manifest, a.output, a.fpr_limits, a.max_memory_gib)


if __name__ == '__main__':
    main()
