"""Stage both private splits as continuous TIFFs; never upload anything.

Uses hard links (no conversion) into a clean server directory. Then run the
unmodified MVTec check_and_prepare_data_for_upload.py on that directory.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mvtec_official_adapter import CATEGORIES, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests', nargs=2, required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output must be a new directory')
    loaded = [(path, json.loads(path.read_text())) for path in a.manifests]
    if {m['split'] for _, m in loaded} != {'test_private', 'test_private_mixed'}:
        p.error('Exactly one manifest for each private split is required')
    ref = loaded[0][1]
    links = []
    import numpy as np
    import tifffile
    for path, manifest in loaded:
        if (not manifest['complete'] or manifest['dataset'] != 'AD2' or
                manifest['categories'] != CATEGORIES['AD2'] or manifest['geometry'] != 'full-frame'):
            p.error('Requires complete, eight-category full-frame AD2 inference')
        for key in ['checkpoint_sha256', 'geometry', 'dtype', 'm', 'map_pipeline', 'input_pipeline', 'source_hashes']:
            if manifest[key] != ref[key]:
                p.error(f'Mismatched private protocols: {key}')
        for row in manifest['samples']:
            source = path.parent/row['prediction']
            if sha256(source) != row['prediction_sha256']:
                p.error(f'Prediction hash mismatch: {source}')
            arr = tifffile.imread(source)
            if arr.dtype != np.float16 or list(arr.shape) != row['native_hw'] or not np.isfinite(arr).all():
                p.error(f'Invalid server TIFF: {source}')
            if np.unique(arr).size <= 2:
                p.error(f'Prediction is binary or constant, not a continuous anomaly map: {source}')
            links.append((source, a.output/row['prediction']))
    for source, destination in links:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
    print(f'Staged {len(links)} unchanged TIFFs in {a.output}. No upload performed.')


if __name__ == '__main__':
    main()
