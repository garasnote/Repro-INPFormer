"""Verify upstream metric parity and fail-closed dataset/manifest handling."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mvtec_official_adapter import VENDOR, official_aupro, samples, sha256, write_json
from evaluate_mvtec_official import evaluate


class OfficialEvaluationTests(unittest.TestCase):
    def test_upstream_file_api_parity_both_cutoffs_and_ties(self):
        import numpy as np
        import tifffile
        from PIL import Image
        sys.path.insert(0, str(VENDOR))
        try:
            spec = importlib.util.spec_from_file_location('upstream_experiment', VENDOR/'evaluate_experiment.py')
            upstream = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(upstream)
            with tempfile.TemporaryDirectory() as tmp:
                masks = [np.zeros((5, 5), dtype=np.uint8) for _ in range(3)]
                masks[1][1:3, 1:3] = 1
                masks[2][3:5, 3:5] = 1
                # Discrete tied scores; cutoff lies between empirical FPR steps.
                rng = np.random.default_rng(7)
                preds = [rng.integers(0, 7, (5, 5)).astype(np.float32) for _ in masks]
                gt_paths, pred_paths = [], []
                for i, (gt, pred) in enumerate(zip(masks, preds)):
                    f = Path(tmp)/str(i)
                    tifffile.imwrite(f.with_suffix('.tiff'), pred)
                    Image.fromarray(gt*255).save(f.with_suffix('.png'))
                    gt_paths.append(str(f.with_suffix('.png')) if gt.any() else None)
                    pred_paths.append(str(f))
                actual = official_aupro(preds, masks)
                for limit in (0.05, 0.3):
                    expected = upstream.calculate_au_pro_au_roc(gt_paths, pred_paths, limit)[0]
                    self.assertEqual(actual[f'AU-PRO_{limit:.2f}'], expected)
        finally:
            sys.path.remove(str(VENDOR))

    def test_perfect_maps_and_input_guards(self):
        import numpy as np
        masks = [np.zeros((3, 3), dtype=np.uint8), np.eye(3, dtype=np.uint8)]
        self.assertEqual(official_aupro([m.astype(float) for m in masks], masks),
                         {'AU-PRO_0.05': 1., 'AU-PRO_0.30': 1.})
        with self.assertRaises(ValueError):
            official_aupro([np.full((3, 3), np.nan)]*2, masks)
        with self.assertRaises(ValueError):
            official_aupro([np.zeros((3, 3))]*2, [np.zeros((3, 3))]*2)

    def test_manifest_end_to_end_and_missing_population(self):
        import numpy as np
        import tifffile
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'data'
            out = Path(tmp)/'predictions'
            out.mkdir()
            for kind in ['good', 'bad']:
                p = root/'can/test_public'/kind/'000_regular.png'
                p.parent.mkdir(parents=True)
                Image.new('RGB', (8, 8)).save(p)
            gt = np.zeros((8, 8), dtype=np.uint8)
            gt[2:4, 2:4] = 255
            p = root/'can/test_public/ground_truth/bad/000_regular_mask.png'
            p.parent.mkdir(parents=True)
            Image.fromarray(gt).save(p)
            rows = samples(root, 'AD2', 'test_public', ['can'])
            self.assertEqual(len(rows), 2)
            for row in rows:
                pred = (gt/255).astype(np.float16) if row['label'] else np.zeros((8, 8), np.float16)
                p = out/row['prediction']
                p.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(p, pred)
                row.update(native_hw=[8, 8], image_score=float(row['label']),
                           prediction_sha256=sha256(p), image_sha256=sha256(root/row['image']),
                           mask_sha256=sha256(root/row['mask']) if row['mask'] else None)
            manifest = dict(complete=True, data_root=str(root), dataset='AD2', split='test_public',
                            categories=['can'], geometry='full-frame', dtype='float16',
                            image_score_definition='fixture', samples=rows)
            write_json(out/'manifest.json', manifest)
            evaluate(out/'manifest.json', Path(tmp)/'metrics', [0.05, .3], 1)
            result = json.loads((Path(tmp)/'metrics/summary.json').read_text())
            self.assertEqual(result['mean']['I-AUROC'], 1)
            self.assertEqual(result['mean']['P-AUROC'], 1)
            self.assertFalse(result['full_category_set'])
            manifest['samples'] = rows[:1]
            write_json(out/'manifest.json', manifest)
            with self.assertRaises(ValueError):
                evaluate(out/'manifest.json', Path(tmp)/'bad_metrics', [0.05], 1)
            with self.assertRaises(FileNotFoundError):
                samples(root, 'AD2', 'test_public', ['can', 'fabric'])


if __name__ == '__main__':
    unittest.main()
