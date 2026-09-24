"""Audit the existing Full checkpoint and smoke-test the unchanged trainer.

Creates only a new audit directory. Does not save a trained model or modify
the training implementation. No-INP here explicitly means the legacy branch.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    'inp_only': dict(loss_y=0, loss_lambda=0.0, no_inp=False),
    'inp_lsm': dict(loss_y=3, loss_lambda=0.0, no_inp=False),
    'inp_lc': dict(loss_y=0, loss_lambda=0.2, no_inp=False),
    'legacy_no_inp': dict(loss_y=0, loss_lambda=0.0, no_inp=True),
}
sys.path.insert(0, str(ROOT))
CATEGORIES = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
FULL = ROOT / 'saved_results' / ('INP-Former-Multi-Class-8cat-sweep_dataset=MVTec-AD2_'
    'Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6_y=3_lambda=0.2_seed=1_INP')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--configs', nargs='+', choices=sorted(CONFIGS), default=['inp_only'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    import torch
    import INP_Former_Multi_Class as train
    from export_mvtec_predictions import build_model

    assert torch.cuda.is_available(), 'Smoke requires an allocated CUDA GPU'
    assert 'A100' in torch.cuda.get_device_name(0), 'Smoke requires an A100'
    torch.set_num_threads(4)
    checkpoint = torch.load(FULL / 'model.pth', map_location='cpu', weights_only=True)
    assert tuple(checkpoint['prototype_token'].shape) == (6, 768)
    assert all(torch.isfinite(t).all() for t in checkpoint.values())
    model = build_model(6, checkpoint, 'cpu')
    backbone_path = ROOT / 'backbones/weights/dinov2_vitb14_reg4_pretrain.pth'
    backbone = torch.load(backbone_path, map_location='cpu', weights_only=True)
    encoder_keys = [key for key in checkpoint if key.startswith('encoder.')]
    different = [key for key in encoder_keys
                 if key[8:] not in backbone or not torch.equal(checkpoint[key], backbone[key[8:]])]
    assert not different, f'Full encoder differs from cached pretrained weights: {different}'
    log = (FULL / 'log.txt').read_text()
    epoch_numbers = [int(e) for e in re.findall(r'^epoch \[(\d+)/200\]', log, re.M)]
    assert epoch_numbers == list(range(1, 201))
    tail = log.split('epoch [200/200]')[-1]
    assert all(f'{cat}: I-Auroc:' in tail for cat in CATEGORIES)
    assert 'Mean:' in tail
    manifest_path = ROOT / 'evaluation_results/aupro-reconciliation-v1/full-frame/M6/public_predictions/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert sha256(FULL / 'model.pth') == manifest['checkpoint_sha256']
    source_match = {key: sha256(ROOT / key) == value for key, value in manifest['source_hashes'].items()}
    assert all(source_match.values()), source_match
    report = {
        'full_checkpoint': str(FULL / 'model.pth'),
        'full_checkpoint_sha256': manifest['checkpoint_sha256'],
        'strict_state_load': True, 'all_checkpoint_tensors_finite': True,
        'prototype_shape': list(checkpoint['prototype_token'].shape),
        'checkpoint_tensor_count': len(checkpoint),
        'encoder_tensors_equal_pretrained': len(encoder_keys),
        'pretrained_sha256': sha256(backbone_path),
        'epochs': epoch_numbers, 'categories': CATEGORIES,
        'export_source_hashes_match_current': source_match,
        'original_training_source_snapshot_present': False,
        'provenance_limit': 'Original command is recoverable from Slurm; the original job did not archive source hashes or a training-image hash manifest.',
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'training_source_sha256': sha256(ROOT / 'INP_Former_Multi_Class.py'),
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'job_id': os.environ.get('SLURM_JOB_ID'),
    }
    del model, checkpoint, backbone
    gc.collect()
    # ImageFolder matches the actual training file discovery, ordering and extensions.
    data_transform = train.get_data_transforms(448, 392)[0]
    records = []
    counts = {}
    for category in CATEGORIES:
        dataset = train.ImageFolder(ROOT / 'data/mvtec_ad2' / category / 'train', transform=data_transform)
        assert dataset.classes == ['good']
        counts[category] = len(dataset)
        for filename, _ in dataset.samples:
            path = Path(filename)
            records.append({'path': str(path.relative_to(ROOT / 'data/mvtec_ad2')),
                            'sha256': sha256(path), 'size': path.stat().st_size,
                            'mtime_ns': path.stat().st_mtime_ns})
    assert len(records) == 2528
    save(args.output / 'training_manifest.json', {'categories': CATEGORIES, 'counts': counts, 'samples': records})
    report['training_counts'] = counts
    report['training_manifest_sha256'] = sha256(args.output / 'training_manifest.json')
    save(args.output / 'full_audit.json', report)
    print('FULL_AUDIT ' + json.dumps(report), flush=True)

    class SmokeComplete(Exception):
        pass

    original_init = train.INP_Former.__init__
    original_step = train.StableAdamW.step
    for configuration in args.configs:
        evidence = {'configuration': configuration, 'steps': []}
        captured = {}

        def capture_init(self, *a, **kw):
            original_init(self, *a, **kw)
            captured['model'] = self

        def checked_step(optimizer, *a, **kw):
            model = captured['model']
            assert all(p.grad is None for p in model.encoder.parameters())
            assert not ({id(p) for p in model.encoder.parameters()} &
                        {id(p) for group in optimizer.param_groups for p in group['params']})
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    assert torch.isfinite(parameter.grad).all(), name
            assert model.bottleneck[0].fc1.weight.grad is not None
            assert model.decoder[0].attn.q.weight.grad is not None
            if configuration == 'legacy_no_inp':
                assert model.prototype_token.grad is None
                assert all(p.grad is None for p in model.aggregation.parameters())
            else:
                assert model.prototype_token.grad is not None
            before = model.bottleneck[0].fc1.weight.detach().clone()
            result = original_step(optimizer, *a, **kw)
            torch.cuda.synchronize()
            evidence['steps'].append({
                'step': len(evidence['steps']) + 1,
                'finite_gradients': True, 'encoder_has_no_gradients': True,
                'encoder_excluded_from_optimizer': True,
                'bottleneck_changed': not torch.equal(before, model.bottleneck[0].fc1.weight),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'lr': optimizer.param_groups[0]['lr'],
            })
            print('SMOKE_STEP ' + json.dumps(evidence['steps'][-1]), flush=True)
            if len(evidence['steps']) == 2:
                raise SmokeComplete()
            return result

        train.INP_Former.__init__ = capture_init
        train.StableAdamW.step = checked_step
        train.device = 'cuda:0'
        train.print_fn = lambda message: print(message, flush=True)
        config = dict(dataset='MVTec-AD2', data_path=str(ROOT / 'data/mvtec_ad2'),
                      item_list=CATEGORIES, seed=1, encoder='dinov2reg_vit_base_14',
                      input_size=448, crop_size=392, INP_num=6, batch_size=16,
                      total_epochs=200, phase='train', **CONFIGS[configuration],
                      eval_epochs='', load_from=None,
                      save_dir=str(args.output), save_name=configuration)
        evidence['args'] = config
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        try:
            train.main(SimpleNamespace(**config))
        except SmokeComplete:
            evidence['passed'] = True
        finally:
            train.INP_Former.__init__ = original_init
            train.StableAdamW.step = original_step
        assert evidence.get('passed'), 'Smoke did not reach two optimizer steps'
        assert evidence['steps'][1]['bottleneck_changed'], 'No parameter update at nonzero LR'
        evidence['elapsed_seconds'] = time.monotonic() - start
        save(args.output / f'{configuration}_smoke.json', evidence)
        print('SMOKE_PASS ' + json.dumps(evidence), flush=True)
        captured.clear()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
