"""Run a fixed-M AD2 ablation using an archived copy of the original trainer.

Only the requested INP switch and loss arguments change. Timing and GPU
sampling do not change the optimizer, scheduler, initialization or data loader.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
CONFIGS = {
    'inp_only': dict(loss_y=0, loss_lambda=0.0, no_inp=False),
    'inp_lsm': dict(loss_y=3, loss_lambda=0.0, no_inp=False),
    'inp_lc': dict(loss_y=0, loss_lambda=0.2, no_inp=False),
    'legacy_no_inp': dict(loss_y=0, loss_lambda=0.0, no_inp=True),
}
FULL_MEAN_EPOCH_SECONDS = 142.37688442211055
FULL_MEDIAN_EPOCH_SECONDS = 142.0


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def prepare(args):
    audit = json.loads((args.audit_dir / 'full_audit.json').read_text())
    smoke_name = args.config + '_smoke.json'
    smoke = json.loads((args.audit_dir / smoke_name).read_text())
    assert smoke['passed'] and len(smoke['steps']) == 2
    assert sha256(ROOT / 'INP_Former_Multi_Class.py') == audit['training_source_sha256']
    baseline_manifest = json.loads((ROOT / 'evaluation_results/aupro-reconciliation-v1/full-frame/M6/public_predictions/manifest.json').read_text())
    for filename in ['models/uad.py', 'models/vision_transformer.py', 'utils.py', 'dataset.py']:
        assert sha256(ROOT / filename) == baseline_manifest['source_hashes'][filename], filename
    args.run_dir.mkdir(parents=True, exist_ok=False)
    source = args.run_dir / 'source'
    files = subprocess.check_output(['git', 'ls-files', '--', '*.py'], cwd=ROOT, text=True).splitlines()
    files += ['export_mvtec_predictions.py', 'evaluate_mvtec_official.py', 'mvtec_official_adapter.py',
              'scripts/run_ad2_component.py']
    for filename in sorted(set(files)):
        destination = source / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / filename, destination)
    shutil.copytree(ROOT / 'third_party', source / 'third_party', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(args.audit_dir / 'training_manifest.json', args.run_dir / 'training_manifest.json')
    hashes = {str(p.relative_to(source)): sha256(p) for p in source.rglob('*') if p.is_file()}
    config = dict(dataset='MVTec-AD2', data_path=str(ROOT / 'data/mvtec_ad2'),
                  item_list=CATEGORIES, seed=1, encoder='dinov2reg_vit_base_14',
                  input_size=448, crop_size=392, INP_num=6, batch_size=16,
                  total_epochs=200, phase='train', **CONFIGS[args.config],
                  eval_epochs='', load_from=None,
                  save_dir=str(args.run_dir.parent), save_name=args.run_dir.name)
    save(args.run_dir / 'prepared.json', {
        'prepared_utc': timestamp(), 'configuration': args.config, 'args': config,
        'architecture': 'encoder-patch-memory-cross-attention' if config['no_inp'] else 'inp-guided-decoder',
        'git_head': audit['git_head'], 'source_hashes': hashes,
        'training_manifest_sha256': sha256(args.run_dir / 'training_manifest.json'),
        'backbone_sha256': audit['pretrained_sha256'],
        'smoke_evidence': str((args.audit_dir / smoke_name).resolve()),
        'smoke_sha256': sha256(args.audit_dir / smoke_name),
        'original_full_checkpoint_sha256': audit['full_checkpoint_sha256'],
        'full_provenance_limit': audit['provenance_limit'],
        'public_geometry': 'full-frame', 'walltime_hours': 12,
    })
    print(f'PREPARED {args.run_dir}', flush=True)


def run(args):
    start = time.time()
    prepared = json.loads((args.run_dir / 'prepared.json').read_text())
    assert prepared['configuration'] == args.config
    save(args.run_dir / 'execution_claim.json', {'job_id': os.environ.get('SLURM_JOB_ID'), 'started_utc': timestamp()})
    source = args.run_dir / 'source'
    for filename, expected in prepared['source_hashes'].items():
        assert sha256(source / filename) == expected, f'Archived source changed: {filename}'
    assert sha256(args.run_dir / 'training_manifest.json') == prepared['training_manifest_sha256']
    training_manifest = json.loads((args.run_dir / 'training_manifest.json').read_text())
    for row in training_manifest['samples']:
        assert sha256(ROOT / 'data/mvtec_ad2' / row['path']) == row['sha256'], row['path']
    assert sha256(ROOT / 'backbones/weights/dinov2_vitb14_reg4_pretrain.pth') == prepared['backbone_sha256']
    os.chdir(ROOT)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    sys.path.insert(0, str(source))
    import torch
    import INP_Former_Multi_Class as train
    assert Path(train.__file__).resolve().parent == source
    assert torch.cuda.is_available()
    gpu = torch.cuda.get_device_properties(0)
    assert 'A100' in gpu.name, gpu.name
    gpu_id = str(gpu.uuid)
    save(args.run_dir / 'runtime.json', {
        'job_id': os.environ.get('SLURM_JOB_ID'), 'node': os.environ.get('SLURMD_NODENAME'),
        'gpu': gpu.name, 'gpu_uuid': gpu_id, 'torch': torch.__version__,
        'torch_threads': torch.get_num_threads(), 'entrypoint': train.__file__,
        'args': prepared['args'], 'started_utc': timestamp(),
        'CUDA_LAUNCH_BLOCKING': os.environ['CUDA_LAUNCH_BLOCKING'],
    })
    samples = []
    stop = threading.Event()

    def sample_gpu():
        with (args.run_dir / 'gpu_samples.jsonl').open('x', buffering=1) as stream:
            while not stop.is_set():
                row = {'unix_time': time.time(), 'utc': timestamp(), 'gpu': gpu.name}
                try:
                    raw = subprocess.check_output([
                        'nvidia-smi', '-i', gpu_id,
                        '--query-gpu=utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits'],
                        text=True, timeout=5).strip().split(',')
                    row.update(utilization_percent=float(raw[0]), memory_used_mib=float(raw[1]), power_w=float(raw[2]))
                except (subprocess.SubprocessError, ValueError, OSError) as exc:
                    row['error'] = str(exc)
                samples.append(row)
                stream.write(json.dumps(row) + '\n')
                stop.wait(15)

    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()
    durations = []
    original_tqdm = train.tqdm
    epoch_stream = (args.run_dir / 'epoch_times.jsonl').open('x', buffering=1)

    def timed_tqdm(loader, **kwargs):
        torch.cuda.synchronize()
        epoch_start = time.time()
        clock_start = time.perf_counter()
        yield from original_tqdm(loader, **kwargs)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - clock_start
        durations.append(seconds)
        epoch = len(durations)
        row = {'epoch': epoch, 'seconds': seconds, 'start_unix': epoch_start,
               'end_unix': time.time(), 'completed_utc': timestamp()}
        epoch_stream.write(json.dumps(row) + '\n')
        if epoch in (5, 10):
            post_warmup = durations[1:]
            mean = statistics.mean(post_warmup)
            remaining = (200 - epoch) * mean
            util = [s['utilization_percent'] for s in samples
                    if s['unix_time'] >= row['end_unix'] - sum(post_warmup) and 'utilization_percent' in s]
            report = {
                'completed_epochs': epoch, 'utc': timestamp(), 'warmup_exclusion': 'epoch 1 (100 warmup iterations; 158 batches/epoch)',
                'mean_post_warmup_epoch_s': mean, 'median_post_warmup_epoch_s': statistics.median(post_warmup),
                'remaining_training_s': remaining,
                'estimated_training_completion_utc': (datetime.now(timezone.utc) + timedelta(seconds=remaining)).isoformat(),
                'gpu': gpu.name, 'gpu_samples': len(util),
                'gpu_utilization_mean_percent': statistics.mean(util) if util else None,
                'gpu_utilization_median_percent': statistics.median(util) if util else None,
                'full_mean_post_warmup_epoch_s': FULL_MEAN_EPOCH_SECONDS,
                'full_median_post_warmup_epoch_s': FULL_MEDIAN_EPOCH_SECONDS,
                'epoch_time_ratio_to_full': mean / FULL_MEAN_EPOCH_SECONDS,
                'full_job_runtime_s': 28707,
                'previous_full_public_inference_s': 240,
                'previous_full_public_cpu_evaluation_s': 2267,
                'previous_public_cpu_evaluation_range_s': [2267, 3013],
                'inference_estimate_limit': 'Previous Full L4 run; No-INP and A100 may differ.',
                'walltime_hours': 12,
                'walltime_risk_with_10min_final_eval_and_10min_export_allowance':
                    time.time() - start + remaining + 1200 > 12 * 3600,
                'queue_time_included': False,
            }
            save(args.run_dir / f'timing_epoch{epoch}.json', report)
            print('TIMING_REPORT ' + json.dumps(report), flush=True)

    train.tqdm = timed_tqdm
    train.device = 'cuda:0'
    train.print_fn = train.get_logger(args.run_dir.name, str(args.run_dir)).info
    print('EFFECTIVE_TRAINING_ARGS ' + json.dumps(prepared['args']), flush=True)
    try:
        train.main(SimpleNamespace(**prepared['args']))
    finally:
        epoch_stream.close()
        stop.set()
        sampler.join(timeout=6)
    assert len(durations) == 200 and (args.run_dir / 'model.pth').is_file()
    save(args.run_dir / 'training_complete.json', {'completed_utc': timestamp(), 'epochs': 200,
         'training_epoch_seconds': sum(durations), 'checkpoint_sha256': sha256(args.run_dir / 'model.pth')})
    command = [sys.executable, str(source / 'export_mvtec_predictions.py'), '--dataset', 'AD2',
               '--data-root', str(ROOT / 'data/mvtec_ad2'), '--checkpoint', str(args.run_dir / 'model.pth'),
               '--output', str(args.run_dir / 'public_predictions'), '--split', 'test_public',
               '--m', '6', '--geometry', 'full-frame', '--require-epoch200']
    if prepared['args']['no_inp']:
        command.append('--no-inp')
    save(args.run_dir / 'public_inference_command.json', command)
    print('PUBLIC_INFERENCE_COMMAND ' + json.dumps(command), flush=True)
    subprocess.run(command, check=True)
    save(args.run_dir / 'public_inference_complete.json', {'completed_utc': timestamp()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, choices=sorted(CONFIGS))
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--audit-dir', type=Path, default=ROOT / 'evaluation_results/ad2-component-audit-20260919')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.audit_dir = args.audit_dir.resolve()
    if args.prepare_only:
        prepare(args)
    else:
        run(args)
