"""Print a dependency-safe inference/CPU-evaluation plan; --submit enqueues it.

No training commands. Existing checkpoints are required for every selected model.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mvtec_official_adapter import write_json


def submit(command):
    # FRIDA's sbatch wrapper prints a banner before the parsable job ID.
    raw = subprocess.check_output(command, cwd=ROOT, text=True)
    ids = [line.split(';')[0] for line in raw.splitlines()
           if re.fullmatch(r'\d+(;[A-Za-z0-9_.-]+)?', line.strip())]
    if len(ids) != 1:
        raise RuntimeError(f'Could not identify submitted job; inspect scheduler before retry: {raw}')
    return ids[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', choices=['full-frame', 'legacy-crop256'], required=True)
    p.add_argument('--submit', action='store_true')
    p.add_argument('--run-tag', default='aupro-reconciliation-v1')
    p.add_argument('--models', nargs='+', type=int, choices=[1, 2, 4, 6, 8, 12, 16], default=[6, 2, 1])
    p.add_argument('--after-job', type=int, help='Additional successful validation job required before export')
    args = p.parse_args()
    if not args.run_tag or Path(args.run_tag).name != args.run_tag:
        p.error('run-tag must be one path component')
    output_root = ROOT/'evaluation_results'/args.run_tag/args.geometry
    ledger = output_root/'submitted_jobs.json'
    if args.submit and ledger.exists():
        p.error(f'Already submitted; inspect {ledger} before any retry')
    entries = []
    for m in args.models:
        checkpoint = ROOT/'saved_results'/(
            'INP-Former-Multi-Class-8cat-sweep_dataset=MVTec-AD2_Encoder=dinov2reg_vit_base_14_'
            f'Resize=448_Crop=392_INP_num={m}_y=3_lambda=0.2_seed=1_INP/model.pth')
        output = output_root/f'M{m}'/'public_predictions'
        if output.exists():
            p.error(f'Output already exists; refusing duplicate export: {output}')
        if not checkpoint.is_file():
            p.error(f'Checkpoint missing for M={m}; finish training first: {checkpoint}')
        dep_arg = [f'--dependency=afterok:{args.after_job}'] if args.after_job else []
        export = ['sbatch', '--parsable', *dep_arg, '--kill-on-invalid-dep=yes',
                  str(ROOT/'scripts/export_ad2_official.sh'), str(m), str(checkpoint), str(output), args.geometry]
        print(shlex.join(export), flush=True)
        if args.submit:
            export_id = submit(export)
        else:
            export_id = f'EXPORT_JOB_M{m}'
        entry = {'m':m, 'after_job':args.after_job, 'export_job':export_id, 'output':str(output)}
        entries.append(entry)
        if args.submit:
            write_json(ledger, entries)  # Preserve submitted job even if CPU submission fails.
        evaluate = ['sbatch', '--parsable', f'--dependency=afterok:{export_id}', '--kill-on-invalid-dep=yes',
                    str(ROOT/'scripts/evaluate_ad2_official_cpu.sh'), str(output/'manifest.json'),
                    str(output_root/f'M{m}'/'metrics')]
        print(shlex.join(evaluate), flush=True)
        if args.submit:
            entry['evaluation_job'] = submit(evaluate)
            write_json(ledger, entries)
    if args.submit:
        print(json.dumps(entries, indent=2))


if __name__ == '__main__':
    main()
