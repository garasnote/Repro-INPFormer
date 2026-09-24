"""Print progress and training ETA for the AD2 component runs."""
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'saved_results/ad2-components-M6-seed1-20260919'

for run in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    if (run / 'training_complete.json').exists():
        done = 'export done' if (run / 'public_inference_complete.json').exists() else 'exporting predictions'
        print(f'{run.name:15s} training complete, {done}')
        continue
    times = run / 'epoch_times.jsonl'
    if not times.exists():
        print(f'{run.name:15s} not started')
        continue
    rows = [json.loads(line) for line in times.read_text().splitlines()]
    post = [r['seconds'] for r in rows[1:]] or [rows[-1]['seconds']]
    mean = statistics.mean(post)
    eta = datetime.fromtimestamp(rows[-1]['end_unix'], timezone.utc) + timedelta(seconds=(200 - len(rows)) * mean)
    print(f'{run.name:15s} epoch {len(rows):3d}/200  {mean:6.1f} s/epoch  '
          f'training done ~{eta:%Y-%m-%d %H:%M} UTC ({(eta - datetime.now(timezone.utc)).total_seconds() / 3600:.1f} h)')
