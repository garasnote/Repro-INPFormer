"""
Aggregate results from ablation experiments into tables.

Parses log.txt files from saved_results/ directories.
Outputs: mean±std tables for multi-seed loss ablation,
per-category tables for single runs, LaTeX-formatted output.

Usage:
  python aggregate_results.py                    # all results
  python aggregate_results.py --filter "seed"    # only seed ablation runs
  python aggregate_results.py --latex            # LaTeX table output
"""
import os
import re
import argparse
import numpy as np
from collections import defaultdict


RESULTS_DIR = 'saved_results'

METRICS = ['I-Auroc', 'P-AUROC', 'P-AUPRO']

ABLATION_CONFIGS = {
    'no_inp': {'y': 0, 'lambda': 0.0, 'inp_tag': 'noINP', 'label': 'No INP (backbone only)'},
    'inp_only': {'y': 0, 'lambda': 0.0, 'inp_tag': 'INP', 'label': 'INP only (no losses)'},
    'inp_lc': {'y': 0, 'lambda': 0.2, 'inp_tag': 'INP', 'label': r'INP + $\mathcal{L}_c$'},
    'inp_lsm': {'y': 3, 'lambda': 0.0, 'inp_tag': 'INP', 'label': r'INP + $\mathcal{L}_{sm}$'},
    'full': {'y': 3, 'lambda': 0.2, 'inp_tag': 'INP', 'label': r'INP + $\mathcal{L}_{sm}$ + $\mathcal{L}_c$ (Full)'},
}


def parse_log(log_path):
    """Parse a log.txt file, return dict of {category: {metric: value}} and mean."""
    results = {}
    mean = {}
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "category: I-Auroc:0.9988, I-AP:0.9996, ..., P-AUPRO:0.9789"
            # or "Mean: I-Auroc:0.9967, ..."
            match = re.match(r'^(\w[\w\s]*?):\s*I-Auroc:([\d.]+),\s*I-AP:([\d.]+),\s*I-F1:([\d.]+),\s*P-AUROC:([\d.]+),\s*P-AP:([\d.]+),\s*P-F1:([\d.]+),\s*P-AUPRO:([\d.]+)', line)
            if match:
                name = match.group(1).strip()
                vals = {
                    'I-Auroc': float(match.group(2)),
                    'I-AP': float(match.group(3)),
                    'I-F1': float(match.group(4)),
                    'P-AUROC': float(match.group(5)),
                    'P-AP': float(match.group(6)),
                    'P-F1': float(match.group(7)),
                    'P-AUPRO': float(match.group(8)),
                }
                if name == 'Mean':
                    mean = vals
                else:
                    results[name] = vals
    return results, mean


def parse_dir_name(dirname):
    """Extract config from directory name."""
    info = {}
    m = re.search(r'dataset=([^_]+)', dirname)
    if m: info['dataset'] = m.group(1)
    m = re.search(r'Encoder=([^_]+_vit_[^_]+_\d+)', dirname)
    if m: info['encoder'] = m.group(1)
    m = re.search(r'Resize=(\d+)', dirname)
    if m: info['resize'] = int(m.group(1))
    m = re.search(r'Crop=(\d+)', dirname)
    if m: info['crop'] = int(m.group(1))
    m = re.search(r'INP_num=(\d+)', dirname)
    if m: info['inp_num'] = int(m.group(1))
    m = re.search(r'y=(\d+)', dirname)
    if m: info['y'] = int(m.group(1))
    m = re.search(r'lambda=([\d.]+)', dirname)
    if m: info['lambda'] = float(m.group(1))
    m = re.search(r'seed=(\d+)', dirname)
    if m: info['seed'] = int(m.group(1))
    m = re.search(r'(noINP|INP)$', dirname)
    if m: info['inp_tag'] = m.group(1)
    return info


def identify_config(info):
    """Map parsed info to ablation config name."""
    for name, cfg in ABLATION_CONFIGS.items():
        if (info.get('y') == cfg['y'] and
            info.get('lambda') == cfg['lambda'] and
            info.get('inp_tag') == cfg['inp_tag']):
            return name
    return None


def gather_all_results():
    """Scan saved_results/ and return structured data."""
    all_results = []
    for dirname in sorted(os.listdir(RESULTS_DIR)):
        log_path = os.path.join(RESULTS_DIR, dirname, 'log.txt')
        if not os.path.isfile(log_path):
            continue
        info = parse_dir_name(dirname)
        per_cat, mean = parse_log(log_path)
        if not mean:
            continue
        all_results.append({
            'dir': dirname,
            'info': info,
            'per_category': per_cat,
            'mean': mean,
        })
    return all_results


def print_seed_ablation(all_results, dataset, latex=False):
    """Print mean±std table for multi-seed loss ablation."""
    grouped = defaultdict(list)
    for r in all_results:
        info = r['info']
        if info.get('dataset') != dataset:
            continue
        if 'seed' not in info:
            continue
        cfg_name = identify_config(info)
        if cfg_name:
            grouped[cfg_name].append(r['mean'])

    if not grouped:
        return

    print(f"\n{'='*80}")
    print(f"LOSS ABLATION — {dataset} (mean ± std over seeds)")
    print(f"{'='*80}")

    if latex:
        print(r"\begin{tabular}{l" + "c" * len(METRICS) + "}")
        print(r"\toprule")
        print("Config & " + " & ".join(METRICS) + r" \\")
        print(r"\midrule")
    else:
        header = f"  {'Config':<30s}"
        for m in METRICS:
            header += f" | {m:>14s}"
        header += f" | {'N':>3s}"
        print(header)
        print("-" * 80)

    config_order = ['no_inp', 'inp_only', 'inp_lc', 'inp_lsm', 'full']
    for cfg_name in config_order:
        if cfg_name not in grouped:
            continue
        means_list = grouped[cfg_name]
        n = len(means_list)
        label = ABLATION_CONFIGS[cfg_name]['label']

        if latex:
            row = f"{label}"
            for m in METRICS:
                vals = [d[m] for d in means_list]
                mu, std = np.mean(vals) * 100, np.std(vals) * 100
                if cfg_name == 'full':
                    row += f" & \\textbf{{{mu:.2f}}}$\\pm${std:.2f}"
                else:
                    row += f" & {mu:.2f}$\\pm${std:.2f}"
            row += r" \\"
            print(row)
        else:
            row = f"  {label:<30s}"
            for m in METRICS:
                vals = [d[m] for d in means_list]
                mu, std = np.mean(vals) * 100, np.std(vals) * 100
                row += f" | {mu:6.2f}±{std:4.2f}  "
            row += f" | {n:3d}"
            print(row)

    if latex:
        print(r"\bottomrule")
        print(r"\end{tabular}")


def print_single_run_table(all_results, dataset, latex=False):
    """Print per-category table for standard runs (no seed in name)."""
    standard = [r for r in all_results
                if r['info'].get('dataset') == dataset
                and 'seed' not in r['info']
                and r['info'].get('inp_tag') != 'noINP']

    if not standard:
        return

    for r in standard:
        info = r['info']
        print(f"\n{'='*80}")
        enc = info.get('encoder', '?')
        inp = info.get('inp_num', '?')
        res = f"{info.get('resize', '?')}/{info.get('crop', '?')}"
        print(f"{dataset} | {enc} | M={inp} | {res}")
        print(f"{'='*80}")

        if latex:
            cats = sorted(r['per_category'].keys())
            print(r"\begin{tabular}{l" + "c" * len(METRICS) + "}")
            print(r"\toprule")
            print("Category & " + " & ".join(METRICS) + r" \\")
            print(r"\midrule")
            for cat in cats:
                vals = r['per_category'][cat]
                row = f"{cat}"
                for m in METRICS:
                    row += f" & {vals[m]*100:.2f}"
                row += r" \\"
                print(row)
            print(r"\midrule")
            row = "\\textbf{Mean}"
            for m in METRICS:
                row += f" & \\textbf{{{r['mean'][m]*100:.2f}}}"
            row += r" \\"
            print(row)
            print(r"\bottomrule")
            print(r"\end{tabular}")
        else:
            cats = sorted(r['per_category'].keys())
            header = f"  {'Category':<16s}"
            for m in METRICS:
                header += f" | {m:>8s}"
            print(header)
            print("-" * 80)
            for cat in cats:
                vals = r['per_category'][cat]
                row = f"  {cat:<16s}"
                for m in METRICS:
                    row += f" | {vals[m]*100:8.2f}"
                print(row)
            print("-" * 80)
            row = f"  {'MEAN':<16s}"
            for m in METRICS:
                row += f" | {r['mean'][m]*100:8.2f}"
            print(row)


def print_inp_count_ablation(all_results, dataset, latex=False):
    """Print INP count (M) ablation results."""
    runs = [r for r in all_results
            if r['info'].get('dataset') == dataset
            and r['info'].get('y') == 3
            and r['info'].get('lambda') == 0.2
            and r['info'].get('inp_tag') == 'INP'
            and 'seed' not in r['info']]

    if not runs:
        return

    by_m = {}
    for r in runs:
        m = r['info'].get('inp_num')
        if m:
            by_m[m] = r['mean']

    if not by_m:
        return

    print(f"\n{'='*80}")
    print(f"INP COUNT ABLATION — {dataset}")
    print(f"{'='*80}")

    if latex:
        ms = sorted(by_m.keys())
        print(r"\begin{tabular}{l" + "c" * len(ms) + "}")
        print(r"\toprule")
        print("Metric & " + " & ".join(f"M={m}" for m in ms) + r" \\")
        print(r"\midrule")
        for metric in METRICS:
            row = metric
            for m in ms:
                row += f" & {by_m[m][metric]*100:.2f}"
            row += r" \\"
            print(row)
        print(r"\bottomrule")
        print(r"\end{tabular}")
    else:
        header = f"  {'M':>4s}"
        for m in METRICS:
            header += f" | {m:>8s}"
        print(header)
        print("-" * 60)
        for m_val in sorted(by_m.keys()):
            row = f"  {m_val:4d}"
            for metric in METRICS:
                row += f" | {by_m[m_val][metric]*100:8.2f}"
            print(row)


def main():
    parser = argparse.ArgumentParser(description='Aggregate INP-Former results')
    parser.add_argument('--latex', action='store_true', help='Output LaTeX tables')
    parser.add_argument('--filter', type=str, default='', help='Filter dirs by substring')
    parser.add_argument('--dataset', type=str, default='', help='Filter by dataset (MVTec-AD, VisA, Real-IAD)')
    args = parser.parse_args()

    all_results = gather_all_results()

    if args.filter:
        all_results = [r for r in all_results if args.filter in r['dir']]

    print(f"Found {len(all_results)} result directories")

    datasets = [args.dataset] if args.dataset else ['MVTec-AD', 'VisA', 'Real-IAD', 'MVTec-AD2']
    datasets = [d for d in datasets if any(r['info'].get('dataset') == d for r in all_results)]

    for dataset in datasets:
        print_seed_ablation(all_results, dataset, latex=args.latex)
        print_inp_count_ablation(all_results, dataset, latex=args.latex)
        print_single_run_table(all_results, dataset, latex=args.latex)

    print("\nDone.")


if __name__ == '__main__':
    main()
