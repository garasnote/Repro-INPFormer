#!/usr/bin/env python3
"""Generate a provenance-first report for the MVTec missing ablation runs."""

from __future__ import annotations

import argparse
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "ABLATION_RESULTS_PROVENANCE.md"
RESULTS_ROOT = REPO_ROOT / "saved_results"
SCHEDULER_LOG_ROOT = REPO_ROOT / "logs"
ORIGINAL_JOB_ID = "142606"
METRICS = ("I-Auroc", "P-AUROC", "P-AUPRO")


@dataclass(frozen=True)
class RunSpec:
    config: str
    seed: int
    loss_lambda: str
    job_id: str
    task_id: int
    log_stem: str
    status: str

    @property
    def run_name(self) -> str:
        return (
            "INP-Former-Multi-Class_dataset=MVTec-AD_"
            "Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_"
            f"INP_num=6_y=0_lambda={self.loss_lambda}_seed={self.seed}_INP"
        )

    @property
    def result_dir(self) -> Path:
        return RESULTS_ROOT / self.run_name

    @property
    def metric_log(self) -> Path:
        return self.result_dir / "log.txt"

    @property
    def checkpoint(self) -> Path:
        return self.result_dir / "model.pth"

    @property
    def stdout_log(self) -> Path:
        return SCHEDULER_LOG_ROOT / f"{self.log_stem}-{self.job_id}_{self.task_id}.out"

    @property
    def stderr_log(self) -> Path:
        return SCHEDULER_LOG_ROOT / f"{self.log_stem}-{self.job_id}_{self.task_id}.err"


def build_runs(corrective_job_id: str) -> tuple[tuple[RunSpec, ...], tuple[RunSpec, ...]]:
    valid = (
        *(RunSpec("INP only", seed, "0.0", ORIGINAL_JOB_ID, seed, "mvtec-inp-abl", "valid existing run") for seed in (1, 2, 3)),
        RunSpec("INP only", 42, "0.0", corrective_job_id, 0, "mvtec-inp-corrective", "corrective rerun"),
        RunSpec("INP only", 123, "0.0", corrective_job_id, 1, "mvtec-inp-corrective", "corrective rerun"),
        *(RunSpec("INP + Lc", seed, "0.2", ORIGINAL_JOB_ID, seed + 5, "mvtec-inp-abl", "valid existing run") for seed in (1, 2, 3)),
        RunSpec("INP + Lc", 42, "0.2", corrective_job_id, 2, "mvtec-inp-corrective", "corrective rerun"),
        RunSpec("INP + Lc", 123, "0.2", corrective_job_id, 3, "mvtec-inp-corrective", "corrective rerun"),
    )
    excluded = (
        RunSpec("INP only", 0, "0.0", ORIGINAL_JOB_ID, 0, "mvtec-inp-abl", "EXCLUDED ACCIDENTAL RUN"),
        RunSpec("INP only", 4, "0.0", ORIGINAL_JOB_ID, 4, "mvtec-inp-abl", "EXCLUDED ACCIDENTAL RUN"),
        RunSpec("INP + Lc", 0, "0.2", ORIGINAL_JOB_ID, 5, "mvtec-inp-abl", "EXCLUDED ACCIDENTAL RUN"),
        RunSpec("INP + Lc", 4, "0.2", ORIGINAL_JOB_ID, 9, "mvtec-inp-abl", "EXCLUDED ACCIDENTAL RUN"),
    )
    return valid, excluded


def parse_metrics(run: RunSpec) -> dict[str, float]:
    if not run.metric_log.is_file():
        raise FileNotFoundError(f"Missing metric log: {run.metric_log}")
    if not run.checkpoint.is_file() or run.checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty checkpoint: {run.checkpoint}")
    if not run.stdout_log.is_file() or not run.stderr_log.is_file():
        raise FileNotFoundError(f"Missing scheduler log for task {run.task_id}")

    text = run.metric_log.read_text(encoding="utf-8")
    if "epoch [200/200]" not in text:
        raise ValueError(f"Run did not record epoch 200: {run.metric_log}")
    category_lines = re.findall(r"^(?!Mean:)[A-Za-z][A-Za-z0-9_ ]*: I-Auroc:", text, re.MULTILINE)
    if len(category_lines) != 15:
        raise ValueError(f"Expected 15 category results in {run.metric_log}; found {len(category_lines)}")
    mean_lines = re.findall(r"^Mean:.*$", text, re.MULTILINE)
    if len(mean_lines) != 1:
        raise ValueError(f"Expected one Mean line in {run.metric_log}; found {len(mean_lines)}")

    values: dict[str, float] = {}
    for metric in METRICS:
        match = re.search(rf"{re.escape(metric)}:([0-9.]+)", mean_lines[0])
        if not match:
            raise ValueError(f"Missing {metric} in {run.metric_log}")
        values[metric] = float(match.group(1)) * 100.0

    stdout = run.stdout_log.read_text(encoding="utf-8")
    expected = f"TRAINING SUCCESS: {'inp_only' if run.config == 'INP only' else 'inp_lc'}, seed={run.seed}"
    if expected not in stdout:
        raise ValueError(f"Missing success marker in {run.stdout_log}")
    return values


def aggregate(rows: list[tuple[RunSpec, dict[str, float]]], config: str) -> dict[str, tuple[float, float]]:
    selected = [metrics for run, metrics in rows if run.config == config]
    if len(selected) != 5:
        raise ValueError(f"Expected five runs for {config}; found {len(selected)}")
    return {
        metric: (
            statistics.mean(row[metric] for row in selected),
            statistics.pstdev(row[metric] for row in selected),
        )
        for metric in METRICS
    }


def code(value: Path | str) -> str:
    return f"`{value}`"


def render_report(
    rows: list[tuple[RunSpec, dict[str, float]]],
    excluded: tuple[RunSpec, ...],
    corrective_job_id: str,
    aggregation_job_id: str,
) -> str:
    generator = Path(__file__).resolve()
    original_launch_script = REPO_ROOT / "scripts" / "train_mvtec_missing_ablation_array.sh"
    corrective_launch_script = REPO_ROOT / "scripts" / "train_mvtec_corrective_ablation_array.sh"
    lines = [
        "# MVTec AD missing-ablation results and provenance",
        "",
        "This file is generated. Update it by running:",
        "",
        f"    python {generator} --corrective-job-id {corrective_job_id} --aggregation-job-id {aggregation_job_id}",
        "",
        "The generator accepts only seeds 1, 2, 3, 42, and 123 for each configuration. It requires an epoch-200 result,",
        "15 category rows, one final Mean row, a nonempty checkpoint, and a scheduler success marker for every run.",
        "Reported standard deviations are population standard deviations (`statistics.pstdev`), matching",
        f"{code(REPO_ROOT / 'aggregate_results.py')}.",
        "",
        "## Official/canonical evaluation outputs",
        "",
        "**HISTORICAL RESULT NOT AVAILABLE LOCALLY**",
        "",
        "No raw official/canonical evaluation output for the original authors' Table 5 is present in this repository.",
        "The values copied into the manuscript are reported values, not locally verifiable evaluation artifacts.",
        "Recover the original Table 5 per-run evaluation output from the authors' experiment archive or rerun the",
        "official released protocol before treating those values as parsed canonical output.",
        "",
        "The following historical reproduction artifacts cited by the manuscript are also absent:",
        "",
        f"- MVTec AD standard baseline result: recover `{RESULTS_ROOT / 'INP-Former-Multi-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6'}/log.txt` and its `model.pth`.",
        f"- Historical No-INP, INP + Lsm, and Full seed results: recover the archived `saved_results` directories plus `{SCHEDULER_LOG_ROOT / 'abl-final-89627.out'}` (job 89627) and the earlier `abl-loss-seed-<job-id>.out` whose job ID is not recorded locally.",
        f"- Strict AU-PRO evaluation: recover `{SCHEDULER_LOG_ROOT / 'aupro-strict-89711.out'}` from job 89711.",
        f"- Inference benchmark: recover `{SCHEDULER_LOG_ROOT / 'infer-bench-88273.out'}` from job 88273.",
        "",
        "Expected historical per-seed paths (all currently absent):",
        "",
        "| Configuration | Seed | Expected metric source | Expected checkpoint | Status |",
        "|---|---:|---|---|---|",
    ]
    historical_configs = (
        ("No INP", 0, "0.0", "noINP"),
        ("INP + Lsm", 3, "0.0", "INP"),
        ("Full", 3, "0.2", "INP"),
    )
    for config, loss_y, loss_lambda, inp_tag in historical_configs:
        for seed in (1, 2, 3, 42, 123):
            run_name = (
                "INP-Former-Multi-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_"
                f"Resize=448_Crop=392_INP_num=6_y={loss_y}_lambda={loss_lambda}_seed={seed}_{inp_tag}"
            )
            result_dir = RESULTS_ROOT / run_name
            lines.append(
                f"| {config} | {seed} | {code(result_dir / 'log.txt')} | "
                f"{code(result_dir / 'model.pth')} | HISTORICAL RESULT NOT AVAILABLE LOCALLY |"
            )

    lines.extend([
        "",
        "These historical numbers are intentionally excluded from the generated metric tables below because their",
        "raw evaluation sources are unavailable.",
        "",
        "## Newly generated ablation outputs",
        "",
        f"Original launch script: {code(original_launch_script)}. Scheduler array: `{ORIGINAL_JOB_ID}`.",
        f"Corrective launch script: {code(corrective_launch_script)}. Scheduler array: `{corrective_job_id}`.",
        f"Dependent aggregation job: `{aggregation_job_id}`.",
        "Protocol encoded by both launch scripts: MVTec AD, INP enabled, Lsm disabled (`y=0`), M=6,",
        "DINOv2-register ViT-Base/14, input/crop 448/392, batch size 16, and 200 epochs.",
        "Final valid seed set: 1, 2, 3, 42, 123. Seeds 42 and 123 are corrective reruns.",
        "",
        "Every metric in the following table is parsed from the final `Mean:` line of the source file named in",
        "the Metric source column.",
        "",
        "| Config | Seed | Job ID | I-AUROC | P-AUROC | P-AUPRO | Result path |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for run, metrics in rows:
        lines.append(
            f"| {run.config} | {run.seed} | {run.job_id} | {metrics['I-Auroc']:.2f} | "
            f"{metrics['P-AUROC']:.2f} | {metrics['P-AUPRO']:.2f} | {code(run.result_dir)} |"
        )

    lines.extend([
        "",
        "Every metric above is parsed from the final `Mean:` line of `<Result path>/log.txt`.",
        "",
        "### Excluded accidental runs",
        "",
        "The following completed outputs remain on disk for provenance but are never passed to aggregation:",
        "",
        "| Config | Seed | Job ID | Status | Result path |",
        "|---|---:|---:|---|---|",
    ])
    for run in excluded:
        lines.append(
            f"| {run.config} | {run.seed} | {run.job_id} | {run.status} | {code(run.result_dir)} |"
        )

    lines.extend([
        "",
        "### Aggregates",
        "",
        "Each aggregate metric is derived from the five per-seed metric source files listed above for that",
        "configuration; no manuscript value is used as an input.",
        "",
        "| Configuration | I-AUROC | P-AUROC | P-AUPRO | Aggregate sources |",
        "|---|---:|---:|---:|---|",
    ])
    for config in ("INP only", "INP + Lc"):
        values = aggregate(rows, config)
        source_paths = [code(run.metric_log) for run, _ in rows if run.config == config]
        formatted = {metric: f"{mean:.2f} ± {std:.2f}" for metric, (mean, std) in values.items()}
        lines.append(
            f"| {config} | {formatted['I-Auroc']} | {formatted['P-AUROC']} | "
            f"{formatted['P-AUPRO']} | {'<br>'.join(source_paths)} |"
        )

    lines.extend([
        "",
        "### Per-seed result, checkpoint, and scheduler-log paths",
        "",
        "| Configuration | Seed | Result/checkpoint directory | Checkpoint | Scheduler stdout | Scheduler stderr |",
        "|---|---:|---|---|---|---|",
    ])
    for run, _ in rows:
        lines.append(
            f"| {run.config} | {run.seed} | {code(run.result_dir)} | {code(run.checkpoint)} | "
            f"{code(run.stdout_log)} | {code(run.stderr_log)} |"
        )

    lines.extend([
        "",
        "## Machine-readable provenance summary",
        "",
        f"MARKDOWN FILE: {OUTPUT_PATH}",
        f"MARKDOWN GENERATOR: {generator}",
        "GENERATOR LINE(S): output/source roots 13-17; per-run path mapping 21-57; valid/excluded run manifests 60-75; validation and metric parsing 78-107; aggregation 110-120; Markdown rendering 127-278; write call 281-292.",
        "OFFICIAL/CANONICAL RESULT SOURCES: HISTORICAL RESULT NOT AVAILABLE LOCALLY; recover original Table 5 outputs and the historical artifacts listed above.",
        "NEW RUN RESULT SOURCES:",
    ])
    lines.extend(f"- {run.metric_log}" for run, _ in rows)
    lines.append("CHECKPOINT DIRECTORIES:")
    lines.extend(f"- {run.result_dir}" for run, _ in rows)
    lines.append("LOG DIRECTORIES:")
    lines.extend(f"- metric log: {run.metric_log}" for run, _ in rows)
    lines.extend(f"- scheduler stdout: {run.stdout_log}" for run, _ in rows)
    lines.extend(f"- scheduler stderr: {run.stderr_log}" for run, _ in rows)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrective-job-id", required=True)
    parser.add_argument("--aggregation-job-id", default="manual")
    args = parser.parse_args()
    valid, excluded = build_runs(args.corrective_job_id)
    rows = [(run, parse_metrics(run)) for run in valid]
    OUTPUT_PATH.write_text(
        render_report(rows, excluded, args.corrective_job_id, args.aggregation_job_id),
        encoding="utf-8",
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
