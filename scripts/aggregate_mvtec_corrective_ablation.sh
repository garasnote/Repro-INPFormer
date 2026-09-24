#!/bin/bash
#SBATCH --job-name=mvtec-inp-aggregate
#SBATCH --output=logs/mvtec-inp-aggregate-%j.out
#SBATCH --error=logs/mvtec-inp-aggregate-%j.err
#SBATCH --partition=frida
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -Eeuo pipefail

BASE_DIR="${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
CORRECTIVE_ARRAY_JOB_ID="${CORRECTIVE_ARRAY_JOB_ID:?Submit with --export=ALL,CORRECTIVE_ARRAY_JOB_ID=<job-id>}"

cd "${BASE_DIR}"
python scripts/generate_ablation_provenance.py \
    --corrective-job-id "${CORRECTIVE_ARRAY_JOB_ID}" \
    --aggregation-job-id "${SLURM_JOB_ID}"

test -s "${BASE_DIR}/ABLATION_RESULTS_PROVENANCE.md"
echo "AGGREGATION SUCCESS: ${BASE_DIR}/ABLATION_RESULTS_PROVENANCE.md"
