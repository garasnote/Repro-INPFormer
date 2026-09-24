#!/bin/bash
#SBATCH --job-name=ad2-component
#SBATCH --output=logs/ad2-component-%j.out
#SBATCH --error=logs/ad2-component-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: sbatch $0 <config> <run-dir>" >&2
    exit 2
fi

config="$1"
run_dir="$2"
source "${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}/scripts/env.sh"
base_dir="${INPFORMER_ROOT}"

echo "Config=${config}"
echo "RunDir=${run_dir}"
echo "Started=$(date --iso-8601=seconds)"

inp_run "${base_dir}" \
    python scripts/run_ad2_component.py \
        --config "${config}" \
        --run-dir "${run_dir}"

echo "Finished=$(date --iso-8601=seconds)"
