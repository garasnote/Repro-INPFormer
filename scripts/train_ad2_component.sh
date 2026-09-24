#!/bin/bash
#SBATCH --job-name=ad2-component
#SBATCH --output=logs/ad2-component-%j.out
#SBATCH --error=logs/ad2-component-%j.err
#SBATCH --partition=dev
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=aga
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
base_dir="${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR}}"
container="${base_dir}/containers/inpformer_env.sqfs"

echo "Config=${config}"
echo "RunDir=${run_dir}"
echo "Started=$(date --iso-8601=seconds)"

srun \
    --container-image="${container}" \
    --container-mounts=/shared:/shared \
    --container-workdir="${base_dir}" \
    python scripts/run_ad2_component.py \
        --config "${config}" \
        --run-dir "${run_dir}"

echo "Finished=$(date --iso-8601=seconds)"
