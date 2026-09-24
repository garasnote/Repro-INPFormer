#!/usr/bin/env bash
# Stage and validate one AD2 private submission candidate, then run the
# unmodified official checker/compressor. This script never uploads anything.
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ad2-private-package-%x-%j.out
#SBATCH --error=logs/ad2-private-package-%x-%j.err

set -Eeuo pipefail

if [[ $# -ne 1 || ! $1 =~ ^(1|6)$ ]]; then
    echo "Usage: sbatch --job-name=ad2-private-package-mN $0 {1|6}" >&2
    exit 2
fi

m=$1
source "${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}/scripts/env.sh"
repo="${INPFORMER_ROOT}"
run_root="${repo}/evaluation_results/ad2-private-submission-v1/M${m}"
candidate="${run_root}/submission_m${m}"

inp_run "${repo}" \
    python scripts/prepare_ad2_server_submission.py \
        --manifests \
            "${run_root}/test_private/manifest.json" \
            "${run_root}/test_private_mixed/manifest.json" \
        --output "${candidate}"

cd "${run_root}"
inp_run "${run_root}" \
    python "${repo}/third_party/mvtec/MVTecAD2_public_code_utils/check_and_prepare_data_for_upload.py" \
        "submission_m${m}"
