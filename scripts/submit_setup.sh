#!/bin/bash

# Submit the complete FRIDA setup pipeline from the repository root.
# Existing job IDs may be supplied in the environment to resume a partial submission.

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p data containers logs

submit_job() {
    local raw job_id
    raw="$(sbatch --parsable "$@")"
    job_id="$(printf '%s\n' "${raw}" | awk '
        /^[[:space:]]*[0-9]+(;[^[:space:]]+)?[[:space:]]*$/ {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "")
            sub(/;.*/, "")
            id=$0
        }
        END { print id }
    ')"
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "Could not extract a Slurm job ID from sbatch output:" >&2
        printf '%s\n' "${raw}" >&2
        return 1
    fi
    printf '%s\n' "${job_id}"
}

if [[ -r "containers/inpformer_env.sqfs" ]]; then
    CONTAINER_JOB="${CONTAINER_JOB:-existing}"
    MVTEC_JOB="${MVTEC_JOB:-$(submit_job scripts/download_mvtec.sh)}"
    VISA_JOB="${VISA_JOB:-$(submit_job scripts/download_visa.sh)}"
else
    CONTAINER_JOB="${CONTAINER_JOB:-$(submit_job scripts/setup_inpformer_env.sh)}"
    MVTEC_JOB="${MVTEC_JOB:-$(submit_job --dependency="afterok:${CONTAINER_JOB}" scripts/download_mvtec.sh)}"
    VISA_JOB="${VISA_JOB:-$(submit_job --dependency="afterok:${CONTAINER_JOB}" scripts/download_visa.sh)}"
fi
AD2_JOB="${AD2_JOB:-$(submit_job scripts/extract_mvtecad2.sh)}"
SMOKE_JOB="${SMOKE_JOB:-$(submit_job --dependency="afterok:${MVTEC_JOB}:${VISA_JOB}" scripts/smoke_test.sh)}"

echo "container=${CONTAINER_JOB} mvtec=${MVTEC_JOB} visa=${VISA_JOB} ad2=${AD2_JOB} smoke=${SMOKE_JOB}"
