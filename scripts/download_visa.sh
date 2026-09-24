#!/bin/bash
#SBATCH --job-name=dl-visa
#SBATCH --output=logs/dl-visa-%j.out
#SBATCH --error=logs/dl-visa-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00

# Download VisA and apply Amazon's official one-class split.
# Submit from the repository root after the container job succeeds.

set -Eeuo pipefail

source "${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}/scripts/env.sh"
PROJECT_ROOT="${INPFORMER_ROOT}"
LOG="${PROJECT_ROOT}/logs/setup-visa-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-manual}.log"

mkdir -p "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/logs"
exec > >(tee -a "${LOG}") 2>&1
trap 'status=$?; echo "SETUP FAILED: VisA download or preparation failed (exit ${status})"; exit "${status}"' ERR
exec 9>"${PROJECT_ROOT}/logs/.setup-visa.lock"
flock --nonblock 9 || { echo "Another VisA setup job is already running." >&2; false; }

export NVIDIA_VISIBLE_DEVICES=void
inp_run "${PROJECT_ROOT}" \
    bash -c '
        set -Eeuo pipefail

        dataset_url="https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
        expected_sha256="2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"
        cache_dir="data/.cache/visa"
        work_dir="data/.visa_work"
        output="data/visa"
        partial="data/visa.partial"
        archive="${cache_dir}/VisA_20220922.tar"
        tools_dir="${cache_dir}/spot-diff"

        if [[ -e "${output}" ]]; then
            echo "Refusing to overwrite existing dataset: ${output}" >&2
            false
        fi

        mkdir -p "${cache_dir}" "${work_dir}" "${partial}"
        wget --continue --https-only --output-document="${archive}" "${dataset_url}"
        echo "${expected_sha256}  ${archive}" | sha256sum --check --strict -

        if [[ ! -d "${tools_dir}/.git" ]]; then
            git clone --depth 1 https://github.com/amazon-science/spot-diff.git "${tools_dir}"
        fi

        if [[ ! -d "${work_dir}/candle" ]]; then
            tar -tf "${archive}" >/dev/null
            tar -xf "${archive}" -C "${work_dir}"
        fi

        python "${tools_dir}/utils/prepare_data.py" \
            --split-type 1cls \
            --data-folder "${work_dir}" \
            --save-folder "${partial}" \
            --split-file "${tools_dir}/split_csv/1cls.csv"

        categories=(candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum)
        for category in "${categories[@]}"; do
            [[ -d "${partial}/1cls/${category}/train/good" ]]
            [[ -d "${partial}/1cls/${category}/test/good" ]]
            [[ -d "${partial}/1cls/${category}/test/bad" ]]
            [[ -d "${partial}/1cls/${category}/ground_truth/bad" ]]
        done

        mv "${partial}" "${output}"
        rm -rf -- "${work_dir}" "${cache_dir}"
    '

[[ -d "${PROJECT_ROOT}/data/visa/1cls/candle/train/good" ]]
echo "SETUP SUCCESS"
