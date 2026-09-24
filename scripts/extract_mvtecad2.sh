#!/bin/bash
#SBATCH --job-name=dl-mvtecad2
#SBATCH --output=logs/dl-mvtecad2-%j.out
#SBATCH --error=logs/dl-mvtecad2-%j.err
#SBATCH --partition=amd
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00

# Download and extract all eight MVTec AD 2 categories.
# Submit from the repository root: sbatch scripts/extract_mvtecad2.sh

set -Eeuo pipefail

PROJECT_ROOT="${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
DEST="${PROJECT_ROOT}/data/mvtec_ad2"
LOG="${PROJECT_ROOT}/logs/setup-mvtecad2-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-manual}.log"

mkdir -p "${DEST}" "${PROJECT_ROOT}/logs"
exec > >(tee -a "${LOG}") 2>&1
trap 'status=$?; echo "SETUP FAILED: MVTec AD 2 download or extraction failed (exit ${status})"; exit "${status}"' ERR
exec 9>"${PROJECT_ROOT}/logs/.setup-mvtecad2.lock"
flock --nonblock 9 || { echo "Another MVTec AD 2 setup job is already running." >&2; false; }

echo ">>> Destination: ${DEST}"
echo ">>> Disk space:"
df -h "${PROJECT_ROOT}/data"

declare -a CATEGORIES=(can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts)
declare -a URLS=(
    "${MVTECAD2_CAN_URL:-}"
    "${MVTECAD2_FABRIC_URL:-}"
    "${MVTECAD2_FRUIT_JELLY_URL:-}"
    "${MVTECAD2_RICE_URL:-}"
    "${MVTECAD2_SHEET_METAL_URL:-}"
    "${MVTECAD2_VIAL_URL:-}"
    "${MVTECAD2_WALLPLUGS_URL:-}"
    "${MVTECAD2_WALNUTS_URL:-}"
)

for i in "${!CATEGORIES[@]}"; do
    category="${CATEGORIES[$i]}"
    url="${URLS[$i]}"
    archive="${DEST}/.${category}.tar.gz"

    if [[ -d "${DEST}/${category}/train/good" && -d "${DEST}/${category}/test_public" ]]; then
        echo ">>> ${category} is already complete; skipping."
        continue
    fi
    if [[ -e "${DEST}/${category}" ]]; then
        echo "Incomplete category exists at ${DEST}/${category}; move it aside and resubmit." >&2
        false
    fi

    if [[ -z "${url}" ]]; then
        variable="MVTECAD2_${category^^}_URL"
        echo "Missing current URL for ${category}. Set ${variable} from the official MVTec AD 2 page." >&2
        false
    fi

    echo ">>> Downloading ${category}"
    wget --continue --https-only --output-document="${archive}" "${url}"
    tar -tzf "${archive}" >/dev/null

    extract_dir="$(mktemp -d "${DEST}/.extract-${category}.XXXXXX")"
    tar -xzf "${archive}" --no-same-permissions -C "${extract_dir}"
    chmod -R u+w "${extract_dir}"
    if [[ ! -d "${extract_dir}/${category}/train/good" || ! -d "${extract_dir}/${category}/test_public" ]]; then
        echo "Unexpected archive layout for ${category}: ${extract_dir}" >&2
        false
    fi
    mv "${extract_dir}/${category}" "${DEST}/${category}"
    for document in license.txt readme.txt; do
        if [[ -f "${extract_dir}/${document}" && ! -e "${DEST}/${document}" ]]; then
            mv "${extract_dir}/${document}" "${DEST}/${document}"
        elif [[ -f "${extract_dir}/${document}" ]]; then
            rm -f -- "${extract_dir}/${document}"
        fi
    done
    rmdir "${extract_dir}"
    rm -f -- "${archive}"
    echo ">>> ${category} verified."
done

for category in "${CATEGORIES[@]}"; do
    [[ -d "${DEST}/${category}/train/good" ]]
    [[ -d "${DEST}/${category}/test_public/good" ]]
    [[ -d "${DEST}/${category}/test_public/bad" ]]
    [[ -d "${DEST}/${category}/test_public/ground_truth/bad" ]]
done

du -sh "${DEST}"
echo "SETUP SUCCESS"
