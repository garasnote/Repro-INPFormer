# Shared helpers for the job scripts in this directory. Source it, don't run it.
#
# Resolves the repository root and provides `inp_run WORKDIR CMD...`, which runs
# CMD inside the project Enroot container when the cluster supports it (Slurm +
# Pyxis) and the image exists, and in the current Python environment otherwise.
#
# Environment overrides:
#   INPFORMER_ROOT              repository root (default: sbatch submit dir, else this repo)
#   INPFORMER_CONTAINER         container image, or "none" to always run natively
#                               (default: $INPFORMER_ROOT/containers/inpformer_env.sqfs)
#   INPFORMER_CONTAINER_MOUNTS  extra comma-separated Pyxis mounts, e.g. /data:/data

# sbatch runs a spooled copy of the script, so BASH_SOURCE only helps outside Slurm.
INPFORMER_ROOT="${INPFORMER_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [[ ! -f "${INPFORMER_ROOT}/INP_Former_Multi_Class.py" ]]; then
    echo "INPFORMER_ROOT=${INPFORMER_ROOT} is not the repository root." >&2
    echo "Submit from the repository root or set INPFORMER_ROOT." >&2
    exit 2
fi
export INPFORMER_ROOT
INPFORMER_CONTAINER="${INPFORMER_CONTAINER:-${INPFORMER_ROOT}/containers/inpformer_env.sqfs}"
mkdir -p "${INPFORMER_ROOT}/logs"

inp_has_pyxis() {
    command -v srun >/dev/null 2>&1 && srun --help 2>/dev/null | grep -q -- '--container-image'
}

inp_use_container() {
    [[ "${INPFORMER_CONTAINER}" != none && -r "${INPFORMER_CONTAINER}" ]] && inp_has_pyxis
}

inp_run() {
    local workdir="$1"
    shift
    if inp_use_container; then
        local mounts="${INPFORMER_ROOT}:${INPFORMER_ROOT}"
        [[ "${workdir}" != "${INPFORMER_ROOT}"* ]] && mounts+=",${workdir}:${workdir}"
        [[ -n "${INPFORMER_CONTAINER_MOUNTS:-}" ]] && mounts+=",${INPFORMER_CONTAINER_MOUNTS}"
        srun --container-image="${INPFORMER_CONTAINER}" \
            --container-mounts="${mounts}" \
            --container-workdir="${workdir}" \
            "$@"
    else
        (cd "${workdir}" && "$@")
    fi
}
