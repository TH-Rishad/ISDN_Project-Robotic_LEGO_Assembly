#!/bin/bash
# hollow_box with the corrected contact classification (cell/contacts.py):
# the flush-wall SDF contact is no longer double-counted as "penetration", and
# flush corner grazes are permitted neighbour_rub. Outputs to
# Kean/results/m05_collision/hollow_box_contactfix/seed_<n>/ and logs/.
# Submit from the repo root:
#   sbatch --export=ALL,SHAPE=hollow_box Kean/scripts/m05_hollow_box_contactfix.sbatch
#   REPLACE=1 sbatch --array=2 --export=ALL,SHAPE=hollow_box,REPLACE=1 Kean/scripts/m05_hollow_box_contactfix.sbatch
#SBATCH --job-name=m05cf
#SBATCH --partition=gpu-a30
#SBATCH --account=membrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a30:1
#SBATCH --array=0-4
#SBATCH --time=06:00:00
#SBATCH --output=Kean/results/m05_collision/logs/%x_%A_%a.out
#SBATCH --error=Kean/results/m05_collision/logs/%x_%A_%a.err

set -eu
cd "${SLURM_SUBMIT_DIR}"
shape="${SHAPE:?set SHAPE=cube|arch|hollow_box|S3}"
case "${shape}" in
    cube|arch|hollow_box|S3) ;;
    *) echo "unknown shape: ${shape}" >&2; exit 2 ;;
esac
set +u
if [ -f "${HOME}/.bashrc" ]; then
    # shellcheck disable=SC1090
    . "${HOME}/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    conda activate RoboFabEnv
fi
set -u
id="${SLURM_ARRAY_TASK_ID}"
echo "m05 (contact fix): ${shape} seed ${id} on ${CUDA_VISIBLE_DEVICES:-unset}"
out="${SLURM_SUBMIT_DIR}/Kean/results/m05_collision/${shape}_contactfix/seed_${id}"
mkdir -p "${out}"
replace=""
if [ "${REPLACE:-0}" = 1 ]; then
    replace="--replace"
fi
# hollow_box: same --no-strict path as the baseline (m05_collision.sbatch); the
# strict full-path planner tilts the 1-wide walls.
strict=""
if [ "${shape}" = "hollow_box" ]; then
    strict="--no-strict"
fi
bash Kean/scripts/m05.sh --record --experimental-grasp-offset ${strict} --shape "${shape}" --job "${id}" --out "${out}" ${replace}
