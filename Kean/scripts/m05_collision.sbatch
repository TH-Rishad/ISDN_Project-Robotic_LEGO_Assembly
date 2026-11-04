#!/bin/bash
# Kean/results/m05_collision/<shape>/seed_<n>/  and logs/
# One shape per submission: five seeds, one A30 each.
# Submit from the repo root:
#   sbatch --export=ALL,SHAPE=cube Kean/scripts/m05_collision.sbatch
#   sbatch --export=ALL,SHAPE=arch Kean/scripts/m05_collision.sbatch
#   sbatch --export=ALL,SHAPE=hollow_box Kean/scripts/m05_collision.sbatch
#   sbatch --export=ALL,SHAPE=S3 Kean/scripts/m05_collision.sbatch
# Retry one seed: REPLACE=1 sbatch --array=2 --export=ALL,SHAPE=cube,REPLACE=1 Kean/scripts/m05_collision.sbatch
#SBATCH --job-name=m05
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
echo "m05: ${shape} seed ${id} on ${CUDA_VISIBLE_DEVICES:-unset}"
out="${SLURM_SUBMIT_DIR}/Kean/results/m05_collision/${shape}/seed_${id}"
mkdir -p "${out}"
replace=""
if [ "${REPLACE:-0}" = 1 ]; then
    replace="--replace"
fi
# hollow_box has no braced steps, and the strict full-path cuRobo planner tilts
# its 1-wide walls (b_005) that the Cartesian tail seats cleanly. Other shapes
# need the strict collision-checked paths (arm A holds the brace).
strict=""
if [ "${shape}" = "hollow_box" ]; then
    strict="--no-strict"
fi
bash Kean/scripts/m05.sh --record --experimental-grasp-offset ${strict} --shape "${shape}" --job "${id}" --out "${out}" ${replace}
