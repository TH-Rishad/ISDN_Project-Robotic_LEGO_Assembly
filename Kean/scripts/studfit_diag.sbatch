#!/bin/bash
# S3 steps 13-15 with the pyramid already placed, so A is free to support.
# Submit from the repo root (the ISDN_studfit worktree):
#   mkdir -p Kean/results/studfit_logs
#   sbatch Kean/scripts/studfit_diag.sbatch
# Full pyramid + S3, no start limits:
#   OUT=Kean/results/dual_pyramid_s3_studfit_007 START_A=0 START_B=0 STEP_LIMIT= \
#     sbatch Kean/scripts/studfit_diag.sbatch
#SBATCH --job-name=studfit
#SBATCH --partition=gpu-a30
#SBATCH --account=membrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a30:1
#SBATCH --time=04:00:00
#SBATCH --output=Kean/results/studfit_logs/%x_%j.out
#SBATCH --error=Kean/results/studfit_logs/%x_%j.err

set -eu
cd "${SLURM_SUBMIT_DIR}"
set +u
if [ -f "${HOME}/.bashrc" ]; then
    # shellcheck disable=SC1090
    . "${HOME}/.bashrc"
fi
set -u

out="${OUT:-Kean/results/dual_studfit_diag_027}"
start_a="${START_A:-11}"
start_b="${START_B:-13}"
step_limit="${STEP_LIMIT-3}"
limit_arg=""
if [ -n "${step_limit}" ]; then
    limit_arg="--step-limit ${step_limit}"
fi
echo "studfit: out=${out} start A=${start_a} B=${start_b} step_limit=${step_limit:-none} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
# dual_build.sh uses Sakib/.venv. Do not activate RoboFabEnv.
bash Kean/scripts/dual_build.sh --build-a pyramid --build-b S3 --stud-fit \
    --viewer null --start-step-a "${start_a}" --start-step-b "${start_b}" \
    ${limit_arg} --after-failure 3 --out "${out}"
