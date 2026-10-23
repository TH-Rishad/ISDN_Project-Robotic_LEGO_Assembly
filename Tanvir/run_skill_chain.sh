#!/usr/bin/env bash
# Unattended chain for the skill-level wriggle policy (WRIGGLE_FIT_RL.md §6.9):
#   1. train until TRAIN_END (wall clock, default 07:45 today), checkpoint every 15 min
#   2. select the best checkpoint on the family validation split (5 contexts per class, levels 0 and 2,
#      braces none + wriggle = 120 episodes per checkpoint; ScriptedWriggle scored on the same episodes)
#   3. held-out evaluation of the selected checkpoint (repo + family test + probe, levels 0 2 4, braces none + wriggle, 2 seeds)
#   4. the same for checkpoint 0 (theta = 0, the scripted controller), as the comparison
# Run: setsid nohup bash Tanvir/run_skill_chain.sh > Tanvir/results/runs/skill_main/chain.log 2>&1 &
set -u
NAME=${RUN_NAME:-skill_main}      # not $NAME: WSL sets that to the host name
TRAIN_END=${TRAIN_END:-07:45}
PY=${PY:-$HOME/.venv/bin/python}            # the repo's python (the PATH one is an empty venv on this machine)
cd "$(dirname "$0")/.."
T=Tanvir
RUN=$T/results/runs/$NAME
mkdir -p "$RUN"
stamp() { echo "[$(date '+%F %T %Z')] $*"; }

HOURS=$(awk -v e="$(date -d "$TRAIN_END" +%s)" -v n="$(date +%s)" 'BEGIN{h=(e-n)/3600; if (h<0.25) h=0.25; printf "%.3f", h}')
stamp "stage 1: train, max-hours $HOURS (until about $TRAIN_END)"
$PY $T/train_skill.py --name "$NAME" --workers 7 --per-worker 3 --horizon 50 --max-hours "$HOURS" --ckpt-minutes 15 \
    --init-std 0.4 --lr 3e-4 --gamma 0.97 --shaping 0.5 --critic-warmup 5000 --unlock-gate 0.4 --uniform-frac 0.1 \
    > "$RUN/train.log" 2>&1
stamp "stage 1 done (exit $?)"

stamp "stage 2: select the best checkpoint"
$PY $T/eval_skill.py --select "$NAME" --per-class 5 > "$RUN/select.log" 2>&1
stamp "stage 2 done (exit $?)"

stamp "stage 3: held-out evaluation of best.pt"
$PY $T/eval_skill.py --controller "skill:$RUN/best.pt" --contexts repo family probe --levels 0 2 4 \
    --braces none wriggle --seeds 2 --tag "skill_${NAME#skill_}" > "$RUN/eval_best.log" 2>&1
stamp "stage 3 done (exit $?)"

stamp "stage 4: held-out evaluation of checkpoint 0 (theta = 0)"
$PY $T/eval_skill.py --controller "skill:$RUN/ckpt_00000000.pt" --contexts repo family probe --levels 0 2 4 \
    --braces none wriggle --seeds 2 --tag "skill_${NAME#skill_}_ckpt0" > "$RUN/eval_ckpt0.log" 2>&1
stamp "stage 4 done (exit $?)"
stamp "chain finished"
