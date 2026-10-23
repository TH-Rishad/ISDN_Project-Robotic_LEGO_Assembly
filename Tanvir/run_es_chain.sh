#!/usr/bin/env bash
# Unattended chain for the paired ES search (WRIGGLE_FIT_RL.md §6.11), all with the pad shift of §6.10:
#   1. ES search (es_skill.py), wall-clock capped
#   2. select the best of the last 4 means, their average and theta = 0 on the validation split (120 episodes each)
#   3. held-out evaluation of the best (scripted_cheap_v2 protocol)
#   4. the same for theta = 0 (the scripted controller with the pad shift), the paired comparison
# Run: setsid nohup bash Tanvir/run_es_chain.sh > Tanvir/results/runs/es_main/chain.log 2>&1 &
set -u
RUN_NAME=${RUN_NAME:-es_main}
HOURS=${HOURS:-1.65}
PY=${PY:-$HOME/.venv/bin/python}
cd "$(dirname "$0")/.."
T=Tanvir
RUN=$T/results/runs/$RUN_NAME
mkdir -p "$RUN"
stamp() { echo "[$(date '+%F %T %Z')] $*"; }
stamp "stage 1: ES search, max-hours $HOURS"
$PY $T/es_skill.py --name "$RUN_NAME" --max-hours "$HOURS" --pairs 8 --sigma-b 0.2 --sigma-w 0.1 --lr 0.05 --zero-every 3 > "$RUN/train.log" 2>&1
stamp "stage 1 done (exit $?)"
stamp "stage 2: select"
$PY $T/es_skill.py --name "$RUN_NAME" --select > "$RUN/select.log" 2>&1
stamp "stage 2 done (exit $?)"
stamp "stage 3: held-out evaluation of best.json"
$PY $T/eval_skill.py --controller "es:$RUN/best.json" --pad-shift --contexts repo family probe --levels 0 2 4 \
    --braces none wriggle --seeds 2 --tag "$RUN_NAME" > "$RUN/eval_best.log" 2>&1
stamp "stage 3 done (exit $?)"
stamp "stage 4: held-out evaluation of theta = 0 (scripted with the pad shift)"
$PY $T/eval_skill.py --controller "es:$RUN/zero.json" --pad-shift --contexts repo family probe --levels 0 2 4 \
    --braces none wriggle --seeds 2 --tag "${RUN_NAME}_zero" > "$RUN/eval_zero.log" 2>&1
stamp "stage 4 done (exit $?)"
stamp "chain finished"
