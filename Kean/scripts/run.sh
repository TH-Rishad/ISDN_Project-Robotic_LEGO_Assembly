#!/bin/sh
# Thin entry point: Sakib owns interpreter routing and environment provisioning.
set -eu
HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
SAKIB=$(CDPATH= cd -- "$HERE/../../Sakib" && pwd)
PY="$SAKIB/.venv/bin/python"
[ -x "$PY" ] || {
    echo "kean: Sakib/.venv is unavailable; ask Sakib to provision the shared environment." >&2
    exit 1
}
[ "$#" -gt 0 ] || { echo "usage: bash Kean/scripts/run.sh SCRIPT [ARGS...]" >&2; exit 2; }
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export ROBOFAB_PY="$PY"
exec bash "$SAKIB/brickassembly/scripts/run.sh" "$@"
