#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
exec sh "$HERE/run.sh" "$HERE/p3_env.py" "$@"
