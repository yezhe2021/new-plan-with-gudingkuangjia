#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec 9>pipeline.lock
flock -n 9
for phase in smoke train short; do
    "$1" -u run_m64.py "$phase"
done
"$1" -u eval_long.py
