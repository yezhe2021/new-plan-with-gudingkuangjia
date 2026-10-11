#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>.pipeline.lock
flock -n 9 || { echo 'Analysis already active'; exit 2; }
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
export MPLCONFIGDIR="$PWD/.mplconfig"
if [[ ! -f manifest.json ]]; then "$PY" -u experiment.py prepare; fi
if [[ ! -f smoke/completed.json ]]; then "$PY" -u experiment.py smoke; fi
if [[ ! -f results/completed.json ]]; then "$PY" -u experiment.py run; fi
