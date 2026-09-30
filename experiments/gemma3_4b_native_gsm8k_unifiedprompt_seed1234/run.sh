#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
cd "$ROOT"
mkdir -p logs results
"$PY" -u native_eval.py
