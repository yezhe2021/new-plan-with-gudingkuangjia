#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HETEROFOLD_FP32_COMPAT=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
PY=/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python
date --iso-8601=seconds
printf '%s\n' 'START existing-data HeteroFold reproduction; explicit V100 FP32 compatibility'
"$PY" -u tests.py
"$PY" -u prepare_existing.py
"$PY" -u run_pipeline.py --mode formal --evaluate hub
