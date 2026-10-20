#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
exec 9>logs/pipeline.lock
flock -n 9 || exit 1
PY=/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python
"$PY" -u prepare.py
if [ ! -f smoke/completed.json ]; then
  "$PY" -u capture.py llama smoke
  "$PY" -u capture.py qwen smoke
  "$PY" -u smoke.py
fi
"$PY" -u capture.py llama
"$PY" -u capture.py qwen
"$PY" -u train.py a
"$PY" -u train.py b
"$PY" -u evaluate.py capture
"$PY" -u evaluate.py translate
"$PY" -u evaluate.py score
