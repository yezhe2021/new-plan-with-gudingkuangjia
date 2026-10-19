#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
exec 9>logs/pipeline.lock
flock -n 9 || { echo 'Pipeline already running'; exit 1; }
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
run() { "$PY" -u run.py "$@"; }
run prepare
if [ ! -f smoke/completed.json ]; then
  run capture gemma smoke
  run capture llama smoke
  run evaluate smoke
fi
run capture gemma formal
run capture llama formal
run evaluate formal
echo 'ALL MIXED KV GENERATION EXPERIMENTS COMPLETED'
