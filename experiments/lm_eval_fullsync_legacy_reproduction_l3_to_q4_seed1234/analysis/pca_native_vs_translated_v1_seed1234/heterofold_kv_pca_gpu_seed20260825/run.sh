#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HETEROFOLD_DTYPE=float16 HETEROFOLD_V100_SDPA=1
export HF_HOME="$(cd ../.. && pwd)/hf_cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
exec 9>logs/run.lock
flock -n 9 || { echo 'Analysis already running'; exit 1; }
exec /hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u run.py "$@"
