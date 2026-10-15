#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HETEROFOLD_DTYPE=float16
export HETEROFOLD_LLAMA3_PATH=/hy-tmp/yezhe/all_models/models/LLM-Research/Llama-3.2-3B-Instruct
export HETEROFOLD_QWEN4_PATH=/hy-tmp/yezhe/all_models/models/Qwen/Qwen3-4B
export HF_HOME="$PWD/hf_cache"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
exec 9>logs/pipeline.lock
flock -n 9 || { echo 'Pipeline already running'; exit 1; }
case "${1:-pipeline}" in
  smoke) exec .venv/bin/python -u reproduction/gpu_smoke.py ;;
  pipeline) exec .venv/bin/python -u reproduction/pipeline.py ;;
  evaluate|evaluation-smoke)
    export HETEROFOLD_V100_SDPA=1
    export HETEROFOLD_CAPTURE_DEVICE=cpu
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    if [ "$1" = evaluation-smoke ]; then
      exec .venv/bin/python -u reproduction/evaluation_smoke.py
    fi
    exec .venv/bin/python -u reproduction/resume_evaluation.py ;;
  *) echo 'Usage: run_pipeline.sh [smoke|pipeline|evaluate|evaluation-smoke]'; exit 2 ;;
esac
