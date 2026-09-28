#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/gsm8k_generation_quick.log") 2>&1
echo "[$(date '+%F %T')] Llama-to-Qwen GSM8K quick study: 32 rows, Sender Native + Stage-B"
"$PY" -u "$ROOT/evaluate_gsm8k_generation.py" llama --limit 32 --max-new-tokens 384 --conditions sender_native,stage_b
echo "[$(date '+%F %T')] QUICK STANDARD GSM8K GENERATION COMPLETED"
