#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/gsm8k_generation.log") 2>&1
echo "[$(date '+%F %T')] Llama-to-Qwen standard GSM8K smoke"
"$PY" -u "$ROOT/evaluate_gsm8k_generation.py" llama --limit 2 --max-new-tokens 64
echo "[$(date '+%F %T')] Llama-to-Qwen standard GSM8K study"
"$PY" -u "$ROOT/evaluate_gsm8k_generation.py" llama
echo "[$(date '+%F %T')] Gemma-to-Qwen standard GSM8K smoke"
"$PY" -u "$ROOT/evaluate_gsm8k_generation.py" gemma --limit 2 --max-new-tokens 64
echo "[$(date '+%F %T')] Gemma-to-Qwen standard GSM8K study"
"$PY" -u "$ROOT/evaluate_gsm8k_generation.py" gemma
echo "[$(date '+%F %T')] STANDARD GSM8K GENERATION COMPLETED"
