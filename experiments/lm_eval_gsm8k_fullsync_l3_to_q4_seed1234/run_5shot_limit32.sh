#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
TAG=5shot_limit32
LM_ARGS=(
  --include_path "$ROOT/tasks"
  --tasks gsm8k_local
  --num_fewshot 5
  --limit 32
  --batch_size 1
  --device cuda:0
  --log_samples
  --seed 1234
)

mkdir -p "$ROOT/results" "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/${TAG}.log") 2>&1

echo "[$(date '+%F %T')] START llama_native ${TAG}"
"$PY" -m lm_eval "${LM_ARGS[@]}" --model hf \
  --model_args "pretrained=/home/yezhe/all_models/models/LLM-Research/Llama-3___2-3B-Instruct,dtype=float16,attn_implementation=eager" \
  --output_path "$ROOT/results/llama_native_eager_${TAG}"

echo "[$(date '+%F %T')] START qwen_native_hf ${TAG}"
"$PY" -m lm_eval "${LM_ARGS[@]}" --model hf \
  --model_args "pretrained=/home/yezhe/all_models/models/Qwen/Qwen3-4B,dtype=float16,attn_implementation=eager" \
  --output_path "$ROOT/results/qwen_native_hf_eager_${TAG}"

echo "[$(date '+%F %T')] START fullsync_genb ${TAG}"
"$PY" "$ROOT/run_lm_eval.py" "${LM_ARGS[@]}" --model fullsync_l3_to_q4 \
  --model_args "mode=fullsync,max_gen_toks=256,max_length=32768,audit_path=$ROOT/results/fullsync_genb_eager_${TAG}_audit.jsonl" \
  --output_path "$ROOT/results/fullsync_genb_eager_${TAG}"

echo "[$(date '+%F %T')] GSM8K STANDARD LM-EVAL ${TAG} COMPLETED"
