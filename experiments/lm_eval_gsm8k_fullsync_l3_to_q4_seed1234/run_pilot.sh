#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
LM_ARGS=(--include_path "$ROOT/tasks" --tasks gsm8k_local --num_fewshot 0 --limit 10 --batch_size 1 --device cuda:0 --log_samples --seed 1234)
mkdir -p "$ROOT/results" "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/pilot.log") 2>&1

"$PY" "$ROOT/tests.py"
"$PY" -m lm_eval "${LM_ARGS[@]}" --model hf \
  --model_args "pretrained=/home/yezhe/all_models/models/LLM-Research/Llama-3___2-3B-Instruct,dtype=float16,attn_implementation=eager" \
  --output_path "$ROOT/results/llama_native_eager_0shot_limit10"
"$PY" -m lm_eval "${LM_ARGS[@]}" --model hf \
  --model_args "pretrained=/home/yezhe/all_models/models/Qwen/Qwen3-4B,dtype=float16,attn_implementation=eager" \
  --output_path "$ROOT/results/qwen_native_hf_eager_0shot_limit10"
"$PY" "$ROOT/run_lm_eval.py" "${LM_ARGS[@]}" --model fullsync_l3_to_q4 \
  --model_args "mode=native_qwen,max_gen_toks=256,max_length=32768,audit_path=$ROOT/results/qwen_native_custom_eager_audit.jsonl" \
  --output_path "$ROOT/results/qwen_native_custom_eager_0shot_limit10"
"$PY" "$ROOT/compare_native.py"
"$PY" "$ROOT/run_lm_eval.py" "${LM_ARGS[@]}" --model fullsync_l3_to_q4 \
  --model_args "mode=fullsync,max_gen_toks=256,max_length=32768,audit_path=$ROOT/results/fullsync_backend_audit.jsonl" \
  --output_path "$ROOT/results/fullsync_genb_eager_0shot_limit10"
echo "[$(date '+%F %T')] LM-EVAL CORRECTNESS PILOT COMPLETED"
