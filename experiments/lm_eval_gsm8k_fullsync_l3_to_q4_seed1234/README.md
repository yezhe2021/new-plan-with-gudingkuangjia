# lm-eval GSM8K Full-Sync pilot

This evaluation-only experiment compares three conditions through the stock
`lm_eval==0.4.12` GSM8K protocol: native Llama3.2-3B, native Qwen3-4B, and frozen
Llama-to-Qwen Full-Sync Stage-A plus the GSM8K Generation Residual64 Stage-B.

Because server A has no external network, `tasks/gsm8k_local.yaml` reproduces
the stock 0.4.12 task verbatim while changing only `dataset_path` to the local
official train/test JSONL files. The correctness pilot additionally overrides
only `num_fewshot=0` and `limit=10`. The custom backend treats the final target
`Answer:` as a Qwen-native suffix; everything before it is the reusable prefix
translated from Llama KV. It implements greedy decoding, task-provided stop
strings, EOS, and `max_gen_toks`, and returns continuation text only. lm-eval
alone performs prompt sampling, answer extraction, filtering, and scoring.
All three conditions use the HF backend's stock 256-token generation budget.
All models explicitly use eager attention because the KV capture/injection
protocol requires that implementation; the native baselines use the same path.

Run with `bash run_pilot.sh`. Results, lm-eval sample logs, and independent
Full-Sync alignment/generation diagnostics are written below `results/`.
If the shared GPU is busy, `run_when_gpu_free.sh` waits until at least 20 GiB
is free and then executes the same pilot once.
