# Execution status

- Status: completed.
- Server: A.
- Preflight: passed (`both=61`, `llama_only=12`, `gemma_only=25`, `neither=30`).
- Smoke: passed after fixing probe-time `DynamicCache` mutation by rebuilding a fresh cache for every forward pass.
- Formal audit: completed for all 128 frozen GSM8K test rows.
- Temporary K/V state tensors: removed after successful completion.
- Formal outputs: `results/summary.json` and `results/per_sample.jsonl`.

The pipeline finished with `ALL EXPERIMENTS COMPLETED` at 2026-09-29 11:06:38.
