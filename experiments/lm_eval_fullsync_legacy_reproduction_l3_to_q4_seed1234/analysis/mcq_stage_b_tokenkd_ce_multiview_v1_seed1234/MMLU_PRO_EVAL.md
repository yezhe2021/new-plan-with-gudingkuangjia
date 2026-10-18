# MMLU-Pro follow-up of scheme 3

Evaluate the existing validation-selected epoch-4 Stage B with the unchanged frozen MCQ Stage A on server B. No further training or new module is added. The original training scripts, manifest and completed MCQ/MMLU results are preserved.

Reuse exactly the same 128 original MMLU-Pro test questions across 14 subjects, the same 5 official validation demonstrations, raw completion prompts without a chat template, greedy decoding, 2048 new-token budget, `Question:` stop, and installed lm-eval 0.4.12's `custom-extract` / exact-match scoring. Original Qwen/Llama native, Stage A and old Stage B results are reused for comparison. Do not mix this run with the earlier PCA-compression MMLU-Pro experiment.

`evaluate_mmlu_pro.py prepare` verifies every original question, prompt and generation argument. `smoke` tests the longest prompt with an explicitly separate 8-token generation budget. `run` evaluates the full original 2048-token protocol. Each exact-input greedy generation is saved immediately and can be replayed only with the same checkpoint, manifest and generation arguments. Official harness generation, stop handling, filters and aggregate scoring remain in control. At batch size 1, processing one request at a time adds progress logging and persistence without changing the request contents.

Outputs: `evaluation/mmlu_pro/manifest.json`, `smoke.json`, `raw_generations.jsonl`, official per-subject sample logs, `new_stage_b.json`, `summary.json`, and `status.json`. Log: `logs/mmlu_pro.log`. Both the fixed prompt manifest and checkpoint hashes are retained. No model weights or KV tensors are copied into evaluation outputs.
