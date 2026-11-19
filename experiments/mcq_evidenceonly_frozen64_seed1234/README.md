# Frozen MCQ Writer: evidence-only Sender pilot

Server B only. No training. First 32 official test examples from OpenBookQA and ARC-Challenge each (64 total); smoke first one each. Seed 1234.

Frozen checkpoints: `lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/runs/mcq/stage_a/best.pt` and `stage_b/best.pt`. Exact SHA256 hashes are recorded in output config.

Conditions:

- `old_original_protocol`: original official task context and original full-context Stage-A/B transfer; historical reference.
- `query_aware_memory`: Sender `Question -> Options`; translate only Options; Receiver `Options memory -> native Question -> Answer`.
- `evidence_only_memory`: Sender `Options` only; same translated region and same Receiver as query-aware control.
- `native_oracle_memory`: Qwen Options-only native cache, then native Question/Answer.
- `qwen_full_optionsfirst`: full native prefill of the identical Oracle text; cache correctness control.

The primary causal comparison is query-aware versus evidence-only memory. The original protocol differs in additional ways and is not a single-variable control. Native Oracle deliberately uses Options-only Qwen evidence states. Sender never receives gold answers.

Full-Sync copy/drop mapping, all option tokens except native target token0. A token crossing the Options/Question boundary remains native. Post-k_norm/pre-RoPE keys are placed at actual Receiver positions. Native cache versus full-prefill top-choice parity is asserted per question.

Answer-text summed likelihood and character-length-normalized likelihood are both saved. This modified prompt is a mechanism diagnostic, not an official lm-eval benchmark result. Outputs contain every question, choice, gold label, Sender prompt, scores, prediction, per-dataset counts, and paired correctness counts.

Run `python -u run.py --smoke`, then `python -u run.py`. No monitoring or automatic retraining.
