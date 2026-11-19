# OpenBookQA fact-only frozen KV test

Server B; no training; 64 test examples. Frozen Stage-A/B MCQ weights from `lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/runs/mcq`.

Data: official OpenBookQA V1 Sep2018 archive, `Additional/test_complete.jsonl`. Join first64 local Main test IDs and assert exact equality of question, choices, and answerKey. Sender receives only the annotated `fact1` text (plus BOS and a constant newline). Never use annotator answer/rationale fields as Sender input.

Conditions:

1. `translated_fact_kv`: translate fact-only Llama KV using frozen MCQ Stage-A+Stage-B; Qwen consumes that memory then natively reads Question, full Options, and Answer prefix.
2. `no_fact_kv`: Qwen reads the identical Question/Options/Answer suffix, without any fact text or fact KV.

Use the same fixed native newline token0 in both. All other fact tokens use Full-Sync copy/drop translation. No fact word may enter the native anchor/readout. Receiver positions are contiguous, so adding memory shifts the Question position; this comparison measures the total effect of the fact-memory insertion, not position-controlled semantic benefit alone.

0shot, no chat template, candidate-answer-text likelihood. Report raw and character-length-normalized accuracy, shared/exclusive correct counts, and every prediction. This is a gold-supporting-fact mechanism diagnostic, not closed-book official benchmark evaluation or a retrieval test.

Two-question smoke includes native fact-cache/full-prefill top-choice parity. Then run64. Dataset and checkpoint SHA256 hashes recorded in output config. Predictions and inputs saved in `results/per_sample.json`; final counts in `results/summary.json`.
