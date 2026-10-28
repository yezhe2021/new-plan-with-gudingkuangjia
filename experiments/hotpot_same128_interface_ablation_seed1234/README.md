# Same128 HotpotQA input-interface ablation

Reuse the exact HotpotQA-trained Stage-A best and Stage-B last checkpoints from the 141/512 result. No Reader, training, checkpoint selection, or selector fitting. Randomly select 128 of those original512 using seed1234 without reading outputs, lengths or question types; retain the exact original prepared rows for reproduction. Check disjointness from checkpoint training and validation. Retain all questions and complete support paragraphs, without the earlier bridge-only/short-context filters.

Stages on identical questions:

1. A_original: rerun the original full-context prompt, source capture, translator mapping, cache assembly, and greedy generator. Regenerate native and A+B answers; require all128 cleaned predictions and scores to match stored results. Record any discrepancy and stop before ablations.
2. B_support_only: remove distractor paragraphs; preserve original passage numbers and original instructions; still transfer the complete prompt, including the question.
3. C_selector_prompt: switch to the earlier evidence-selector prompt and document formatting; still transfer the complete prompt, including the question.
4. D_evidence_only: retain the exact C Qwen input token sequence, but transfer only evidence KV and let Qwen process the question. Continue using the original greedy generator.
5. E_diagnostic_decoder: same D inputs and cache; switch to the earlier diagnostic greedy loop, retaining common answer cleaning. Also record the prior diagnostic token-cleaning output for comparison.

Each condition has a native Qwen control. Require C/D native prefill logits to agree within FP16 tolerance on two fixed questions, before ablations. All model parameters are frozen. A runs on all128 first; then B-E run per question and save atomic outputs. Each native/cache object is cloned or reconstructed before reuse. No persistent KV tensors. Reproduction and preflight failures preserve original records and stop the pipeline.

Report EM/F1, generated token counts/cap hits, prediction changes, original full512 and selected128 scores, and paired adjacent-step deltas with 5000 bootstrap resamples of original questions. Effects are conditional on this fixed order, not a full factorial estimate. No automatic retraining, threshold changes, or GitHub upload.

Commands in the attnkv environment on server B: `python prepare.py`; `python run_pipeline.py`. Follow `status.json`, `runner_status.json`, `logs/pipeline.log`; final `RESULTS.md` and `summary.json`. Original result source remains immutable.
