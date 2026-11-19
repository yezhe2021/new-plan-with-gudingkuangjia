# Gemma native baselines: remaining five datasets

Authorized 2026-10-24. Complete the Gemma3-4B-IT native results on the exact historical OBQA128, ARC128, HellaSwag64, MMLU-Pro128, SciQ64 test subsets. Reuse the completed GSM8K32 Gemma native results, pinned by file hashes. No training and no changes to the completed translator/Reader experiment.

Gemma uses FP32 on V100 because the earlier FP16 audit produced nonfinite states. Each input uses the existing exact raw task prompt, Gemma's native user/assistant template, and the existing answer cue. No gold labels are appended. Candidates use the same standalone tokenization policy as the transfer experiment. Four MCQ datasets sum conditional label token log probabilities; HellaSwag averages continuation token log probabilities. These are native likelihood evaluations, not answer-generation accuracy.

Before evaluation, two real examples (label task + multi-token HellaSwag) compare the candidate-scoring optimization against independent full-sequence/full-logit reference scores. A mismatch stops execution. Every record is written atomically and linked to the immutable manifest. `summary.json` consolidates all six tasks and four conditions: Qwen Native, Gemma Native, Gemma→Qwen A+B, Gemma→Qwen A+B+Reader. `comparison_per_sample.jsonl` contains 544 aligned comparisons.

MMLU-Pro remains the pre-existing custom split from official test; the numbers are not leaderboard scores. Different model-native tokenizers can yield different HellaSwag token counts, so normalized scoring follows the same policy rather than imposing the Qwen tokenizer on Gemma.

The worker is detached from SSH. Failure writes failure.json and stops; no blind automatic retries or uploads.
