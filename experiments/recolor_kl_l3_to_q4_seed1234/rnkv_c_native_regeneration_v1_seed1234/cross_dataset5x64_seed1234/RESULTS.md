# Frozen RN-KV: five new tasks,64 each

| Dataset | Metric | C | RN-KV | Native | RN-KV minus C (pp) |
|---|---|---:|---:|---:|---:|
| 2wiki | em | 9.38% | 21.88% | 32.81% | +12.50 |
| 2wiki | f1 | 22.05% | 35.32% | 43.45% | +13.27 |
| musique | em | 1.56% | 12.50% | 25.00% | +10.94 |
| musique | f1 | 7.64% | 17.85% | 33.07% | +10.21 |
| sciq | accuracy | 98.44% | 100.00% | 100.00% | +1.56 |
| mmlu_pro | accuracy | 25.00% | 26.56% | 53.12% | +1.56 |
| hellaswag | accuracy | 57.81% | 50.00% | 45.31% | -7.81 |

- Frozen original C and final RN-KV epoch4/step1024. Full32 learned memory tokens, four trained Rank16 adapters, construction/answer masks and layers19-30 replacement exactly retained. No retraining, recalibration, architecture changes or test-selected checkpoints.
- Fixed64 per task, seed1234; MMLU-Pro uses near-equal category stratification. Exclude normalized questions/available source IDs in C calibration, RN training/validation and parent/dependency tests. No length, answerability, correctness or support-based selection; stop rather than truncate or silently resample if a selected item exceeds context.
- All original 2Wiki and MuSiQue paragraphs retained, no supporting-fact/decomposition/evidence labels in prompts. SciQ uses its provided support (possibly empty) and all four choices with a fixed per-question seeded option permutation. MMLU-Pro uses every provided option, including items with fewer than10 choices; no chain-of-thought/answer field in the prompt.
- SciQ/MMLU-Pro use candidate LABEL conditional log likelihood with the original ARC answer cue. HellaSwag uses the original short_context prompt and candidate ENDING log likelihood divided by its token count. Scores are computed with the same routine across all3 arms. Native canonical-cache vs official cached/direct-prefill scoring is diagnosed in smoke.
- Generation uses the same Hotpot short-answer prompt and greedy64 limit. 2Wiki source has canonical answer only, so report original-format answer EM/F1, not v1.1 alias scoring. MuSiQue-Ans uses canonical+answer_aliases max EM/F1 and its empty-answer conventions. No supporting-fact/joint metrics or official full-benchmark/leaderboard claim.
- All3 arms share prompt/tokenization/answer cue/metrics; Native and C retain original suffix positions while RN adds32 memory positions as in its trained architecture. Only the explicitly labeled Native baseline may prefill native body. No teacher KV or hidden states enter either student construction path.
- New to RN supervised task mixture, not a guarantee against semantic/document overlap, unrelated historical evaluations or model pretraining. Question-level paired bootstrap intervals and rescued/harmed IDs, n64/task; do not average unlike task metrics into a universal score.
- Detached fail-stop evaluation, pause at question boundaries; other paused experiments remain paused. Smoke predictions may be reused for the exact same fixed test rows after correctness gates, with provenance. No GSM8K, adapter-off sweep, automatic upload or continuous monitoring.
