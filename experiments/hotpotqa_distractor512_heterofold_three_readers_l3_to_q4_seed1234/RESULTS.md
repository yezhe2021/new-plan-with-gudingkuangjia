# HotpotQA distractor512: frozen HeteroFold and Readers

512 dev questions sampled with seed1234; every original paragraph preserved. Shared prompt, greedy generation (64 tokens), answer EM/F1. No training.

| Method | n | EM | F1 | Cap count |
|---|---:|---:|---:|---:|
| qwen_native | 512 | 50.00% | 62.36% | 1 |
| heterofold | 512 | 26.95% | 38.90% | 1 |
| old_kl_reader | 512 | 1.76% | 8.17% | 81 |
| taskce_reader | 512 | 3.32% | 9.89% | 26 |
| hybrid_reader | 512 | 2.54% | 8.86% | 72 |

- Standard HotpotQA distractor dev subset, n512; seeded random sample, all original passages retained, no prompt truncation or gold paragraph selection. Answer EM/F1 only; supporting-fact/joint metrics are not evaluated.
- Existing HeteroFold calibration includes HotpotQA train questions; Readers trained on OBQA/ARC/GSM8K/MMLU-Pro and never on HotpotQA. This is an existing-checkpoint comparison, not equal-domain or equal-budget training.
- Llama3.2-3B-Instruct to Qwen3-4B, FP16 V100. Existing official calibrated HeteroFold map; fixed final epoch4/step1024 Readers with exact frozen grouped Stage-A and residual Stage-B.
- HeteroFold keeps its official bare-text pre-norm Sender capture and native-header cache construction. Our method keeps chat Sender capture, post-k-norm/pre-RoPE mapping and native body prefill/anchor. Same semantic prompt and Qwen continuation suffix, but different existing transfer interfaces; no prefill-free or speed superiority claim.
- Qwen thinking disabled, common greedy decoding capped at64 new tokens. Strip only chat control tokens/outer whitespace before scoring the entire answer; no gold-based or fuzzy answer extraction.
- V100 SDPA uses explicit KV-head repeat for all conditions without changing attention geometry. No finetuning, recalibration, checkpoint selection, or automatic upload.
