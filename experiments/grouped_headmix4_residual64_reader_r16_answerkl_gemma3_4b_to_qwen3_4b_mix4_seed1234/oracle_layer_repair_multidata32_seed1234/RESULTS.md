# Gemma to Qwen Oracle layer replacement, five tasks

Existing frozen Gemma Stage-A+B, no Reader/training/GSM8K. Same160 questions and receiver protocol as previous Llama Oracle.

| Arm | openbookqa | arc_challenge | hellaswag | mmlu_pro | sciq | Macro |
|---|---:|---:|---:|---:|---:|---:|
| translated | 59.38% | 43.75% | 31.25% | 43.75% | 87.50% | 53.12% |
| native | 68.75% | 84.38% | 31.25% | 46.88% | 96.88% | 65.62% |
| reject_all | 9.38% | 21.88% | 25.00% | 6.25% | 21.88% | 16.88% |
| g1_K | 62.50% | 43.75% | 28.12% | 37.50% | 87.50% | 51.88% |
| g1_V | 53.12% | 40.62% | 34.38% | 34.38% | 84.38% | 49.38% |
| g1_KV | 53.12% | 43.75% | 40.62% | 34.38% | 84.38% | 51.25% |
| g1_mask | 53.12% | 43.75% | 25.00% | 34.38% | 87.50% | 48.75% |
| g2_K | 56.25% | 46.88% | 37.50% | 37.50% | 84.38% | 52.50% |
| g2_V | 56.25% | 46.88% | 31.25% | 46.88% | 87.50% | 53.75% |
| g2_KV | 56.25% | 46.88% | 34.38% | 43.75% | 81.25% | 52.50% |
| g2_mask | 59.38% | 46.88% | 31.25% | 40.62% | 81.25% | 51.88% |
| g3_K | 59.38% | 46.88% | 31.25% | 43.75% | 84.38% | 53.12% |
| g3_V | 59.38% | 43.75% | 34.38% | 40.62% | 87.50% | 53.12% |
| g3_KV | 56.25% | 43.75% | 34.38% | 37.50% | 84.38% | 51.25% |
| g3_mask | 46.88% | 40.62% | 31.25% | 40.62% | 78.12% | 47.50% |
| g4_K | 59.38% | 50.00% | 34.38% | 40.62% | 96.88% | 56.25% |
| g4_V | 65.62% | 46.88% | 31.25% | 43.75% | 90.62% | 55.62% |
| g4_KV | 71.88% | 56.25% | 28.12% | 46.88% | 93.75% | 59.38% |
| g4_mask | 28.12% | 21.88% | 28.12% | 31.25% | 50.00% | 31.87% |
| g5_K | 68.75% | 68.75% | 31.25% | 40.62% | 96.88% | 61.25% |
| g5_V | 65.62% | 46.88% | 31.25% | 43.75% | 90.62% | 55.62% |
| g5_KV | 68.75% | 75.00% | 40.62% | 37.50% | 96.88% | 63.75% |
| g5_mask | 18.75% | 18.75% | 28.12% | 21.88% | 21.88% | 21.88% |
| g6_K | 59.38% | 43.75% | 34.38% | 31.25% | 87.50% | 51.25% |
| g6_V | 56.25% | 46.88% | 31.25% | 43.75% | 84.38% | 52.50% |
| g6_KV | 59.38% | 46.88% | 37.50% | 40.62% | 87.50% | 54.37% |
| g6_mask | 59.38% | 46.88% | 37.50% | 37.50% | 87.50% | 53.75% |

- Frozen existing Gemma3-4B-it to Qwen3-4B grouped Stage-A and rank64 cross-head residual Stage-B. No Reader, training, recalibration or new Gemma source capture. Original Gemma features computed in FP32 and stored in FP16 remain unchanged.
- Exactly the same five-task32 question IDs and original Qwen prompt/frame/body/suffix/candidate scoring/normalization as the previous Llama Oracle, checked field-by-field. GSM8K excluded entirely from this experiment.
- Six Qwen receiver groups: human layers1-6,7-12,13-18,19-24,25-30,31-36. Each group independently replaces K,V,KV with corresponding native Qwen entries or masks direct attention access to original transferred-body positions. Full translation,native,and all-transferred-body mask baselines;27 configurations/160 questions/4320 sample configurations.
- Intervention scope is all original translated body token columns after the native anchor, including Question/options, not the earlier Hotpot evidence-only scope. Original cache positions, GQA8KV/32Query heads, header/anchor, suffix and RoPE remain unchanged. Replacement occurs in physical post-RoPE cache space.
- Masking forbids direct attention to selected original cache positions in selected layers; it does not zero tensors, remove token positions or recompute existing body states. Native anchor and newly processed suffix remain accessible, so this is not a literal no-information baseline.
- Native teacher cache is available during this offline Oracle diagnosis. This does not supply an inference-time repair mechanism without original text and is not a prefill-free claim.
- Per-task historical translated and native candidate predictions and scores must reproduce before accepting each sample. GPU smoke also checks native full/cached parity, all-native/all-translated assembly, K/V/layer isolation, fixed anchor and masked-content invariance.
- These32/task are historical exploratory examples. Multiple layer/role searches are not independently confirmed. Existing Gemma and Llama Stage-A training mixtures differ, so cross-pair comparisons are not a sender-only controlled ablation.
- Previous translated+native concatenation training is paused at a saved optimizer/scaler checkpoint and is not automatically resumed after this evaluation.
