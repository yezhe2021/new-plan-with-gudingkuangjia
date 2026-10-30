# A/C/D native KV substitution and masking:64 per task

| Method | Condition | ARC64 accuracy | HellaSwag64 accuracy | HotpotQA64 EM | HotpotQA64 F1 |
|---|---|---:|---:|---:|---:|
| Qwen | native | 90.62% | 35.94% | 48.44% | 62.57% |
| A | unmasked | 65.62% | 42.19% | 15.62% | 31.64% |
| A | g1_k | 62.50% | 42.19% | 12.50% | 29.65% |
| A | g1_v | 64.06% | 42.19% | 17.19% | 33.16% |
| A | g1_kv | 62.50% | 40.62% | 15.62% | 30.58% |
| A | g1_mask | 67.19% | 39.06% | 17.19% | 29.96% |
| A | g2_k | 62.50% | 42.19% | 14.06% | 32.44% |
| A | g2_v | 60.94% | 37.50% | 26.56% | 36.63% |
| A | g2_kv | 67.19% | 39.06% | 25.00% | 40.30% |
| A | g2_mask | 64.06% | 35.94% | 18.75% | 32.40% |
| A | g3_k | 62.50% | 40.62% | 7.81% | 27.05% |
| A | g3_v | 57.81% | 31.25% | 0.00% | 6.77% |
| A | g3_kv | 62.50% | 39.06% | 14.06% | 32.55% |
| A | g3_mask | 62.50% | 35.94% | 6.25% | 16.74% |
| A | g4_k | 79.69% | 42.19% | 20.31% | 36.61% |
| A | g4_v | 23.44% | 34.38% | 0.00% | 1.97% |
| A | g4_kv | 75.00% | 43.75% | 37.50% | 51.55% |
| A | g4_mask | 45.31% | 34.38% | 0.00% | 9.66% |
| A | g5_k | 82.81% | 43.75% | 14.06% | 28.89% |
| A | g5_v | 68.75% | 37.50% | 9.38% | 17.84% |
| A | g5_kv | 84.38% | 43.75% | 17.19% | 34.62% |
| A | g5_mask | 18.75% | 42.19% | 6.25% | 21.54% |
| A | g6_k | 64.06% | 43.75% | 12.50% | 29.41% |
| A | g6_v | 59.38% | 43.75% | 7.81% | 14.09% |
| A | g6_kv | 65.62% | 45.31% | 14.06% | 29.89% |
| A | g6_mask | 64.06% | 45.31% | 9.38% | 20.13% |
| C | unmasked | 65.62% | 40.62% | 29.69% | 43.86% |
| C | g1_k | 65.62% | 42.19% | 29.69% | 42.91% |
| C | g1_v | 65.62% | 42.19% | 28.12% | 40.80% |
| C | g1_kv | 65.62% | 42.19% | 26.56% | 41.61% |
| C | g1_mask | 68.75% | 34.38% | 29.69% | 39.47% |
| C | g2_k | 65.62% | 40.62% | 29.69% | 43.75% |
| C | g2_v | 65.62% | 39.06% | 31.25% | 42.83% |
| C | g2_kv | 65.62% | 40.62% | 31.25% | 44.51% |
| C | g2_mask | 64.06% | 34.38% | 29.69% | 40.36% |
| C | g3_k | 70.31% | 39.06% | 28.12% | 41.44% |
| C | g3_v | 67.19% | 39.06% | 28.12% | 41.30% |
| C | g3_kv | 67.19% | 42.19% | 28.12% | 41.38% |
| C | g3_mask | 57.81% | 34.38% | 10.94% | 25.49% |
| C | g4_k | 81.25% | 40.62% | 29.69% | 43.39% |
| C | g4_v | 62.50% | 37.50% | 34.38% | 47.85% |
| C | g4_kv | 75.00% | 39.06% | 32.81% | 47.89% |
| C | g4_mask | 40.62% | 37.50% | 0.00% | 8.58% |
| C | g5_k | 81.25% | 39.06% | 25.00% | 39.35% |
| C | g5_v | 67.19% | 39.06% | 31.25% | 44.17% |
| C | g5_kv | 84.38% | 39.06% | 26.56% | 41.23% |
| C | g5_mask | 15.62% | 39.06% | 20.31% | 32.19% |
| C | g6_k | 65.62% | 37.50% | 29.69% | 43.96% |
| C | g6_v | 67.19% | 42.19% | 26.56% | 40.41% |
| C | g6_kv | 64.06% | 40.62% | 29.69% | 44.79% |
| C | g6_mask | 65.62% | 37.50% | 21.88% | 30.64% |
| D | unmasked | 64.06% | 43.75% | 10.94% | 29.42% |
| D | g1_k | 64.06% | 42.19% | 6.25% | 27.09% |
| D | g1_v | 64.06% | 42.19% | 7.81% | 27.82% |
| D | g1_kv | 64.06% | 43.75% | 6.25% | 24.32% |
| D | g1_mask | 64.06% | 32.81% | 10.94% | 28.39% |
| D | g2_k | 64.06% | 46.88% | 9.38% | 28.83% |
| D | g2_v | 67.19% | 42.19% | 18.75% | 38.93% |
| D | g2_kv | 64.06% | 43.75% | 10.94% | 30.84% |
| D | g2_mask | 64.06% | 39.06% | 9.38% | 30.29% |
| D | g3_k | 64.06% | 48.44% | 7.81% | 27.70% |
| D | g3_v | 59.38% | 42.19% | 3.12% | 17.98% |
| D | g3_kv | 64.06% | 45.31% | 21.88% | 40.07% |
| D | g3_mask | 59.38% | 42.19% | 4.69% | 21.37% |
| D | g4_k | 76.56% | 45.31% | 29.69% | 44.76% |
| D | g4_v | 20.31% | 34.38% | 0.00% | 2.77% |
| D | g4_kv | 73.44% | 46.88% | 35.94% | 52.42% |
| D | g4_mask | 42.19% | 40.62% | 3.12% | 16.03% |
| D | g5_k | 79.69% | 43.75% | 14.06% | 32.31% |
| D | g5_v | 59.38% | 42.19% | 17.19% | 37.38% |
| D | g5_kv | 79.69% | 43.75% | 20.31% | 37.90% |
| D | g5_mask | 14.06% | 50.00% | 12.50% | 30.15% |
| D | g6_k | 65.62% | 39.06% | 7.81% | 25.45% |
| D | g6_v | 65.62% | 46.88% | 9.38% | 22.54% |
| D | g6_kv | 65.62% | 45.31% | 6.25% | 23.21% |
| D | g6_mask | 67.19% | 40.62% | 4.69% | 17.55% |

- Each dataset uses the first64 questions in the immutable parent order, selected without outcome-dependent filtering. Same prompts, token grid, public header, scoring and decoding limits as Recolor-KL. No GSM8K.
- A/C/D and both LLMs are frozen; no Reader, Grouped Stage-A, Full-Sync, retraining, or automatic resume of other paused experiments.
- Each method: unmasked plus6 independent groups times native K-only, native V-only, native KV, or attention-access mask:25 configurations. One shared native baseline/question.192 questions,14592 total configurations;4032 compatible unmasked/mask results reused with immutable provenance.
- Groups are human layers1-6,7-12,13-18,19-24,25-30,31-36. Replace only ALL original transferred Prompt body positions including questions/options/evidence. Same range for masks. Public header and newly computed suffix/generated states retained.
- Native Qwen KV is explicitly offline oracle information. Substitution occurs after cache assembly in post-k_norm/post-RoPE coordinates on matching Qwen token positions. No re-normalization/re-RoPE. Changes apply before reprocessing suffix/answer; independent cloned caches avoid cross-condition contamination.
- No intervention recomputes original Prompt body states after mixing. Native higher-layer states contain effects of a fully native computation; results diagnose causal interventions in hybrid caches, not independent additive layer errors.
- Smoke checks ordinary baseline historical parity, self-substitution bitwise logits, all-native endpoint vs native cached baseline, cached-native vs direct-prefill, protected positions/components, all6 groups, and complete scoring/generation.
- Masking blocks attention access rather than zeroing caches. Both K/V access is blocked; K-only/V-only substitutions separate component effects. Mask dependency does not automatically imply translation error.
- Report paired score changes, native-substitution minus same-group mask, rescued/harmed IDs, shared-correct retention and recovery ratios only when native exceeds translated. Many exploratory comparisons on64 questions require held-out confirmation; no per-dataset best mask is promoted as a universal method.
- Native cached vs historical direct-prefill differences, if any, are explicitly recorded; current native baseline uses the same continuation/cache scoring protocol as hybrids.
- Old larger mask run was stopped at user request and retained. Existing outputs are never overwritten. Pipeline fails on errors, saves per configuration, runs detached, no automatic upload or monitoring.
- Canonical native-cache baseline uses the SAME separately prefetched public header as every translated/hybrid arm; only native body positions come from full Qwen prefill. Header-only and long-prefill FP16 kernels can differ. Full-native-body substitution must match this baseline exactly; cached native is additionally checked against the original direct-prefill scorer during smoke.
