# Main-protocol evidence-column Oracle repair pilot

Original full-prompt main protocol, exact same32 questions and checkpoints,28 configurations. Question/instruction KV held translated under evidence interventions; native baseline is full native cache.

| Arm | EM | F1 | Gold token-mean NLL |
|---|---:|---:|---:|
| translated | 18.75% | 28.45% | 3.2921 |
| native | 40.62% | 53.24% | 4.7735 |
| reject_all | 9.38% | 9.38% | 5.9467 |
| evidence_native_all | 31.25% | 38.35% | 3.2100 |
| g1_K | 25.00% | 32.72% | 3.9647 |
| g1_V | 18.75% | 25.95% | 3.4493 |
| g1_KV | 21.88% | 30.64% | 3.3579 |
| g1_mask | 18.75% | 24.17% | 3.8729 |
| g2_K | 25.00% | 34.48% | 3.3161 |
| g2_V | 18.75% | 28.45% | 3.3384 |
| g2_KV | 25.00% | 34.48% | 3.3737 |
| g2_mask | 18.75% | 28.45% | 3.4129 |
| g3_K | 21.88% | 31.35% | 3.1560 |
| g3_V | 18.75% | 28.45% | 3.3177 |
| g3_KV | 18.75% | 30.01% | 3.1931 |
| g3_mask | 21.88% | 31.04% | 3.4407 |
| g4_K | 28.12% | 37.31% | 2.8440 |
| g4_V | 18.75% | 28.76% | 3.3368 |
| g4_KV | 21.88% | 32.84% | 2.8086 |
| g4_mask | 15.62% | 21.79% | 3.3445 |
| g5_K | 21.88% | 30.54% | 3.1093 |
| g5_V | 25.00% | 32.62% | 3.2701 |
| g5_KV | 25.00% | 32.10% | 3.0297 |
| g5_mask | 9.38% | 13.97% | 3.6769 |
| g6_K | 21.88% | 30.64% | 2.9694 |
| g6_V | 18.75% | 25.95% | 3.8997 |
| g6_KV | 18.75% | 27.51% | 3.1532 |
| g6_mask | 6.25% | 12.72% | 4.2161 |

- Original HeteroFold/ours main input protocol preserved: exact existing full prompt, source/receiver frame IDs and suffix IDs. Same32 pilot IDs, fixed Hotpot A+B, no Reader, greedy64, same answer EM/F1. All32 translated and full-native outputs must reproduce saved main-protocol predictions and raw generation exactly.
- Only fully-contained evidence paragraph/header token columns are replaced or masked; instruction/Question tokens and boundary-crossing mixed tokens remain unchanged. Native replacement is post-k_norm/post-RoPE at the identical physical positions.
- Question KV remains precomputed translated KV under evidence interventions, and can retain evidence information. Mask-all blocks direct evidence-position access during suffix/generation/NLL, not all indirect evidence information; it is not a question-only baseline and does not reprocess Question.
- Full native baseline replaces the complete cache. All-evidence-native control replaces evidence columns only, leaving Question/instruction translated; these endpoints need not agree. Four baselines plus6 groups x4 interventions =28 configurations (896 generations).
- No new training or parameters. Per-layer attention masking, not zeroing. Six fixed groups of6 target layers; no head sweep. All configurations get full generation and answer-token-mean Gold NLL without NLL screening.
- These32 pilot questions have already been inspected and remain exploratory; next128 IDs remain reserved and are not evaluated. Paired intervals are uncorrected for multiple comparisons. No confirmation/winner claim.
- Recovery ratios use same-row full-native versus translated F1 gap when positive, unclipped. Native availability is Oracle information, not a deployable method. Negative single-group results cannot rule out group interactions.
