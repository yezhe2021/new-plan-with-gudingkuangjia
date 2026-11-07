# Full-Body Full-Sync Depth-State Translator

| Task | Metric | Historical Recolor+C | DST Stage1 | DST Stage2 | Native |
|---|---|---:|---:|---:|---:|
| arc | correct | 67.19% | 62.50% | 60.94% | 87.50% |
| hotpotqa | em | 29.69% | 25.00% | 35.94% | 56.25% |
| hotpotqa | f1 | 46.53% | 37.78% | 53.47% | 68.52% |
| 2wiki | em | 9.38% | 9.38% | 25.00% | 32.81% |
| 2wiki | f1 | 22.05% | 19.15% | 35.22% | 43.45% |
| musique | em | 1.56% | 3.12% | 17.19% | 25.00% |
| musique | f1 | 7.64% | 6.61% | 23.12% | 33.07% |
| sciq | correct | 98.44% | 98.44% | 96.88% | 100.00% |
| mmlu_pro | correct | 25.00% | 20.31% | 23.44% | 53.12% |
| hellaswag | correct | 57.81% | 45.31% | 50.00% | 45.31% |
| gsm8k | correct | 70.31% | 64.06% | 68.75% | 89.06% |
| gsm8k | legacy_correct | 70.31% | 64.06% | 68.75% | 89.06% |

- Full-Sync+new DST changes both alignment and architecture relative to the historical HeteroFold baseline. Historical data/training budgets also differ. End-to-end differences do not isolate the benefit of depth recurrence.
- carry_state=false provides the same allocated parameter structure and local readouts with recurrence reset per Receiver layer; effective use of recurrent parameters changes. This control must be trained separately at the same budget before causal architecture claims.
- Full-Sync guarantees shared synchronization-unit boundaries. Ordered Copy/Drop can select a source state whose end exceeds an individual target token end inside that unit. Audit and report intra-unit lookahead rather than claim strict per-token causality.
- Sender/Receiver remain frozen. Only DST trains in both stages. No Recolor/C initialization or residual wrapper is used; complete RAW Sender features feed layer-specific direct full-width K/V projections plus the joint depth latent. All heads may mix in each dense projection.
- Static Memory is disabled for the initial independent-translator comparison. Existing Native/Recolor+C baselines use identical scoring and rows but historical training differs. Eight64-question cohorts, single seed exploratory; original labels and both GSM scorers retained.
