# Full-Body Full-Sync Depth-State Translator

| Task | Metric | Historical Recolor+C | DST Stage2 | DRT Stage1 | DRT Stage2 | Native |
|---|---|---:|---:|---:|---:|---:|
| arc | correct | 67.19% | 60.94% | 67.19% | 59.38% | 87.50% |
| hotpotqa | em | 29.69% | 35.94% | 25.00% | 40.62% | 56.25% |
| hotpotqa | f1 | 46.53% | 53.47% | 35.53% | 56.05% | 68.52% |
| 2wiki | em | 9.38% | 25.00% | 7.81% | 18.75% | 32.81% |
| 2wiki | f1 | 22.05% | 35.22% | 15.42% | 29.06% | 43.45% |
| musique | em | 1.56% | 17.19% | 3.12% | 9.38% | 25.00% |
| musique | f1 | 7.64% | 23.12% | 7.58% | 17.22% | 33.07% |
| sciq | correct | 98.44% | 96.88% | 96.88% | 96.88% | 100.00% |
| mmlu_pro | correct | 25.00% | 23.44% | 21.88% | 18.75% | 53.12% |
| hellaswag | correct | 57.81% | 50.00% | 46.88% | 51.56% | 45.31% |
| gsm8k | correct | 70.31% | 68.75% | 46.88% | 70.31% | 89.06% |
| gsm8k | legacy_correct | 70.31% | 68.75% | 46.88% | 70.31% | 89.06% |

- Full-Sync guarantees shared synchronization-unit boundaries. Ordered Copy/Drop can select a source state whose end exceeds an individual target token end inside that unit. Audit and report intra-unit lookahead rather than claim strict per-token causality.
- Static Memory is disabled for the initial independent-translator comparison. Existing Native/Recolor+C baselines use identical scoring and rows but historical training differs. Eight64-question cohorts, single seed exploratory; original labels and both GSM scorers retained.
- DRT is a fresh model, not initialized from DST or Recolor+C. It uses input-dependent attention across Sender depth at each aligned token. Historical C differs in interface and training budget.
- DRT and DST differ in architecture and parameter count (exact counts in manifest). No dynamic-vs-static routing ablation is scheduled.
