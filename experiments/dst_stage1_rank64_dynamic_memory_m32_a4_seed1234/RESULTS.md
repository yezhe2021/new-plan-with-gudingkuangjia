# DST Stage 1 + Rank64 with Dynamic Memory, M32+A4 all36

| Task | Metric | Rank64 | Rank64 + Static | Rank64 + Dynamic | Recolor+C+Static | Native |
|---|---|---:|---:|---:|---:|---:|
| arc | correct | 67.19% | 67.19% | 70.31% | 68.75% | 87.50% |
| hotpotqa | em | 45.31% | 42.19% | 40.62% | 43.75% | 56.25% |
| hotpotqa | f1 | 58.85% | 56.93% | 55.68% | 58.16% | 68.52% |
| 2wiki | em | 23.44% | 21.88% | 20.31% | 28.12% | 32.81% |
| 2wiki | f1 | 32.60% | 30.83% | 33.48% | 42.07% | 43.45% |
| musique | em | 21.88% | 18.75% | 15.62% | 20.31% | 25.00% |
| musique | f1 | 31.63% | 28.19% | 24.90% | 28.50% | 33.07% |
| sciq | correct | 98.44% | 98.44% | 98.44% | 100.00% | 100.00% |
| mmlu_pro | correct | 21.88% | 25.00% | 25.00% | 35.94% | 53.12% |
| hellaswag | correct | 60.94% | 57.81% | 60.94% | 60.94% | 45.31% |
| gsm8k | correct | 73.44% | 73.44% | 76.56% | 70.31% | 89.06% |
| gsm8k | legacy_correct | 73.44% | 73.44% | 76.56% | 70.31% | 89.06% |

- Dynamic and Static use the same frozen Rank64 translator, parameter shapes and initialization, data, accepted-update budget, answer masks and positions.
- Dynamic construction reads the translated body. The 32 Memory positions are appended, and all 36 answer layers can read body and Memory.
- Fixed final epoch 4; validation is diagnostic. Paired test results use the same eight previously inspected 64-question cohorts and one training seed.
- The no-Memory comparison also changes the answer positions by 32 tokens.
