# D: frozen DST Stage1 + Rank64 + fresh Static M32+A4 all36

| Task | Metric | A Stage1 | B Rank64 | E Full Stage2 | F Full+Static | D Rank64+Static | C+Static | Native |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| arc | correct | 62.50% | 67.19% | 60.94% | 59.38% | 67.19% | 68.75% | 87.50% |
| hotpotqa | em | 25.00% | 45.31% | 35.94% | 34.38% | 42.19% | 43.75% | 56.25% |
| hotpotqa | f1 | 37.78% | 58.85% | 53.47% | 53.45% | 56.93% | 58.16% | 68.52% |
| 2wiki | em | 9.38% | 23.44% | 25.00% | 25.00% | 21.88% | 28.12% | 32.81% |
| 2wiki | f1 | 19.15% | 32.60% | 35.22% | 33.37% | 30.83% | 42.07% | 43.45% |
| musique | em | 3.12% | 21.88% | 17.19% | 17.19% | 18.75% | 20.31% | 25.00% |
| musique | f1 | 6.61% | 31.63% | 23.12% | 22.31% | 28.19% | 28.50% | 33.07% |
| sciq | correct | 98.44% | 98.44% | 96.88% | 98.44% | 98.44% | 100.00% | 100.00% |
| mmlu_pro | correct | 20.31% | 21.88% | 23.44% | 26.56% | 25.00% | 35.94% | 53.12% |
| hellaswag | correct | 45.31% | 60.94% | 50.00% | 46.88% | 57.81% | 60.94% | 45.31% |
| gsm8k | correct | 64.06% | 73.44% | 68.75% | 62.50% | 73.44% | 70.31% | 89.06% |
| gsm8k | legacy_correct | 64.06% | 73.44% | 68.75% | 62.50% | 73.44% | 70.31% | 89.06% |

- Only bias-free K/V bottleneck residuals train in B; frozen original DST Stage1, Qwen and Llama.
- Fixed original Stage1 initialization; no claim it has fully converged. Comparison changes parameterization and trainable count, not just parameter count.
- Full-Sync alignment/splits and Stage2 objectives match original DST. Final epoch4 fixed. No test selection.
- D trains fresh Static M32+A4 all36 after freezing B; no learned Memory warm start. Existing F is DST full-parameter Stage2 plus retrained Static.
- Eight tasks each64, single seed exploratory; no memory in B. Memory adds32 answer positions in D.
