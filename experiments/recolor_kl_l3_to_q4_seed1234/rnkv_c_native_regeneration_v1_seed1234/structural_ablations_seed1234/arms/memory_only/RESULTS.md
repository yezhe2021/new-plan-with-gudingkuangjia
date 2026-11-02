# Memory-only from-scratch ablation: fixed epoch4/step1024

| Task | Metric | C | Full RN-KV | Memory-only | Native | Memory-only minus RN (pp) |
|---|---|---:|---:|---:|---:|---:|
| arc | accuracy | 67.19% | 64.06% | 51.56% | 87.50% | -12.50 |
| hotpotqa | em | 29.69% | 35.94% | 18.75% | 56.25% | -17.19 |
| hotpotqa | f1 | 46.53% | 49.80% | 32.99% | 68.52% | -16.81 |
| 2wiki | em | 9.38% | 21.88% | 12.50% | 32.81% | -9.38 |
| 2wiki | f1 | 22.05% | 35.32% | 19.28% | 43.45% | -16.05 |
| musique | em | 1.56% | 12.50% | 4.69% | 25.00% | -7.81 |
| musique | f1 | 7.64% | 17.85% | 8.78% | 33.07% | -9.07 |
| sciq | accuracy | 98.44% | 100.00% | 95.31% | 100.00% | -4.69 |
| mmlu_pro | accuracy | 25.00% | 26.56% | 18.75% | 53.12% | -7.81 |
| hellaswag | accuracy | 57.81% | 50.00% | 46.88% | 45.31% | -3.12 |

- Same original data,32 prompt initialization, four epochs, batch8,1024 successful updates and Task CE+0.5 Full KL. Only81920 prompt parameters train; full RN-KV has409600 parameters including four Rank16 adapters.
- This is from-scratch memory-only training, distinct from disabling adapters at inference in the jointly trained checkpoint.
- Fixed seven64-question cohorts. Task-wise paired intervals, rescued/harmed IDs and caps remain in evaluation/<task>/summary.json. Previously inspected cohorts; no fresh independent-confirmation claim.
- Comparison measures the contribution of including/training adapters at this budget; it does not isolate structure from parameter capacity.
- Server B local arm completed. Keep-both is delegated to server A and is not included in this table.
