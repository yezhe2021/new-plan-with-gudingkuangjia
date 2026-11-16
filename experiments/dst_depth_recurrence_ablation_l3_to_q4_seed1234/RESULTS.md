# DST Depth-Recurrence Ablation

Llama3.2-3B → Qwen3-4B, seed 1234, final step 1536 per stage. Full-Sync, no Memory.

Offline diagnostics use the same 192 validation rows. NMSE and cosine use raw projected K/V. The attention probe uses the Native Qwen query at the final suffix token, keeps the Native suffix K/V fixed, and measures cosine after the attention output projection. Logit KL and CE use the full gold target.

| Stage | Arm | K NMSE | V NMSE | K cosine | V cosine | Attention output cosine | Logit KL | Gold CE | 8-task mean (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage1 | carry | 0.11610 | 0.31685 | 0.93662 | 0.81053 | 0.96965 | 1.14921 | 7.80135 | 43.66 [41.19, 46.11] |
| stage1 | no_carry | 0.13229 | 0.35721 | 0.92754 | 0.78217 | 0.96117 | 1.53587 | 7.76976 | 33.49 [31.28, 35.66] |
| stage2 | carry | 0.19009 | 0.36044 | 0.90215 | 0.78817 | 0.90879 | 1.09620 | 0.69190 | 56.84 [54.26, 59.40] |
| stage2 | no_carry | 0.20962 | 0.39946 | 0.89062 | 0.75770 | 0.90233 | 1.21496 | 0.72734 | 53.58 [50.95, 56.16] |

## stage1

| Task | Metric | Carry (%) | No-Carry (%) | Carry − No-Carry (pp) |
|---|---|---:|---:|---:|
| arc | correct | 63.28 [54.69, 71.09] | 57.03 [48.44, 65.62] | 6.25 [-0.78, 13.28] |
| hotpotqa | f1 | 33.97 [26.83, 41.22] | 14.40 [9.57, 19.68] | 19.57 [13.70, 25.66] |
| hotpotqa | em | 20.31 [13.28, 27.34] | 6.25 [2.34, 10.94] | 14.06 [8.59, 20.31] |
| 2wiki | f1 | 23.78 [17.35, 30.63] | 11.57 [7.13, 16.56] | 12.21 [6.63, 18.15] |
| 2wiki | em | 15.62 [9.38, 22.66] | 6.25 [2.34, 10.94] | 9.38 [3.91, 14.84] |
| musique | f1 | 11.88 [8.12, 15.83] | 6.76 [3.88, 10.20] | 5.12 [2.47, 7.96] |
| musique | em | 1.56 [0.00, 3.91] | 1.56 [0.00, 3.91] | 0.00 [0.00, 0.00] |
| sciq | correct | 96.09 [92.19, 99.22] | 93.75 [89.06, 97.66] | 2.34 [0.00, 5.47] |
| mmlu_pro | correct | 19.53 [13.28, 26.56] | 18.75 [12.50, 25.78] | 0.78 [-3.91, 5.47] |
| hellaswag | correct | 46.88 [38.28, 55.47] | 50.00 [41.41, 58.59] | -3.12 [-8.59, 2.34] |
| hellaswag | raw_sum_correct | 39.06 [30.47, 47.66] | 39.06 [30.47, 47.66] | 0.00 [-2.34, 2.34] |
| gsm8k | correct | 53.91 [45.31, 62.50] | 15.62 [9.38, 21.88] | 38.28 [28.91, 47.66] |
| gsm8k | legacy_correct | 53.91 [45.31, 62.50] | 16.41 [10.16, 23.44] | 37.50 [28.12, 46.88] |
| Macro | primary | 43.66 [41.19, 46.11] | 33.49 [31.28, 35.66] | 10.18 [8.12, 12.19] |

## stage2

| Task | Metric | Carry (%) | No-Carry (%) | Carry − No-Carry (pp) |
|---|---|---:|---:|---:|
| arc | correct | 75.00 [67.19, 82.03] | 74.22 [66.41, 81.25] | 0.78 [-4.69, 6.25] |
| hotpotqa | f1 | 62.55 [54.98, 69.78] | 56.69 [49.05, 64.21] | 5.86 [0.69, 11.08] |
| hotpotqa | em | 47.66 [39.06, 56.25] | 42.19 [33.59, 50.78] | 5.47 [0.00, 11.72] |
| 2wiki | f1 | 37.18 [29.52, 44.98] | 36.19 [28.88, 43.86] | 0.99 [-5.33, 7.09] |
| 2wiki | em | 27.34 [19.53, 35.16] | 25.00 [17.97, 32.81] | 2.34 [-4.69, 9.38] |
| musique | f1 | 30.74 [24.25, 37.59] | 24.83 [18.84, 31.08] | 5.91 [0.43, 11.49] |
| musique | em | 17.97 [11.72, 25.00] | 13.28 [7.81, 19.53] | 4.69 [-1.56, 10.94] |
| sciq | correct | 96.09 [92.19, 99.22] | 94.53 [89.84, 97.66] | 1.56 [0.00, 3.91] |
| mmlu_pro | correct | 25.78 [18.75, 33.59] | 24.22 [17.19, 32.03] | 1.56 [-3.91, 7.03] |
| hellaswag | correct | 51.56 [42.97, 60.16] | 51.56 [42.97, 60.16] | 0.00 [-5.47, 5.47] |
| hellaswag | raw_sum_correct | 42.97 [34.38, 51.56] | 45.31 [36.72, 53.91] | -2.34 [-7.81, 3.12] |
| gsm8k | correct | 75.78 [67.97, 82.81] | 66.41 [57.81, 74.22] | 9.38 [3.12, 16.41] |
| gsm8k | legacy_correct | 75.78 [67.97, 82.81] | 66.41 [57.81, 74.22] | 9.38 [3.12, 16.41] |
| Macro | primary | 56.84 [54.26, 59.40] | 53.58 [50.95, 56.16] | 3.26 [1.33, 5.18] |

Each test task uses 128 rows. Intervals are paired question bootstrap 95% intervals with 10,000 replicates and within-task resampling for the macro. Macro uses QA token F1 and accuracy elsewhere. HellaSwag token-mean scores and GSM8K strict numeric extraction are primary.

The held-out cohort previously served Gated Joint evaluation and is disjoint from this training, validation and original DST test cohort by normalized question. Both students use zero Native body tokens. The two arms have equal allocated parameters. No-Carry can leave some previous-state parameters without effective gradients.
