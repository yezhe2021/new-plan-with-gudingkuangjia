# DST+Rank64 and Dynamic Memory joint fine-tuning (384 updates)

| Task | Metric | Native | Rank64 | Rank64+Static | Rank64+Dynamic before joint | Joint |
|---|---|---:|---:|---:|---:|---:|
| arc | correct | 87.50% | 67.19% | 67.19% | 70.31% | 73.44% |
| hotpotqa | em | 56.25% | 45.31% | 42.19% | 40.62% | 42.19% |
| hotpotqa | f1 | 68.52% | 58.85% | 56.93% | 55.68% | 57.14% |
| 2wiki | em | 32.81% | 23.44% | 21.88% | 20.31% | 23.44% |
| 2wiki | f1 | 43.45% | 32.60% | 30.83% | 33.48% | 34.74% |
| musique | em | 25.00% | 21.88% | 18.75% | 15.62% | 14.06% |
| musique | f1 | 33.07% | 31.63% | 28.19% | 24.90% | 20.27% |
| sciq | correct | 100.00% | 98.44% | 98.44% | 98.44% | 100.00% |
| mmlu_pro | correct | 53.12% | 21.88% | 25.00% | 25.00% | 23.44% |
| hellaswag | correct | 45.31% | 60.94% | 57.81% | 60.94% | 57.81% |
| gsm8k | correct | 89.06% | 73.44% | 73.44% | 76.56% | 78.12% |
| gsm8k | legacy_correct | 89.06% | 73.44% | 73.44% | 76.56% | 78.12% |
| **mean** | | **61.13** | **51.56** | **50.59** | **50.98** | **51.56** |

Paired over all 512 test rows with 4,000 bootstrap resamples: Joint − Rank64
+0.00 [−2.54, +2.54], Joint − Rank64+Dynamic +0.59 [−1.17, +2.34]. Against
Rank64 the 384 updates touch 42 rows of 512 — 21 recovered, 21 broken, net zero.

- Starting point is the frozen Rank64 translator plus the dynamic memory bank
  from `dst_stage1_rank64_dynamic_memory_m32_a4_seed1234`; both are unfrozen for
  384 updates at 1e-5 on the Rank64 residual and 1e-4 on the memory, epoch 5,
  task CE plus the same 0.5 full-KL as the earlier stages.
- The total is unchanged but the distribution is not. Against Rank64: arc
  +6.25 [+1.56, +12.50], gsm8k +4.69 [+0.00, +10.94], sciq +1.56, mmlu_pro +1.56,
  against musique −7.81 [−15.62, +0.00], hotpotqa −3.12, hellaswag −3.12.
  ARC and GSM8K are the two tasks the joint loss repairs; MuSiQue and HellaSwag
  pay for it.
- Read against the pre-joint dynamic arm the same pattern is milder, +0.59
  overall, with no task interval excluding zero on either side.
- So joint fine-tuning is an error-allocation knob, not a gain: it buys ARC
  (+4 rows) and GSM8K (+3 rows) with MuSiQue (−5 rows) and gives back the rest.
  On this pipeline the Rank64 residual is at a local optimum for a 384-update
  budget; longer schedules were not tried.
