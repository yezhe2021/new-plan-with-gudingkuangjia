# A Learned Fusion Projection Buys Multiple Choice and Costs GSM8K

The trained counterpart to `fusion_equal50_l3_g3_to_q4_frozenab_seed1234`. Instead of a fixed
element-wise mean over the two frozen branches, a bias-free K/V projection is trained on top of the
concatenated pair, and nothing else is: both Stage-A+B branches and the receiver stay frozen, there
is no Reader and no LoRA.

- **Shape** — separate K and V and target-layer matrices, 36 layers, 8 heads, head dim 128,
  150,994,944 trainable parameters.
- **Initialization** — `[0.5 I, 0.5 I]` per branch, so step 0 is *exactly* the equal-mean baseline.
  This is what makes the delta readable as "what training added".
- **Objective** — full-vocabulary forward KL, Native ‖ Student, temperature 1, on the same
  mixed 64-token Native/student prefixes and up to 8 supervised positions as residual Stage B. No
  gold CE.
- **Schedule** — 4 epochs, 1,024 updates at effective batch 8, lr 1e-4, wd 0, clip 30, one warmup
  epoch at student fraction 0, then 0.5. Fixed final epoch-4 step-1024 checkpoint; epoch
  checkpoints are diagnostic only.

Same six fixed subsets and scoring as the parent, so every column below is row-identical.

| Task | n | qwen_native | gemma_native | llama_ab | gemma_ab | equal50_fusion | concat_projection |
|---|---:|---:|---:|---:|---:|---:|---:|
| openbookqa | 128 | 80.47 | 75.78 | 68.75 | 58.59 | 66.41 | 75.78 |
| arc_challenge | 128 | 86.72 | 78.91 | 64.06 | 55.47 | 63.28 | 71.09 |
| hellaswag | 64 | 35.94 | 32.81 | 34.38 | 31.25 | 35.94 | 37.50 |
| gsm8k | 32 | 96.88 | 93.75 | 31.25 | 68.75 | 62.50 | 34.38 |
| mmlu_pro | 128 | 40.62 | 29.69 | 25.78 | 28.91 | 28.91 | 22.66 |
| sciq | 64 | 89.06 | 89.06 | 87.50 | 81.25 | 92.19 | 87.50 |
| **Pooled** | 544 | **69.30** | **63.24** | **53.49** | **50.92** | **56.07** | **56.62** |

Paired against the untrained mean, `projection_only / mean_only`:

| Task | vs equal50_fusion | vs llama_ab | vs gemma_ab |
|---|---|---|---|
| openbookqa | 16 / 4 | 15 / 6 | 24 / 2 |
| arc_challenge | 14 / 4 | 16 / 7 | 25 / 5 |
| hellaswag | 6 / 5 | 5 / 3 | 7 / 3 |
| gsm8k | **2 / 11** | 7 / 6 | **2 / 13** |
| mmlu_pro | 8 / 16 | 9 / 13 | 10 / 18 |
| sciq | 2 / 5 | 4 / 4 | 7 / 3 |

## Reading

Pooled, 1,024 updates move 56.07 → 56.62: +0.55 pp, 48 items gained against 45 lost. That headline
number is useless, because two large opposite movements cancel inside it.

**Choice tasks improve.** openbookqa +9.37, arc_challenge +7.81, hellaswag +1.56. Against the mean
the projection wins 36 rows and loses 13 on those three tasks. On openbookqa it lands exactly on
Gemma native (75.78 on the same 128 rows) — the projection has learned to read the Gemma branch
well enough to match the sender it came from.

**Generation collapses.** gsm8k falls 62.50 → 34.38, a 28-point drop, and against the mean it is
2 rows gained against 11 lost. Against `gemma_ab` it is 2 gained and 13 lost. The branch that had
GSM8K at 68.75 is right there in the concatenation, and the trained projection cannot get it back
out.

The mechanism is in the objective, not the architecture: full-vocabulary KL against the native
teacher with no gold CE, over up to 8 supervised positions per example, is a distribution-matching
loss that rewards sharpening the next-token distribution on short answer prefixes. GSM8K is a
4096-token greedy generation where the failure mode is compounding — a projection tuned to match
a distribution at position 8 has no pressure to keep the tail coherent. The MMLU-Pro drop (−6.25)
points the same way: it is the one task where the pooled model loses to both parents.

Net: fusion with training trades a generative skill for a discriminative one and comes out flat.
The equal mean, which is 150M parameters cheaper and needs no GPU time, is the better default.

Caveats carried from the run's own config:

- 1,024 extra updates over an **untrained** baseline is not an equal-training-budget comparison
  against the single branches, which had full Stage A+B;
- the two branches keep their different Stage-A data mixtures;
- MMLU-Pro is the existing custom split;
- Full-Sync accuracy interface only, with no prefill-free latency claim.
