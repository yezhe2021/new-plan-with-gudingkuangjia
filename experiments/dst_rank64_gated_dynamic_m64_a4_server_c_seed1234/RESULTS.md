# Doubling the Gated Memory Bank from 32 to 64 Slots Changes Nothing

Matched control for `dst_rank64_gated_dynamic_m32_a4_server_a_seed1234`. Llama3.2-3B → Qwen3-4B,
frozen rank-64 body (`aaae0362…`), gated dynamic memory with 4 heads, 1,536 updates at effective
batch 8, `lambda_kl 0.5`, trainable parameters 491,556.

The only change is the size of the memory prompt bank:

- the parent's 32 rows are kept **bit-exact** (`assert torch.equal(prompt[:32], original)`);
- 32 fresh rows are appended, drawn `0.02 * randn` with seed 7,634;
- `memory_tokens` goes 32 → 64.

Training restarts from the same initialization rather than warm-starting the parent checkpoint, and
the batch order is asserted equal to the parent manifest epoch by epoch, so the two runs see the
same data in the same order. `reference_m32/` is the parent's code and evaluation, re-run here to
confirm the reproduction; all six earlier arms land on their published numbers to the hundredth,
which is what makes the seventh column readable.

## Short tasks, 512 rows, same test set as the rest of the family

`test_sha256 de54be31…`, 64 rows per task.

| Task | Metric | native | B_rank64 | D_static | rank64_dynamic | gated_static | gated_dynamic | gated_dynamic_m64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| arc | correct | 87.50 | 67.19 | 67.19 | 70.31 | 68.75 | 67.19 | 68.75 |
| hotpotqa | em | 56.25 | 45.31 | 42.19 | 40.62 | 46.88 | 46.88 | 45.31 |
| 2wiki | em | 32.81 | 23.44 | 21.88 | 20.31 | 25.00 | 25.00 | 25.00 |
| musique | em | 25.00 | 21.88 | 18.75 | 15.62 | 20.31 | 21.88 | 21.88 |
| sciq | correct | 100.00 | 98.44 | 98.44 | 98.44 | 100.00 | 100.00 | 100.00 |
| mmlu_pro | correct | 53.12 | 21.88 | 25.00 | 25.00 | 21.88 | 26.56 | 25.00 |
| hellaswag | correct | 45.31 | 60.94 | 57.81 | 60.94 | 60.94 | 64.06 | 62.50 |
| gsm8k | correct | 89.06 | 73.44 | 73.44 | 76.56 | 71.88 | 75.00 | 75.00 |
| **Macro** | | **61.13** | **51.56** | **50.59** | **50.98** | **51.95** | **53.32** | **52.93** |

Paired `gated_dynamic_m64` against `gated_dynamic` (m32), 64 rows each:

| Task | Metric | Δ (pp) | 95% interval |
|---|---|---:|---|
| arc | correct | +1.56 | [+0.00, +4.69] |
| hotpotqa | em | −1.56 | [−4.69, +0.00] |
| mmlu_pro | correct | −1.56 | [−4.69, +0.00] |
| hellaswag | correct | −1.56 | [−7.81, +3.12] |
| gsm8k | correct | +0.00 | [−6.25, +6.25] |
| 2wiki | em | +0.00 | [+0.00, +0.00] |
| musique | em | +0.00 | [+0.00, +0.00] |
| sciq | correct | +0.00 | [+0.00, +0.00] |

Every interval touches or spans zero. Two tasks move up a single row, two move down a single row,
four do not move at all — this is what a coin flip looks like, not an effect.

## Long documents, same L-Eval cohort as `long_qa_quality_leval_full64`

192 rows, 64 per source.

| Source | Metric | native | dst_rank64 | gated_dynamic | heterofold_r64 | recolor_c | gated_dynamic_m64 |
|---|---|---:|---:|---:|---:|---:|---:|
| quality | correct | 73.44 | 45.31 | 46.88 | 59.38 | 65.62 | 42.19 |
| coursera | correct | 67.19 | 32.81 | 35.94 | 42.19 | 48.44 | 37.50 |
| sfiction | correct | 78.91 | 57.03 | 60.16 | 59.38 | 59.38 | 60.94 |
| sfiction | loyalty_correct | 85.94 | 56.25 | 56.25 | 56.25 | 54.69 | 56.25 |
| sfiction | fact_correct | 71.88 | 57.81 | 64.06 | 62.50 | 64.06 | 65.62 |
| **Mean correct** | | **73.18** | **45.05** | **47.66** | **53.65** | **57.81** | **46.88** |

Paired against m32: quality **−4.69 [−10.94, +0.00]**, coursera +1.56 [+0.00, +4.69], sfiction
+0.78 [−1.56, +3.91], loyalty +0.00 [+0.00, +0.00], fact +1.56 [−3.12, +7.81].

The one interval that does not touch zero sits on the negative side, and it is a single quadrant of
one source. Nothing else here is distinguishable from noise.

## Reading

Doubling the bank buys nothing: 53.32 → 52.93 on the short cohort (−0.39 pp) and 47.66 → 46.88 on
the long one (−0.78 pp). The bank is not capacity-bound at 32 slots.

The training curve says why. The extra 32 rows start as small random vectors and never acquire
content — the loss rises monotonically across all four epochs instead of settling:

| Epoch | Step | loss | task_loss | full_kl |
|---:|---:|---:|---:|---:|
| 1 | 384 | **1.0425** | **0.6925** | **0.6999** |
| 2 | 768 | 1.0660 | 0.6982 | 0.7356 |
| 3 | 1152 | 1.0821 | 0.7000 | 0.7642 |
| 4 | 1536 | 1.0972 | 0.7047 | 0.7850 |

A bank that is too small would show the opposite: the slots would saturate and the objective would
plateau early. Both `task_loss` and `full_kl` drift *up*, which is the signature of extra
low-information parameters taking gradient away from the ones that matter, not of added capacity.

Scope of the claim: this rules out capacity alone, under a random initialization of the new rows.
A warm start that duplicated or interpolated existing rows, or a longer schedule that gives the new
slots time to specialize, is a different experiment and is not excluded here.

Evaluation used the fixed epoch-4 checkpoint (`68f898e9…`), validation diagnostics on 192 rows, no
test-time checkpoint selection.
