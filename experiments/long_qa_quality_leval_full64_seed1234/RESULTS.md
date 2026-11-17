# L-Eval Long Context, Five Arms, 192 Rows With Nothing Clipped

Llama3.2-3B → Qwen3-4B on L-Eval (`cd34b050…`), 64 rows per source — quality, coursera,
sfiction — 192 rows per arm, seed 1234, `truncation: false`. Nothing was clipped: 0 of 192 rows
touched the context limit, so every arm read the whole document. Generation is capped at 16
tokens, prefill chunk 2048, transfer chunk 1024, headers `[151644, 872, 198]`.

Receiver body tokens by source (min / median / max): quality 2721 / 6580 / 8269, coursera
4841 / 7463 / 16300, sfiction 4944 / 14329 / 24115. So coursera and sfiction push past the 8k
regime the eight-task cohort lives in, and sfiction reaches 24k.

Arms:

- `native` — untouched Qwen3-4B reading the document itself.
- `heterofold_r64` — HeteroFold map rank 64 at step 800 (`3009cc34…`) plus correction
  `14a4c5f7…`, folded.
- `dst_rank64` — the DST rank-64 body, rank manifest `aaae0362…`.
- `gated_dynamic` — the same body with the 36-gate memory (`4547b751…`).
- `recolor_c` — the original recolor lineage: map `e2773fa3…` plus correction `0565836…`, step
  1536, not folded. This is a different line of work from the DST arms, not a re-parameterisation
  of them.

## Accuracy (%)

| Source | Metric | native | heterofold_r64 | dst_rank64 | gated_dynamic | recolor_c |
|---|---|---:|---:|---:|---:|---:|
| quality | correct | 73.44 | 59.38 | 45.31 | 46.88 | 65.62 |
| coursera | correct | 67.19 | 42.19 | 32.81 | 35.94 | 48.44 |
| sfiction | correct | 78.91 | 59.38 | 57.03 | 60.16 | 59.38 |
| sfiction | loyalty_correct | 85.94 | 56.25 | 56.25 | 56.25 | 54.69 |
| sfiction | fact_correct | 71.88 | 62.50 | 57.81 | 64.06 | 64.06 |
| **Mean correct** | | **73.18** | **53.65** | **45.05** | **47.66** | **57.81** |

The ordering among translators is the opposite of the short-context table. On the 512-row
eight-task cohort `gated_dynamic` 53.32 and `dst_rank64` 51.56 beat `recolor_c` 44.92 by 7–8 pp.
Here on long documents `recolor_c` leads by 4.2 pp over `heterofold_r64` and by 10–13 pp over the
two DST arms. Retention against native goes 73% (heterofold), 79% (recolor), 65% (gated), 62%
(dst).

## Paired against heterofold_r64 (pp, 64 paired rows)

| Source | native | dst_rank64 | gated_dynamic | recolor_c |
|---|---|---|---|---|
| quality | +14.06 [+0.00, +28.13] | **−14.06 [−25.00, −3.13]** | **−12.50 [−23.44, −1.56]** | +6.25 [−0.04, +14.06] |
| coursera | +25.00 [+14.06, +37.50] | −9.38 [−20.31, +1.56] | −6.25 [−15.63, +3.13] | **+6.25 [+0.00, +14.06]** |
| sfiction | +19.53 [+11.72, +27.36] | −2.34 [−9.38, +4.69] | +0.78 [−4.69, +6.25] | +0.00 [−3.91, +3.91] |

The two DST arms are the only ones that lose to HeteroFold by a resolvable margin, and they do it
on quality, the source where the questions are hardest and the documents are shortest (median
6.6k). Whether that is about length at all is open: quality is the one source where a short
document still needs multi-span recall.

## The loyalty cliff

On sfiction, `loyalty_correct` runs 85.94 for native and 54.69–56.25 for every translator — a 30
pp drop that is identical across four very different transfer methods, including the one that
wins on quality. `fact_correct` drops far less (71.88 → 57.81–64.06). So on this task the loss
is not "the facts did not survive", it is "the answer stopped being anchored to the passage",
and it appears at rank 16, rank 64 and with a 36-gate memory alike. Nothing in this run supports
attributing it to CompressionRank, gate count or folding.

## Caveats

- 64 rows per source. Paired intervals on a single source are ±10 pp wide, so only the quality
  gap between the DST arms and HeteroFold is comfortably outside noise.
- Answers are short and mostly multiple choice under a 16-token cap. This is not the same regime
  as the LongBench run, where qasper open answers kept 60% of native and the failure was traced
  to losing exact token identity under head+tail clipping. Here nothing was clipped, so that
  mechanism is not available — the loss has to come from what the transfer keeps, not from what
  the input pipeline drops.
- `attempts/01_dtype` records the one aborted configuration:
  `mat1 and mat2 must have the same dtype, but got Half and Float`. The shipped run matches the
  map dtype before the first handoff.
