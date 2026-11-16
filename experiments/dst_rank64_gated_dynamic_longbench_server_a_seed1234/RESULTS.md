# LongBench: The Translator Keeps 60% On Open Answers And Loses Everything Once The Context Is Clipped

**Snapshot, 2026-11-16 12:27 — the run is still going.** `full200` has finished
155 of 200 musique rows and has not started qasper. Everything below is computed
from what was on disk at that moment; the musique column will move, qasper is
missing entirely. Two stages are reported separately because they answer
different questions: `pilot64` is complete on both tasks and is the only place
qasper exists, `full200` is the 200-row extension of musique.

Same checkpoint as the short-context table: `runs/epoch4_step1536.pt` of
`dst_rank64_gated_dynamic_m32_a4_server_a_seed1234`, pinned by
`gated_checkpoint` = `4547b751…`. Rank64 and the memory bank are frozen; the
fourth arm `recolor_c` is the frozen Recolor+C correction
(`runs/C/epoch4_step800.pt`) added here for the first time on a long-context
task. Input cap is 16,384 tokens with head/tail retention and a complete
question; generation caps are upstream LongBench's, 32 tokens for musique and
128 for qasper, greedy, non-thinking Qwen.

## pilot64 — 64 questions per task, all four arms

| Task | Clipped | Native | Recolor+C | Gated Dynamic | DST Rank64 |
|---|---:|---:|---:|---:|---:|
| musique | 49/64 | **34.78** | 13.09 | 12.27 | 10.22 |
| qasper | 3/64 | **43.00** | 33.07 | 25.76 | 23.91 |

Paired differences, 10,000 bootstrap resamples (gated − reference):

| Task | vs Rank64 | vs Recolor+C | vs Native |
|---|---:|---:|---:|
| musique | +2.05 [−3.11, +7.50] | −0.83 [−9.74, +7.77] | −22.51 [−33.43, −11.95] |
| qasper | +1.85 [−4.34, +8.18] | **−7.31 [−14.34, −0.09]** | −17.24 [−26.32, −7.96] |

Restricted to rows that fit under the cap, musique is a different experiment:
n=15, native 23.33, gated 25.33, rank64 20.00, recolor_c 10.67, gated − native
**+2.00 [−10.67, +18.67]**. Fifteen rows decide nothing, but the direction is
worth noting — the gap in the overall musique column is produced by trimming,
not by length.

## full200 — musique only, 152 rows scored

| Group | n | Clipped | Native | Gated Dynamic | DST Rank64 | Recolor+C |
|---|---:|---:|---:|---:|---:|---:|
| overall | 152 | 118 | **32.66** | 18.58 | 16.82 | 16.79 |
| fits under cap | 34 | 0 | 30.06 | 25.25 | 24.08 | 16.00 |

gated − rank64 +1.76 [−1.04, +4.67], gated − recolor_c +1.79 [−4.58, +8.04],
gated − native −14.08 [−20.92, −7.21]. On the 34 retained rows the native gap
falls to −4.80 [−18.33, +9.31] and recolor_c is the arm that separates
(−9.25 for gated against it).

## Where the loss actually sits

93 musique rows carry `original_qwen_tokens`, which lets the split be made on
the length the document had before trimming:

| Original length | n | Clipped | Native | Gated | Gated − Native | Gated − Rank64 |
|---|---:|---:|---:|---:|---:|---:|
| 12k–16k | 17 | 0 | 32.1 | 29.9 | **−2.16** | +4.71 |
| >16k | 71 | 71 | **34.5** | 13.2 | **−21.28** | +1.94 |

Inside the cap the translator is within 2.2 points of native. Past the cap it
falls to a third — and native does not fall at all, it rises, from 32.1 to 34.5
on the very rows where two thirds of each document was deleted.

- **The answers were not the part that got deleted.** For the 68 clipped rows
  that have a gold span, the answer occurrence always survives in the retained
  head or tail. Split by where the gold first appears: head n=50, native 34.9 /
  gated 12.2 / rank64 12.1; tail n=18, native 36.3 / gated 18.1 / rank64 10.7.
  Native reads the answer and says it; the translator has it in front of it and
  does not.
- **It is not a splice-point alignment failure either.** A mis-mapping at the
  `head + "\n\n" + tail` join would hit the tail side hardest, since that is the
  segment whose offset origin jumps. The tail group is the *better* one. The
  head group, which is the document's untouched opening, is the worse one.
- **The clipped rows are the easy ones, not the hard ones.** Gold strings occur
  in the context 16.8 times on average in the clipped group against 3.2 times in
  the retained group (median 3 in both). Long-document musique questions repeat
  the answer entity far more often, so a reader that leans on the repetition or
  on parametric memory scores higher there — which is exactly what native does.
- **This is the RULER failure showing up on a semantic task.** The two
  translator arms stay 73% identical in their predictions on clipped rows and
  emit normal-length answers (16 characters on average), so they are not
  degenerating into noise; they are failing to carry the exact entity token
  across a long span. musique grades exact strings (`Green Bay`), so it collapses
  the same way RULER's digit identities did. qasper grades sentences, and the
  translator keeps 60% of native there.
- **Recolor+C overtakes the translator on open-ended reading.** On qasper it is
  33.07 against gated 25.76, a gap that does not cross zero
  (−7.31 [−14.34, −0.09]), while on musique it is the worst arm. It is the first
  task in this project where the untrained-residual baseline significantly beats
  a translator arm, and it happens on the one task in the pair that does not
  grade exact strings.

## Artefacts

- `pilot64/` — 64 rows per task, complete, own manifest.
- `full200/` — the 200-row extension, `status: evaluating`, 155 rows on disk at
  snapshot time, its `manifest.json` pins the pilot manifest and the parent
  manifest and asserts the first 64 row ids match.
- `evaluation/` — the original 93-row musique run that both stages reuse. These
  are the only records that store `original_qwen_tokens`; the later stages wrote
  four fields only, so the length split above can only be computed on those 93.
- `inputs/longbench.zip` (109 MB, sha256 `cb45b11a…`) is excluded from the
  repository; `config.json` and `manifest.json` pin its hash.
- `controls/pause_requested.json` is the empty 10:48 pause flag from the first
  pass. It sits in the parent `controls/`, so it does **not** stop `full200`,
  which only checks `full200/controls/`.
