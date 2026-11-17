# HeteroFold r16 Against DST and Recolor, Same 512 Rows, Plus Handoff Cost

Llama3.2-3B → Qwen3-4B. Five arms on 64 rows per task, eight tasks, 512 rows total. This is the
same cohort the memory line was read on: the native anchor reproduces 61.13 and
`dst_rank64` / `gated_dynamic` reproduce 51.56 / 53.32 exactly, so the two new arms below can be
dropped straight into that table.

`heterofold_r16` is the official HeteroFold map
(`heterofold_official_l3_to_q4_seed20260825/maps/llama3_to_qwen4/heterofold.pt`, `dc2ea778…`):
rank 16, epoch 1 of 4, validation loss 0.99146, trained on 1600 rows and validated on 400. It is
applied as-is, with no fine-tuning on any of these eight tasks. `recolor_c` is the recolor
correction checkpoint (`14a4c5f7…`). Both DST arms were fine-tuned. That asymmetry is the main
thing to keep in mind when reading the gap.

## Accuracy (%, primary metric per task)

| Task | Metric | native | heterofold_r16 | dst_rank64 | gated_dynamic | recolor_c |
|---|---|---:|---:|---:|---:|---:|
| arc | correct | 87.50 | 65.62 | 67.19 | 67.19 | 67.19 |
| hotpotqa | em | 56.25 | 29.69 | 45.31 | 46.88 | 29.69 |
| 2wiki | em | 32.81 | 17.19 | 23.44 | 25.00 | 9.38 |
| musique | em | 25.00 | 6.25 | 21.88 | 21.88 | 1.56 |
| sciq | correct | 100.00 | 98.44 | 98.44 | 100.00 | 98.44 |
| mmlu_pro | correct | 53.12 | 25.00 | 21.88 | 26.56 | 25.00 |
| hellaswag | correct | 45.31 | 60.94 | 60.94 | 64.06 | 57.81 |
| gsm8k | correct | 89.06 | 57.81 | 73.44 | 75.00 | 70.31 |
| **Macro** | | **61.13** | **45.12** | **51.56** | **53.32** | **44.92** |

Two things stand out. First, both new arms land within 0.2 pp of each other at the bottom
(45.12 and 44.92) — two very different mechanisms, same score. Second, every translator loses to
native, and the losses concentrate where the task needs retrieval rather than pattern matching:
musique em 25.00 → 6.25 for heterofold, mmlu_pro 53.12 → 25.00, hotpotqa 56.25 → 29.69.

## Paired against heterofold_r16 (pp, 64 paired rows)

| Task | native | dst_rank64 | gated_dynamic | recolor_c |
|---|---|---|---|---|
| arc | +21.88 [+10.94, +32.81] | +1.56 [−7.81, +12.50] | +1.56 [−7.81, +10.94] | +1.56 [+0.00, +4.69] |
| hotpotqa em | +26.56 [+15.62, +39.06] | **+15.62 [+3.12, +28.12]** | **+17.19 [+4.69, +31.25]** | +0.00 [−9.38, +9.38] |
| 2wiki em | +15.62 [+3.12, +29.69] | +6.25 [−7.81, +20.31] | +7.81 [−6.25, +21.88] | −7.81 [−18.75, +1.56] |
| musique em | +18.75 [+9.38, +29.69] | **+15.62 [+4.69, +26.56]** | **+15.62 [+4.69, +26.60]** | −4.69 [−10.94, +0.00] |
| sciq | +1.56 [+0.00, +4.69] | +0.00 [−4.69, +4.69] | +1.56 [+0.00, +4.69] | +0.00 [−4.69, +4.69] |
| mmlu_pro | +28.12 [+14.06, +42.19] | −3.12 [−17.19, +10.94] | +1.56 [−12.50, +15.66] | +0.00 [−9.38, +9.38] |
| hellaswag | −15.62 [−26.56, −4.69] | +0.00 [−9.38, +7.81] | +3.12 [−6.25, +12.50] | −3.12 [−10.94, +4.69] |
| gsm8k | +31.25 [+18.75, +43.75] | **+15.62 [+4.69, +26.56]** | **+17.19 [+4.69, +29.69]** | **+12.50 [+3.12, +21.88]** |

DST's advantage over heterofold is real but narrow: it clears zero on exactly three tasks
(hotpotqa, musique, gsm8k) and is inside the noise on the other five. recolor_c only clears zero
on gsm8k, and is *worse* than heterofold on 2wiki and musique.

Native is not uniformly ahead, which is worth remembering: on hellaswag every translator beats
native by 12–19 pp. Fine-tuning on the transfer path helps the 4-way choice and hurts everything
that needs to carry facts.

## Handoff cost (batch 2, 3 trials × 2 batches, seconds unless noted)

| Body tokens | Arm | Handoff p50 | p95 | Translation | Gather | Alignment | Peak (GiB) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4096 | heterofold_r16 | 1643 | 1686 | 729 | 587 | 66 | 11.69 |
| 4096 | dst_rank64_gated_dynamic | 1701 | 2084 | 503 | 592 | 396 | 11.67 |
| 16384 | heterofold_r16 | 6565 | 6815 | 3015 | 2645 | 255 | 15.06 |
| 16384 | dst_rank64_gated_dynamic | 6960 | 7345 | 2010 | 2535 | 1233 | 15.04 |
| 32768 | heterofold_r16 | 13371 | 14334 | 5940 | 4832 | 520 | 19.56 |
| 32768 | dst_rank64_gated_dynamic | 14228 | 15492 | 4006 | 5555 | 2526 | 19.54 |

HeteroFold wins on wall clock at every length while losing on translation at every length
(729 vs 503, 3015 vs 2010, 5940 vs 4006 ms). The rest of the pipeline reverses it: its alignment
is 5× cheaper (66/255/520 vs 396/1233/2526 ms), because a single fixed map has nothing to
realign per tap, and at 32k its gather is 0.7 s smaller. So the two arms pay in opposite places,
and at 32k the gap is 0.86 s in HeteroFold's favour — 6% of the handoff, against a 6.4 pp
accuracy deficit. Cheap in time, expensive in accuracy.

Peaks assume the Sender is unloaded during measurement, so they are 3–4 GiB below the parent
directory's numbers and are only comparable inside this table.

`latency_cache_parity.json` compares this reimplementation of the official map against the
official cache: 5.19e-5 key and 2.26e-5 value relative L2, i.e. the arm is faithful and the
accuracy gap is not an implementation artefact.

## Caveats

- `heterofold_r16` is a zero-shot general map; both DST arms carry task fine-tuning. This run
  cannot separate "rank 16 is too thin" from "nobody fine-tuned it".
- `tasks_manifest.json` records `cohort_role: historical development and regression`, and pins a
  separate `test_sha256`. So these 512 rows are the development cohort the DST configuration was
  selected on, while heterofold has never seen them in any form. The 6.4 pp macro gap therefore
  mixes architecture, fine-tuning and development-set optimism, and should not be quoted as a
  clean head-to-head. What survives is the direction and the size of the latency trade: the
  shipped DST path is 0.5–0.9 s slower per handoff and sits 6–8 pp higher on macro here.
