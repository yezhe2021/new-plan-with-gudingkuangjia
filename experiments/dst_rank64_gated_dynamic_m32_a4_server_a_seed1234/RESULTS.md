# Independent Gated Dynamic Memory

| Task | Metric | Native | Rank64 | Static Keep-both | Dynamic Keep-both | Gated Static | Gated Dynamic |
|---|---|---:|---:|---:|---:|---:|---:|
| arc | correct | 87.50% | 67.19% | 67.19% | 70.31% | 68.75% | 67.19% |
| hotpotqa | em | 56.25% | 45.31% | 42.19% | 40.62% | 46.88% | 46.88% |
| hotpotqa | f1 | 68.52% | 58.85% | 56.93% | 55.68% | 60.00% | 60.00% |
| 2wiki | em | 32.81% | 23.44% | 21.88% | 20.31% | 25.00% | 25.00% |
| 2wiki | f1 | 43.45% | 32.60% | 30.83% | 33.48% | 34.31% | 34.56% |
| musique | em | 25.00% | 21.88% | 18.75% | 15.62% | 20.31% | 21.88% |
| musique | f1 | 33.07% | 31.63% | 28.19% | 24.90% | 30.35% | 31.58% |
| sciq | correct | 100.00% | 98.44% | 98.44% | 98.44% | 100.00% | 100.00% |
| mmlu_pro | correct | 53.12% | 21.88% | 25.00% | 25.00% | 21.88% | 26.56% |
| hellaswag | correct | 45.31% | 60.94% | 57.81% | 60.94% | 60.94% | 64.06% |
| gsm8k | correct | 89.06% | 73.44% | 73.44% | 76.56% | 71.88% | 75.00% |
| gsm8k | legacy_correct | 89.06% | 73.44% | 73.44% | 76.56% | 71.88% | 75.00% |
| **mean** | | **61.13** | **51.56** | **50.59** | **50.98** | **51.95** | **53.32** |

Gated Dynamic is the first arm on this pipeline to clear zero against the frozen
Rank64 translator: +1.76 [+0.20, +3.32] over all 512 test rows, 4,000 bootstrap
resamples, 13 rows recovered against 4 lost. Per task the lower bounds sit on
zero rather than above it — hotpotqa +1.56 [+0.00, +4.69], 2wiki +1.56 [+0.00,
+4.69], sciq +1.56 [+0.00, +4.69], mmlu_pro +4.69 [+0.00, +10.94] — while arc
and musique are flat at +0.00, hellaswag +3.12 [−3.12, +10.94], gsm8k +1.56
[−3.12, +6.25].

- Only the Gated Dynamic cell is trained here: a fresh M32+A4 memory bank with a
  separate GQA path before `o_proj`, 36 per-layer tanh gates initialised at zero,
  1,536 updates on task CE plus 0.5 full-KL, Rank64 frozen. All five other
  columns are recomputed in this run from the pinned manifests
  (`static_manifest`, `dynamic_manifest`, `gated_static_manifest`).
- Because the baselines are recomputed here rather than stitched, the run also
  replicates the B-side numbers on A's hardware: all 64 ARC rows match
  `dst_stage1_rank64_dynamic_memory_m32_a4_seed1234` bit for bit, so the +1.76 is
  not a machine difference.
- The gates it learned stay tiny: 36 values in [−0.0328, +0.0394], mean +0.0012,
  20/36 under 1% and 28/36 under 2%, none above 4%. The memory bank is therefore
  allowed to write at most ~4% of its output, which means the +1.76 pp cannot be
  read as "memory content helps" — a perturbation that small moves 9 net rows and
  the margin is exactly those rows. This is the best cell in the memory line
  (53.32, above gated static 51.95 and dynamic 50.98) and simultaneously the
  strongest evidence that the memory itself is not what carries it.
- Two readings are consistent with the data and cannot be separated at this
  sample size: either a 2–4% write from a trained bank genuinely helps retrieval
  tasks, or 1,536 updates with a fresh seed reshuffle a handful of rows. The
  per-task intervals all touching zero favour the second.
