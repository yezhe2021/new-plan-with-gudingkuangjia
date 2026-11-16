# Gated Dynamic joint continuation on fresh evaluation questions

| Task | Metric | A starting Gated Dynamic | D joint 384 | D minus A (points) |
|---|---|---:|---:|---:|
| arc | correct | 75.78% | 75.78% | +0.00 |
| hotpotqa | em | 45.31% | 46.09% | +0.78 |
| hotpotqa | f1 | 60.83% | 60.86% | +0.03 |
| 2wiki | em | 28.91% | 30.47% | +1.56 |
| 2wiki | f1 | 38.63% | 40.73% | +2.09 |
| musique | em | 17.19% | 17.19% | +0.00 |
| musique | f1 | 30.05% | 29.86% | -0.20 |
| sciq | correct | 95.31% | 95.31% | +0.00 |
| mmlu_pro | correct | 25.78% | 24.22% | -1.56 |
| hellaswag | correct | 50.78% | 50.78% | +0.00 |
| gsm8k | correct | 78.12% | 78.12% | +0.00 |
| gsm8k | legacy_correct | 78.12% | 78.12% | +0.00 |
| **mean (em where available)** | | **52.15** | **52.25** | **+0.10** |

A second 384-update joint round leaves the score where it found it. Over 1,024
fresh rows the run rescues 13 and breaks 12 — net one row — and every task
interval contains zero: hotpotqa em +0.78 [−2.34, +3.91], 2wiki em +1.56
[−2.34, +5.47], mmlu_pro −1.56 [−4.69, +1.56], the other five tasks exactly
+0.00. The artifact's `macro_average` (F1 for the three retrieval tasks,
accuracy elsewhere) reads 56.91 → 56.96, the same +0.05.

- Starts from the Gated Dynamic checkpoint of
  `dst_rank64_gated_dynamic_m32_a4_server_a_seed1234`: `gated_manifest_sha256`
  7d66db50… is that directory's `manifest.json` and `gated_checkpoint_sha256`
  4547b751… is its `runs/epoch4_step1536.pt`, both verified by hash on A and B.
  384 updates at 1e-5 on the Rank64 residual, 1e-4 on the memory and 1e-4 on the
  gates, epoch 5, task CE plus 0.5 full-KL.
- Evaluation is a **fresh cohort**: 128 rows per task, `evaluation_seed`
  20261115, drawn from the same sources but disjoint from the 64-per-task set the
  five-arm tables use. Levels therefore do not carry across the two tables — the
  same Gated Dynamic checkpoint reads 75.78 on ARC here and 67.19 there — only
  within-table differences do.
- The gates do not open. Diagnostics at steps 0 / 128 / 256 / 384 give mean
  tanh(gate) +0.00124 / +0.00104 / +0.00069 / +0.00089 with max |gate| 0.0394 →
  0.0436, i.e. 384 joint updates move them by 4e-4 in either direction.
- What does move is the residual path: translated-KV relative L2 goes 0 → 0.0061
  → 0.0083 → 0.0097, while the gradient reaching the Rank64 body stays at
  1.72 / 1.83 / 2.37 / 1.86 against 0.060 / 0.098 / 0.157 / 0.053 on the memory
  path. The objective spends its 384 updates fitting the residual it already
  had, which is consistent with the score not moving.
- Caveat for the record: this run has no Rank64 arm on the fresh cohort, so it
  does not re-test the +1.76 [+0.20, +3.32] that the Gated Dynamic arm opened
  against Rank64 on the old 64-per-task set. That replication is still missing,
  and it is the measurement that would decide whether the arm's margin is a
  cohort artefact.
- `failure.json` records one failed `prepare` at 12:55 on 11-15; the pipeline was
  retried and completed at 17:42. No evaluation data was affected.
