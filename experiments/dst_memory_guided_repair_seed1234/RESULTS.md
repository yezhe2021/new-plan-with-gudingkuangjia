# Memory-guided KV Repair

| Task | Metric | Rank64 | Native | Fixed Oracle patch | Random Oracle patch | Per-question Oracle | Learned repair |
|---|---|---:|---:|---:|---:|---:|---:|
| arc | nll | 1.2189 | 18.4098 | 1.2679 | 1.2744 | 0.5583 | 1.4117 |
| arc | correct | 67.19% | 87.50% | 76.56% | 70.31% | 92.19% | 67.19% |
| hotpotqa | nll | 1.1183 | 3.7059 | 0.9205 | 1.0921 | 0.4889 | 1.0524 |
| hotpotqa | em | 45.31% | 56.25% | 50.00% | 48.44% | 68.75% | 45.31% |
| hotpotqa | f1 | 58.85% | 68.52% | 64.29% | 61.72% | 78.18% | 58.72% |

Learned repair minus Rank64: ARC 0.00 [−4.69, +4.69], HotpotQA 0.00 [+0.00,
+0.00]. Not a small gain — **zero rows changed**, on both tasks, while a single
fixed block over the same rows buys +9.38 and +4.69 and the per-question oracle
buys +25.00 and +23.44.

- The repair head is trained on 2,048 blocks labelled by the oracle scan
  (1,024 ARC + 1,024 HotpotQA), 768 steps at 3e-4, width 128, with
  lambda_rank 0.5 / lambda_kv 0.1 / lambda_kl 0.5, then frozen and evaluated on
  the 64 test rows per task. No test row appears in the labels.
- NLL does move, in opposite directions per task — ARC 1.2189 → 1.4117,
  HotpotQA 1.1183 → 1.0524 — and still flips no answer. On ARC the learned patch
  is worse than doing nothing on both NLL and accuracy.
- The four oracle columns are pinned to `dst_rank64_oracle_block_scan_seed1234`,
  which is this run's `reference`; the values are identical to that directory, so
  the comparison is over the same rows and the same frozen translator.
- Together with the scan this is a clean negative: the headroom is real
  (+25.0 / +23.4 pp), a hand-picked single block captures a quarter of it, a
  random block captures an eighth, and a trained head with 2,048 demonstrations
  of exactly which block to pick captures none of it. Whatever the repair should
  learn, it is not a function that the loss on gold tokens finds.
