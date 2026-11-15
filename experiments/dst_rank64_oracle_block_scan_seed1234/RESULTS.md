# Oracle KV Block Scan on Frozen DST + Rank64

| Task | Metric | Rank64 | Native | Fixed | Random | Per-question oracle |
|---|---|---:|---:|---:|---:|---:|
| arc | nll | 1.2189 | 18.4098 | 1.2679 | 1.2744 | 0.5583 |
| arc | correct | 67.19% | 87.50% | 76.56% | 70.31% | 92.19% |
| hotpotqa | nll | 1.1183 | 3.7059 | 0.9205 | 1.0921 | 0.4889 |
| hotpotqa | em | 45.31% | 56.25% | 50.00% | 48.44% | 68.75% |
| hotpotqa | f1 | 58.85% | 68.52% | 64.29% | 61.72% | 78.18% |

Paired over the 64 test rows of each task with 4,000 bootstrap resamples, all
against the unpatched Rank64 arm: ARC fixed +9.38 [+0.00, +18.75], random +3.12
[−3.12, +9.38], oracle +25.00 [+15.62, +35.94], native +20.31 [+9.38, +32.81];
HotpotQA fixed +4.69 [−6.25, +15.62], random +3.12 [+0.00, +7.81], oracle +23.44
[+14.06, +34.38], native +10.94 [−1.56, +23.44]. Pooled over the 128 rows the
fixed block is +7.03 [+0.00, +14.06].

- Candidate space is 24 blocks: 6 receiver-layer groups of 6 layers × 4 prompt
  sectors, applied to all 36 receiver layers. A patch copies the *native* model's
  K/V into one block, over one quarter of the content tokens, and leaves every
  other position on the frozen translator.
- `fixed` picks one block per task on 32 development rows and applies it to the
  test rows: ARC block 14, HotpotQA block 15. Both indices land in the same layer
  group, receiver layers 12–17, on different prompt sectors — the middle of the
  network, not the readout end.
- `random` draws one block per row; `random_expected` (a 1.2447 / 1.0890 NLL
  average over all 24 blocks) is what an uninformed pick is worth. `oracle` takes
  the block that minimises the gold NLL of that same test row, so it is an upper
  bound with test-set selection, not a score any method can reach.
- The gap between `fixed` and `random` is what a block *choice* transfers:
  6.25 pp on ARC (4 rows), 1.56 pp on HotpotQA (1 row). The rest of the oracle
  headroom, 15.6 pp and 18.8 pp, is per-question and did not transfer.
- Rank64 and the patched arms carry near-identical NLL on ARC while accuracy
  moves 9.4 pp: the patch changes which option wins, not how fluently the answer
  scores. The `native` NLL column is not on the same scale — 18.41 nats per token
  at 87.50% accuracy is internally inconsistent — so compare accuracies across
  arms, not the native NLL.
