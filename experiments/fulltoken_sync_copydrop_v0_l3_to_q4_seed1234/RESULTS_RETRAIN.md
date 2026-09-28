# Full-Sync Copy/Drop V0: fresh retraining

Same balanced 384-question held-out test set as the earlier frozen-Writer pilot.
Llama3.2-3B is the Sender; Qwen3-4B is the Receiver. The Receiver retains its
native Question KV and receives translated KV for every Options token on its
own token grid. Stage-A and Stage-B use newly initialized parameters rather
than continuing the earlier sparse-KV checkpoints.

| Condition | Correct / 384 | Accuracy |
|---|---:|---:|
| RawAnchor32/64, earlier Stage-B | 204 | 53.13% |
| Full-Sync, earlier frozen Stage-B | 222 | 57.81% |
| Full-Sync, freshly trained Stage-A | 170 | 44.27% |
| **Full-Sync, freshly trained Stage-B** | **236** | **61.46%** |
| Full-Sync native Qwen Oracle | 275 | 71.61% |

## Results by dataset

These are grouped from the existing held-out test predictions in
`runs/retrain/results/per_sample.jsonl`; no additional inference or training
was needed. Each dataset contributes 128 unique test questions. The Oracle
uses Qwen native KV at the same Full-Sync Options positions.

| Dataset | Native Qwen Oracle | Fresh Stage-A | Fresh Stage-B |
|---|---:|---:|---:|
| ARC-Challenge | 115/128 (89.84%) | 80/128 (62.50%) | 104/128 (81.25%) |
| MMLU-Pro | 61/128 (47.66%) | 16/128 (12.50%) | 42/128 (32.81%) |
| OpenBookQA | 99/128 (77.34%) | 74/128 (57.81%) | 90/128 (70.31%) |
| **Total** | **275/384 (71.61%)** | **170/384 (44.27%)** | **236/384 (61.46%)** |

Stage-B versus Oracle, counting gold-answer correctness on the *same* test
questions:

| Dataset | Both correct | Stage-B only | Oracle only | Neither correct |
|---|---:|---:|---:|---:|
| ARC-Challenge | 99 | 5 | 16 | 8 |
| MMLU-Pro | 38 | 4 | 23 | 63 |
| OpenBookQA | 81 | 9 | 18 | 20 |
| **Total** | **218** | **18** | **57** | **91** |

Fresh training used 3,072 train and 384 validation examples. Stage-A ran four
epochs (1,536 optimizer steps) with full-token KV reconstruction; Stage-B ran
two epochs (768 steps) with choice-only KL. Both used effective batch 8.
Checkpoint selection used validation only; test was evaluated at the end.
The selected checkpoints were Stage-A epoch 4 and Stage-B epoch 2.

KV caches and model checkpoints are not published in this GitHub report.
