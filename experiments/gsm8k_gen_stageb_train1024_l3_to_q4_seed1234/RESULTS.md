# Results

The formal pipeline completed on 2026-09-21 using the official GSM8K split.
The 1,024 training examples were selected only from `train.jsonl` by a
seed-1234 shuffle. Evaluation used the fixed first 32 rows of the untouched
official `test.jsonl`; there was no validation set or test-based checkpoint
selection.

## Training

| Metric | Value |
|---|---:|
| Train examples | 1,024 |
| Epochs | 2 |
| Optimizer steps | 256 |
| Effective batch | 8 |
| Initial answer-token CE | 5.8806 |
| Final answer-token CE | 0.5074 |

## Free-generation evaluation

| Condition | Split | Exact | Contains gold | Mean generated tokens |
|---|---|---:|---:|---:|
| Qwen full native | test32 | 12/32 (37.50%) | 65.625% | 211.59 |
| Qwen native-cache oracle | test32 | 12/32 (37.50%) | 65.625% | 211.59 |
| Frozen Stage-A | test32 | 0/32 (0.00%) | 0.00% | 308.53 |
| Old MCQ Stage-B | test32 | 0/32 (0.00%) | 0.00% | 252.53 |
| New Generation Stage-B | test32 | 4/32 (12.50%) | 25.00% | 124.81 |
| New Generation Stage-B | train32 | 9/32 (28.125%) | 43.75% | 110.72 |

The generation-specific objective therefore recovered non-zero held-out GSM8K
performance from the same frozen Stage-A base, while retaining a substantial
gap to the native Qwen receiver. The train/test difference also shows that the
pilot learned the task but did not fully generalize at this training scale.

## Protocol audits

- Native full-prefill and native-cache teacher forcing had identical top-1
  predictions at every answer position for all three audit examples.
- FP16 full-prefill/cache CE differences were 0.0011--0.0035.
- A zero-initialized residual reproduced Stage-A exactly (`max error = 0`).
- 576 adapter parameter tensors received finite non-zero gradients; frozen
  Qwen parameters receiving gradients: 0.
- The old MCQ Stage-B was independently reproduced at 0/32 under the new
  evaluator before formal training.

Checkpoints are intentionally omitted. The repository includes scripts,
configuration, manifests, logs, optimizer-step metrics, summaries, and all
per-sample generations needed to inspect the run.
