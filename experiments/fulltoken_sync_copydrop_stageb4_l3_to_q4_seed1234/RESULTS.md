# Full-Sync choice-KL Stage-B: four epochs

## Protocol

- Llama3.2-3B Sender to Qwen3-4B Receiver with the original ordered Full-Sync copy/drop Options K/V mapping.
- The selected Stage-A epoch-4 checkpoint from the original retraining run was reused and frozen. A fresh Residual64 Stage-B adapter was initialized with the original seed.
- Same balanced 3072 train / 384 validation / 384 held-out test examples from OpenBookQA, ARC-Challenge and MMLU-Pro. Effective batch 8; four epochs or 1536 optimizer steps; choice-only KL, learning rate 1e-4 and gradient clip 30.
- Validation chose the checkpoint. Test384 was evaluated once using that selection.

## Validation progression

| Stage-B epoch | Steps | Correct / 384 | Accuracy | Choice KL |
|---:|---:|---:|---:|---:|
| 1 | 384 | 226 | 58.85% | 0.70962 |
| **2** | **768** | **233** | **60.68%** | **0.69985** |
| 3 | 1152 | 228 | 59.38% | 0.68318 |
| 4 | 1536 | 233 | 60.68% | 0.74273 |

Epoch 2 was selected: it tied epoch 4 on validation accuracy and had lower choice KL. The selected checkpoint and test predictions reproduce the original two-epoch result; extending this particular training run did not improve the validation-selected model.

## Held-out test384

| Dataset | Frozen Stage-A | Selected Stage-B | Native Qwen Oracle |
|---|---:|---:|---:|
| ARC-Challenge | 80/128 (62.50%) | 104/128 (81.25%) | 115/128 (89.84%) |
| MMLU-Pro | 16/128 (12.50%) | 42/128 (32.81%) | 61/128 (47.66%) |
| OpenBookQA | 74/128 (57.81%) | 90/128 (70.31%) | 99/128 (77.34%) |
| **Total** | **170/384 (44.27%)** | **236/384 (61.46%)** | **275/384 (71.61%)** |

MMLU-Pro uses fixed disjoint slices of its local official test file for the internal train/validation/test protocol; its row here is an internal held-out result, not an official benchmark test score.

The repository directory contains the implementation, complete queue and pipeline logs (including the initial concurrent OOM attempt), all training steps, four validation candidates, aggregate test metrics and 384 per-sample predictions and choice logits. Checkpoints and K/V caches are excluded.
