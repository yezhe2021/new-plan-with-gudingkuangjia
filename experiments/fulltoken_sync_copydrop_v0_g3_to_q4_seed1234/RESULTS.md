# Gemma3-4B → Qwen3-4B Full-Sync retraining results

This experiment uses the same held-out test IDs and Qwen Receiver protocol as
the Llama3.2-3B → Qwen3-4B Full-Sync experiment. It contains 128 examples from
each of ARC-Challenge, MMLU-Pro and OpenBookQA. Qwen retains its native Question
KV and receives translated KV for every Options token on its own token grid.

## Overall results

| Condition | Correct / 384 | Accuracy | Oracle agreement | Choice KL |
|---|---:|---:|---:|---:|
| Fresh Full-Sync Stage-A | 158 | 41.15% | 42.19% | 1.5749 |
| **Fresh Full-Sync Stage-B** | **245** | **63.80%** | **69.27%** | **0.6044** |
| Native Qwen Full-Sync Oracle | 275 | 71.61% | 100% | 0 |

Stage-A selected epoch 2 (step 768) by validation accuracy. Stage-B selected
epoch 2 (step 768). The native Qwen Oracle predictions have zero mismatches
against the earlier Llama Full-Sync experiment on the same 384 test IDs.

## Results by dataset

| Dataset | Native Qwen Oracle | Fresh Stage-A | Fresh Stage-B |
|---|---:|---:|---:|
| ARC-Challenge | 115/128 (89.84%) | 66/128 (51.56%) | 105/128 (82.03%) |
| MMLU-Pro | 61/128 (47.66%) | 12/128 (9.38%) | 41/128 (32.03%) |
| OpenBookQA | 99/128 (77.34%) | 80/128 (62.50%) | 99/128 (77.34%) |
| **Total** | **275/384 (71.61%)** | **158/384 (41.15%)** | **245/384 (63.80%)** |

## Stage-B versus Oracle paired correctness

| Dataset | Both correct | Stage-B only | Oracle only | Neither correct |
|---|---:|---:|---:|---:|
| ARC-Challenge | 102 | 3 | 13 | 10 |
| MMLU-Pro | 32 | 9 | 29 | 58 |
| OpenBookQA | 85 | 14 | 14 | 15 |
| **Total** | **219** | **26** | **56** | **83** |

## Training protocol

- Balanced, leak-controlled 3,072/384/384 train/validation/test examples.
- Full34 HeadMix256 Writer, independently mapped K/V, bias-free.
- Stage-A: four epochs, 1,536 optimizer steps, full-token KV reconstruction.
- Stage-B: frozen Stage-A plus Residual64, two epochs, 768 optimizer steps,
  choice-only KL.
- Effective batch 8; Stage-A LR 1e-3; Stage-B LR 1e-4; clip 30.
- Checkpoint selection used validation only; test was evaluated once afterward.
- No KV cache was persisted. Checkpoint `.pt` files are not published.

Machine-readable predictions are provided both as one 384-record JSONL and as
three per-dataset 128-record JSONL files. Every record contains gold choice,
Stage-A/Stage-B prediction, native Oracle prediction, token count and logits.
