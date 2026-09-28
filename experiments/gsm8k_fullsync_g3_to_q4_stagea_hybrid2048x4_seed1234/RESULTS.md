# Gemma3-4B → Qwen3-4B GSM8K Full-Sync results

Completed on server A on 2026-09-28. The two-question native-FP32 smoke and the
full pipeline both passed. This run uses fresh Gemma Stage-A training on 1,024
GSM8K train questions for two epochs, followed by a frozen Stage-A and a new
Residual64 Hybrid Stage-B trained on 2,048 train questions for four epochs
(1,024 optimizer steps). Validation ran after the final Stage-B epoch only.

On the held-out 128-question test subset, flexible answer extraction scored:

| Condition | Correct | Flexible accuracy | Strict accuracy |
|---|---:|---:|---:|
| Gemma Stage-A only | 0/128 | 0.00% | 0.00% |
| Gemma Stage-A + Hybrid Stage-B | 86/128 | 67.19% | 1/128 (0.78%) |

The paired Llama3.2-3B → Qwen3-4B reference scored 73/128 (57.03%) on the
same test IDs: 61 both correct, 25 Gemma-only, 12 Llama-only, 30 neither.
These are this experiment's generation/extraction metrics, not lm-evaluation-
harness scores. The large flexible-versus-strict gap means the answer parser
matters; do not interpret 67.19% as strict exact match.

Stage-A selected epoch 2/step 256 with validation representation loss 0.81349,
K cosine 0.93149, and V cosine 0.75701. Stage-B selected the predeclared final
epoch 4/step 1024; validation flexible accuracy was 63.28% (81/128), with
trajectory KL 0.13180.

Raw artifacts: [test summary](runs/test128_summary.json),
[per-sample generations](runs/test128_per_sample.jsonl),
[selection summary](runs/selection_summary.json),
[training log](logs/pipeline.log), and [configuration](config.json).
Model weights, checkpoints, generated KV caches, and temporary tensors are
intentionally excluded from GitHub.
