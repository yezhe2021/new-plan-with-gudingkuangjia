# Gemma to Qwen: Full-Sync DST plus frozen-base Rank64

| Task | Metric | Gemma Recolor+C | DST Stage1 | DST+Rank64 | Native |
|---|---|---:|---:|---:|---:|
| arc | correct | 32.81% | 31.25% | 57.81% | 87.50% |
| hotpotqa | em | 25.00% | 1.56% | 35.94% | 56.25% |
| hotpotqa | f1 | 35.52% | 1.95% | 53.97% | 68.52% |
| 2wiki | em | 6.25% | 3.12% | 15.62% | 32.81% |
| 2wiki | f1 | 21.79% | 5.56% | 28.38% | 43.45% |
| musique | em | 3.12% | 0.00% | 12.50% | 25.00% |
| musique | f1 | 7.24% | 1.04% | 16.51% | 33.07% |
| sciq | correct | 78.12% | 78.12% | 93.75% | 100.00% |
| mmlu_pro | correct | 15.62% | 14.06% | 10.94% | 53.12% |
| hellaswag | correct | 56.25% | 45.31% | 56.25% | 45.31% |
| gsm8k | correct | 70.31% | 54.69% | 73.44% | 89.06% |
| gsm8k | legacy_correct | 70.31% | 54.69% | 73.44% | 89.06% |

- Fresh Gemma3-4B -> Qwen3-4B DST Stage1; 34 Sender layers, 36 Receiver layers, 256 depth state. No Llama weights or Recolor weights initialize DST.
- Stage1 trains all DST parameters using Native raw KV NMSE; Stage2 freezes exactly its final epoch4 checkpoint and trains only 9,437,184 bias-free Rank64 GELU K/V residual parameters. No Memory stage.
- Same ARC/HotpotQA/GSM8K rows, 4 epochs and1536updates per stage, effective batch8, seed1234; Stage2 CE+.5full-vocabulary Native||Student KL with unchanged prefix schedule.
- RAW Gemma body with no added BOS; shared-unit-boundary Full-Sync Copy/Drop, not strict individual-target-token causality; audit every row.
- Gemma FP32 raw projection capture; Qwen FP16; residual before exactly one Qwen k_norm and RoPE. No native body or question input on student evaluation.
- Historical Gemma Recolor+C comparison shares task rows and scorers, but alignment and training budgets differ. This is not a recurrence-only ablation.
- Fixed final epoch4 checkpoint, validation diagnostic only, 8 tasks each64, GSM512 generation cap with strict and legacy scoring; no test checkpoint selection.
