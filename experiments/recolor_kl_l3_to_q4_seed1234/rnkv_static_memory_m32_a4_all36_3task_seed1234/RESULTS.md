# Static versus dynamic Memory: Llama to Qwen

| Task | Metric | C | Dynamic Memory | Static Memory | Native |
|---|---|---:|---:|---:|---:|
| arc | accuracy | 67.19% | 70.31% | 68.75% | 87.50% |
| hotpotqa | em | 29.69% | 46.88% | 43.75% | 56.25% |
| hotpotqa | f1 | 46.53% | 61.09% | 58.16% | 68.52% |
| 2wiki | em | 9.38% | 32.81% | 28.12% | 32.81% |
| 2wiki | f1 | 22.05% | 43.95% | 42.07% | 43.45% |
| musique | em | 1.56% | 21.88% | 20.31% | 25.00% |
| musique | f1 | 7.64% | 27.79% | 28.50% | 33.07% |
| sciq | accuracy | 98.44% | 98.44% | 100.00% | 100.00% |
| mmlu_pro | accuracy | 25.00% | 21.88% | 35.94% | 53.12% |
| hellaswag | accuracy | 57.81% | 60.94% | 60.94% | 45.31% |
| gsm8k | strict final numeric | 70.31% | 75.00% | 70.31% | 89.06% |
| gsm8k | legacy numeric | 70.31% | 75.00% | 70.31% | 89.06% |

- Fresh fixed-memory training, not an inference-time masking ablation. Same fresh seed1234 M32+four Rank16 adapters (409600 parameters), same frozen Llama Recolor+C and Qwen, same byte-identical3072/192/512 rows, same1536 updates and prefix budgets as dynamic reference.
- Input independence concerns canonical KV content at fixed model parameters. Recipient length is used only for deterministic coordinate conversion. Canonical construction reads fixed public-header KV and learned P, never sample body or question. Position relocation does not rerun hidden-state construction at recipient length.
- Frozen Qwen remains differentiable during construction. Recompute after each optimizer update; evaluation caches canonical Memory only while parameter versions match. All32 slots retain gradients through rephasing and frozen Qwen. Public header to Memory distance is fixed rather than sample-length-dependent.
- All answer-stage body caches, positions, masks, scorers and caps are unchanged. This is a static-prefix versus sample-conditioned construction comparison; Native-state restoration and factual recovery are not implied.
- Gold task CE covers the full target. Native/Student rollouts max64, full-vocabulary KL max8 positions; CE+.5KL. Same canonical final checkpoint rule, no task-specific retuning.
- Eight tasks64 questions each, single seed exploratory cohorts previously inspected; keep original labels including known contradictory GSM8K row. Report paired differences to dynamic Memory, C and Native. Similar static/dynamic scores do not establish absence of information in dynamic Memory while translated body remains readable.
- No large cache or model duplication on server B; use existing native-prefix files read-only with hashes. Other paused experiments stay paused.
