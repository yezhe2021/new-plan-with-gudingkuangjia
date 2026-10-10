# QASPER Evidence-KV Strong Baseline — small-scale pilot

No Test split tuning/evaluation. Protocol: independent evidence/question chat messages, full-render tokenization.
HeteroFold-style objective only; not the original HeteroFold architecture or mapping domain.

| Condition | Dev Answer F1 | Answerable-only F1 | EOS | Unanswerable prediction rate |
|---|---:|---:|---:|---:|
| native_cache | 43.05 | 39.82 | 100.00% | 0.00% |
| native_full | 43.05 | 39.82 | 100.00% | 0.00% |
| new_stage_a | 0.00 | 0.00 | 0.00% | 0.00% |
| new_stage_b_attention | 0.00 | 0.00 | 0.00% | 0.00% |
| old_gsm8k_stage_a | 28.00 | 18.18 | 100.00% | 16.00% |
| old_gsm8k_stage_b | 33.20 | 24.09 | 96.00% | 12.00% |
| query_only | 20.32 | 16.28 | 100.00% | 0.00% |
| warm_start_stage_a | 0.15 | 0.17 | 52.00% | 0.00% |
