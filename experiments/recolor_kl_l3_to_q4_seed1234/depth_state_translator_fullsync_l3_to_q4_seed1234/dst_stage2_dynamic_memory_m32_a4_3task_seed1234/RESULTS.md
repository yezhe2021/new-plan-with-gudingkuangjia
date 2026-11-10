# Fresh DST-specific Dynamic Memory, M32+A4 all36

| Task | Metric | DST | DST+trained Static | DST+trained Dynamic | Recolor+C+Static | Native |
|---|---|---:|---:|---:|---:|---:|
| arc | correct | 60.94% | 59.38% | 62.50% | 68.75% | 87.50% |
| hotpotqa | em | 35.94% | 34.38% | 40.62% | 43.75% | 56.25% |
| hotpotqa | f1 | 53.47% | 53.45% | 56.52% | 58.16% | 68.52% |
| 2wiki | em | 25.00% | 25.00% | 20.31% | 28.12% | 32.81% |
| 2wiki | f1 | 35.22% | 33.37% | 30.14% | 42.07% | 43.45% |
| musique | em | 17.19% | 17.19% | 17.19% | 20.31% | 25.00% |
| musique | f1 | 23.12% | 22.31% | 23.27% | 28.50% | 33.07% |
| sciq | correct | 96.88% | 98.44% | 98.44% | 100.00% | 100.00% |
| mmlu_pro | correct | 23.44% | 26.56% | 23.44% | 35.94% | 53.12% |
| hellaswag | correct | 50.00% | 46.88% | 50.00% | 60.94% | 45.31% |
| gsm8k | correct | 68.75% | 62.50% | 62.50% | 70.31% | 89.06% |
| gsm8k | legacy_correct | 68.75% | 62.50% | 62.50% | 70.31% | 89.06% |

- Only current-sample Memory construction changes relative to the DST-specific Static reference. Same frozen final DST Stage2 checkpoint1d46e308, Qwen/Sender,409600 trainable parameters, exact fresh initialization,3072/192/512 data, schedules,1536 updates and full CE+.5KL. No trained Static or dynamic Memory weights warm-start formal training.
- Dynamic Memory uses the existing RNKV full-translated-cache construction. The current-sample DST body is read by32 Memory queries. First-layer Memory projections precede body reading; later-layer Memory K/V can be sample-dependent. Adapters operate only after construction blocks6/12/18/24. This is input conditioning, not evidence of factual recovery by itself.
- The complete DST RAW Full-Sync body is unchanged; native public header only. Memory KV are generated natively at their true positions L..L+31, with k_norm/RoPE once. No static rephasing or per-input learned token alignment is introduced. Body remains readable in all36 answer layers; suffix starts L+32.
- Fixed final epoch4;validation diagnostic only. Four epochs,batch8,lr1e-4. Full gold rationale CE, Native/Student prefix caps64,max8 full-vocabulary KL locations. Prefix quotas4/4/0,3/3/2,2/2/4,2/2/4 as the Static reference. Same Native prefixes reused read-only with provenance.
- CPU checks prove sample dependence of later-layer Memory under fixed weights and same-length changed bodies, static-control independence, unchanged body, token masks and checkpoint gradients. GPU checks verify complete-body visibility, original dynamic-construction parity, longest samples and four consecutive effective batches before fresh formal training.
- Compare DST alone, same-budget trained DST Static Memory, new trained DST Dynamic Memory, Recolor+C+Static and Native, eight64-question cohorts with unchanged prompts/labels/scorers. QA64/GSM512 caps and both numeric metrics. Static-vs-dynamic answer positions/masks match; comparisons to DST alone still include32-position shifts.
- Single seed and previously inspected cohorts are exploratory. Native body prefill prohibited for the student; teacher/baseline only. No new large cache, model duplication, unrelated-job resumption or automatic upload.
