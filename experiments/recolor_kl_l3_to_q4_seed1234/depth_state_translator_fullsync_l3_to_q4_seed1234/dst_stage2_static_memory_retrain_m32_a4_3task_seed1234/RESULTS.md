# Fresh DST-specific Static Memory, M32+A4 all36

| Task | Metric | DST | DST+old Static | DST+retrained Static | Recolor+C+Static | Native |
|---|---|---:|---:|---:|---:|---:|
| arc | correct | 60.94% | 60.94% | 59.38% | 68.75% | 87.50% |
| hotpotqa | em | 35.94% | 40.62% | 34.38% | 43.75% | 56.25% |
| hotpotqa | f1 | 53.47% | 54.00% | 53.45% | 58.16% | 68.52% |
| 2wiki | em | 25.00% | 25.00% | 25.00% | 28.12% | 32.81% |
| 2wiki | f1 | 35.22% | 30.55% | 33.37% | 42.07% | 43.45% |
| musique | em | 17.19% | 14.06% | 17.19% | 20.31% | 25.00% |
| musique | f1 | 23.12% | 22.22% | 22.31% | 28.50% | 33.07% |
| sciq | correct | 96.88% | 96.88% | 98.44% | 100.00% | 100.00% |
| mmlu_pro | correct | 23.44% | 23.44% | 26.56% | 35.94% | 53.12% |
| hellaswag | correct | 50.00% | 50.00% | 46.88% | 60.94% | 45.31% |
| gsm8k | correct | 68.75% | 20.31% | 62.50% | 70.31% | 89.06% |
| gsm8k | legacy_correct | 68.75% | 20.31% | 62.50% | 70.31% | 89.06% |

- Only409600 Static Memory prompt/adapter parameters train. Fixed DST Stage2 final checkpoint1d46e308, Sender and Qwen freeze. Do not retrain DST or select a different DST epoch.
- Memory remains strictly input-independent: fixed public-header-only canonical Qwen construction, same32 learned prompt vectors and four Rank16 adapters after6/12/18/24. Rephase post-k_norm/post-RoPE Memory K when attaching, V unchanged. All36 answer layers retain DST body plus Memory. Full answer rationale CE+.5full-vocabulary KL; no task-specific objective changes.
- Fresh initialization, same3072train/192validation/512test rows and schedules as the old Static control, with existing DST Full-Sync indices added. Four epochs1536 updates, batch8, AdamW1e-4. Native/Student max64 rollout and8 KL positions; full gold CE is not truncated. Native prefixes reused read-only with hashes.
- CPU checks cover actual tiny36-layer Qwen, canonical input independence, cache invalidation and checkpoint gradients. GPU parity reproduces DST/Native/old Static endpoints before baseline reuse. Disposable longest-example and consecutive-batch updates do not warm-start formal training.
- Compare trained DST-specific Static Memory against frozen DST alone, old-Static zero-shot transfer, historical Recolor+C+Static and Native on the same cohorts. This tests adaptation of shared calibration to DST; it does not establish sample-conditioned information recovery.
- All memory conditions share32-slot/all36 Keep-both routing. DST-alone comparison also changes answer positions by32. Position/content effects remain unseparated; a separate masked-position control has not been run.
- Final epoch4/step1536 predeclared; validation diagnostic only. Eight64-question cohorts and one seed are exploratory, original labels (including disputed GSM952) retained. No large cache writes, model duplication, other-job resumption or automatic upload.
