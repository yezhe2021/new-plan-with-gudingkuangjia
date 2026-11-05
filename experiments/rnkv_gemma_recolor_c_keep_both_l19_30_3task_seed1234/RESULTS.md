# Gemma Recolor+C, three-task Keep-both, Memory readable19-30

| Task | Metric | Gemma Recolor+C | +Memory19-30 | Qwen Native |
|---|---|---:|---:|---:|
| arc | correct | 32.81% | 43.75% | 87.50% |
| hotpotqa | em | 25.00% | 35.94% | 56.25% |
| hotpotqa | f1 | 35.52% | 50.80% | 68.52% |
| 2wiki | em | 6.25% | 21.88% | 32.81% |
| 2wiki | f1 | 21.79% | 33.70% | 43.45% |
| musique | em | 3.12% | 6.25% | 25.00% |
| musique | f1 | 7.24% | 14.87% | 33.07% |
| sciq | correct | 78.12% | 93.75% | 100.00% |
| mmlu_pro | correct | 15.62% | 17.19% | 53.12% |
| hellaswag | correct | 56.25% | 53.12% | 45.31% |
| gsm8k | correct | 70.31% | 68.75% | 89.06% |
| gsm8k | legacy_correct | 70.31% | 68.75% | 89.06% |

- Redo ONLY19-30 Keep-both after all Llama mechanism diagnostics finish. Prior Gemma all36 remains held.
- Build a NEW Gemma RAW HeteroFold Recolor from the same1600 train Open-R1+Hotpot calibration prompts and equal effective token weighting, span1/stride4, then train fresh linearR64 C with Attention+Output objective for4epochs/800 updates on1600train/400val. No old grouped A/B or old Reader weights, no old Full-Sync/chat anchor interface.
- Recolor fits pre-k_norm/pre-RoPE source/receiver projections. C residuals are pre-Qwen-k_norm/pre-RoPE. Apply Qwen k_norm and RoPE once at cache assembly. Native header only, no native first-content-token anchor.
- Gemma forward uses FP32 due prior FP16 nonfinite audit; receiver isFP16. Stream teacher attention targets to host and use one layer at a time without changing query positions/objective. Sender,receiver,Recolor freeze during C calibration. Fix final epoch4/step800, no test selection.
- After C completes, freeze Recolor+C and train fresh32 Memory Tokens/fourRank16 adapters,409600 parameters. ARC/Hotpot/GSM1024each,4epochs,1536 updates,batch8,lr1e-4,full gold CE+0.5full-vocab KL,original mixed-prefix schedule. Answer retains translated body everywhere; Memory readable only19-30.
- Copy canonical Llama3-task train/validation/test cohorts byte-for-byte; old grouped Gemma calibration data is not used, so old grouped-model contamination replacements are not needed. Exclude Recolor/C calibration questions from downstream train/validation/test.
- Evaluate8tasks64each under same label/ending scoring,QA64/GSM512 caps and both numeric extraction metrics. Recompute NEW Gemma Recolor+C baseline; reuse pinned Native only after parity checks. This matches translator family/recipe to Llama, while chosen Memory access19-30 and FP32 Gemma adaptation must still be disclosed in pair comparisons.
- Optional bounded40GiB exact FP16 Recolor-output cache; misses always compute source normally. No sample skipping, truncation, weight quantization, source-interface substitution or reuse of old grouped-model caches. Cache scope includes learned Recolor hash.
- All training phases save initial/early/periodic/epoch checkpoints and retry overflowing identical batches with RNG restored. Single GPU, detached dependency gate, pause/failure controls; no automatic upload or continual monitoring.
