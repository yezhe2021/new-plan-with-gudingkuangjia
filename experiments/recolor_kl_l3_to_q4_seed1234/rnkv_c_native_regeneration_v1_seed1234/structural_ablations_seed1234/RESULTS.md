# RN-KV structural ablations: same budgets and seven fixed task cohorts

| Task | Metric | C | Full RN-KV | Memory-only from scratch | Keep body+Memory | Native |
|---|---|---:|---:|---:|---:|---:|
| arc | accuracy | 67.19% | 64.06% | 51.56% | 67.19% | 87.50% |
| hotpotqa | em | 29.69% | 35.94% | 18.75% | 40.62% | 56.25% |
| hotpotqa | f1 | 46.53% | 49.80% | 32.99% | 56.50% | 68.52% |
| 2wiki | em | 9.38% | 21.88% | 12.50% | 29.69% | 32.81% |
| 2wiki | f1 | 22.05% | 35.32% | 19.28% | 42.91% | 43.45% |
| musique | em | 1.56% | 12.50% | 4.69% | 23.44% | 25.00% |
| musique | f1 | 7.64% | 17.85% | 8.78% | 31.88% | 33.07% |
| sciq | accuracy | 98.44% | 100.00% | 95.31% | 98.44% | 100.00% |
| mmlu_pro | accuracy | 25.00% | 26.56% | 18.75% | 28.12% | 53.12% |
| hellaswag | accuracy | 57.81% | 50.00% | 46.88% | 60.94% | 45.31% |

- Two structural ablations. Adapter-not-trained and memory-tokens-only refer to the same first arm. Its adapters are identity from initialization and never used; learned memory tokens are retrained from scratch, not copied from the jointly trained checkpoint.
- Keep-both changes only answer routing within19-30: translated body plus memory are visible. Layers1-18/31-36 retain translated body and block memory as before. Construction, four adapter placements,32 memory positions L..L+31 and answer offset L+32 are unchanged. No learned gate or output fusion.
- Both training arms use the identical original ARC1024+Hotpot1024,4epochs,batch8,1024 successful updates, lr1e-4 and task CE+0.5 full-vocabulary Native||Student KL. Sender, Recolor, all C corrections and Qwen weights freeze; gradients through Qwen to trainable memory parameters remain enabled.
- All seven fixed64-question test cohorts are reused without new selection or task tuning. C/full-RN-KV/Native baselines reuse immutable per-question results; enabled full-checkpoint scorer parity is checked. ARC/SciQ/MMLU use label likelihood; HellaSwag uses length-normalized ending likelihood; Hotpot/2Wiki/MuSiQue retain their established answer scorers.
- No comparison of absolute CE/KL scales alone establishes success. Report task-wise paired deltas, rescue/harm IDs, caps and answers. Previously inspected cohorts are architecture diagnostics, not a fresh independent confirmation set. No numerical-KV reconstruction claim or evidence-only interface claim.
- One GPU, detached sequential pipeline, corrected AMP overflow reject/retry (no skipped examples), initial/early checkpoints, resumable validation/evaluation and question/batch pause boundaries. Other paused experiments stay paused; no automatic upload or continuous monitoring.
