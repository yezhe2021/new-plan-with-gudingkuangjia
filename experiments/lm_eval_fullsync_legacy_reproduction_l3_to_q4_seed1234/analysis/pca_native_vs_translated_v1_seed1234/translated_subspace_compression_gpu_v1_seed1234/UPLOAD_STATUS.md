# Scope of this stopped evaluation

This is the Native-KV intervention using PCA directions fitted from Native,
Stage-A, or Stage-A+B KV, **not** the HeteroFold descriptive PCA analysis.
All three projection conditions use the Native calibration mean.

- Completed: OpenBookQA 24, ARC-Challenge 24, GSM8K 24, HellaSwag 24;
  each sample has all 37 evaluation arms.
- MMLU-Pro was stopped at the user's request. Partial completed arms and sample
  outputs are preserved, but are not presented as a completed 14-sample result.
- `partial_summary.json` and `REPORT.md` summarize the four completed tasks.
- The existing report uses **Strict-match** for GSM8K. Both Strict-match and
  Flexible-extract scores and paired correctness statistics are preserved in
  `partial_summary.json`; do not mix their values. MCQ report uses acc_norm.
- This archive is the CPU-one-thread run. The earlier four-thread diagnostic
  attempt is not combined into these results.
- Scripts, frozen input manifest, calibration audit, numerical gates, raw
  generations/scores, compact metrics and logs are included.
- Model weights, full KV caches, PCA tensor bases and Python caches are omitted.
  The source server retains them; the fitting code is included.

Run location on server B:
`/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/translated_subspace_compression_gpu_v1_seed1234`

The run is stopped, not failed or fully complete. Uploading it does not restart it.
