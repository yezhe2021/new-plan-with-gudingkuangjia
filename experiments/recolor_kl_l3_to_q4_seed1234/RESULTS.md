# Recolor-KL fixed-Recolor Stage-2 comparison

| Group | ARC accuracy | HellaSwag accuracy | GSM8K accuracy | HotpotQA EM | HotpotQA F1 |
|---|---:|---:|---:|---:|---:|
| native | 86.72% | 35.94% | 96.88% | 50.00% | 62.36% |
| A | 63.28% | 42.19% | 68.75% | 19.14% | 33.46% |
| B | 64.84% | 39.06% | 65.62% | 33.20% | 45.86% |
| C | 65.62% | 40.62% | 65.62% | 31.25% | 43.55% |
| D | 63.28% | 43.75% | 84.38% | 18.95% | 35.38% |
| E | 63.28% | 45.31% | 78.12% | 19.34% | 36.76% |

- Frozen exact same existing Recolor (fitted only on1600 train prompts), fixed raw-Sender capture/token alignment/depth stencil. No Grouped Stage-A, Full-Sync, old Stage-B weights or Reader modules are imported.
- All residuals live in pre-k_norm/pre-RoPE Qwen1024-dimensional KV coordinates. Use one shared official finalize_layer for K normalization/RoPE in both calibration and actual KV inference; native chat-header cache only. LLMs FP16 on V100; Recolor constants and corrections FP32, base outputs FP16 as in official ExampleBuilder.
- B/C use original attention KL + key-induced output MSE(nativeV fixed) + value output NMSE(detached routing) formula and native question-token probes. A common finalize_layer removes extra train/deployment rounding differences; an official-objective parity check records the small FP16 operator-order difference.
- C/D linear residual structure/rank/initial factors and optimizer/data/order/budget match. Loss type, supervision positions, and D online prefix exposure differ together; this is not a loss-only same-position ablation.
- D/E use full-vocabulary Native||Student KL, max64-token prefix/max8 scored positions, epoch1 Native and epochs2-4 per-batch balanced Native/Student prefixes. No gold answer loss. E differs by GELU activation with same rank64 factor shapes and initialization.
- All four trained arms use4epochs,800updates,batch8,AdamWlr3e-4,wd1e-4,clip1. Fixed final checkpoint is primary; original HeteroFold best-validation selection is retained only as diagnostic and historical folded checkpoint is separately labeled.
- No residual folding in primary inference for any trained arm, including linear ones: use the same explicit correction/finalize path as training; E is nonlinear and cannot be folded into a single affine map. Track folded-vs-explicit linear discrepancy as a diagnostic, not silently change the test path.
- Calibration1600train/400validation are exactly original Open-R1problem-only+HotpotQAtrainfull-context rows,800+800train/200+200validation. TestARC128/HellaSwag64/GSM8K32 keep original shared sample IDs; HotpotQA512 fixed distractor-dev questions have no ID or normalized-question overlap with any calibration row.
- HotpotQA test is independent of this Stage-2 optimization/tuning but has been evaluated in earlier research; this is not an untouched first-use benchmark. All passages preserved, no support-paragraph filtering/truncation. Answer EM/F1 only.
- Native prompt KV is permitted as training/validation teacher only. Student test receives only Recolor Sender states plus public Receiver header, then original assistant suffix; no native body prefill or native Question hidden states are used in student inputs.
- On-demand raw captures, no corpus-sized KV/attention cache. Reuse only token-prefix traces under this experiment manifest. All validation diagnostics include attention/output metrics and full-vocabulary same-prefix KL to check objective-task mismatch.
- Previously paused translated+native fusion experiment remains paused and will not resume automatically.
