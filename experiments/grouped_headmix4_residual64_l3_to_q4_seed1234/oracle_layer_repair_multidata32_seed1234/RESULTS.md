# Five-task Oracle layer repair: GSM8K skipped by user

Original experiment protocol/checkpoints unchanged. GSM8K omitted from execution and macro average.

| Arm | openbookqa | arc_challenge | hellaswag | mmlu_pro | sciq | Macro |
|---|---:|---:|---:|---:|---:|---:|
| translated | 65.62% | 59.38% | 37.50% | 28.12% | 93.75% | 56.88% |
| native | 68.75% | 84.38% | 31.25% | 46.88% | 96.88% | 65.62% |
| reject_all | 9.38% | 21.88% | 25.00% | 6.25% | 21.88% | 16.88% |
| g1_K | 62.50% | 62.50% | 21.88% | 25.00% | 100.00% | 54.37% |
| g1_V | 59.38% | 59.38% | 40.62% | 28.12% | 96.88% | 56.88% |
| g1_KV | 59.38% | 59.38% | 40.62% | 28.12% | 96.88% | 56.88% |
| g1_mask | 56.25% | 65.62% | 40.62% | 31.25% | 100.00% | 58.75% |
| g2_K | 53.12% | 62.50% | 37.50% | 25.00% | 96.88% | 55.00% |
| g2_V | 59.38% | 62.50% | 37.50% | 25.00% | 93.75% | 55.62% |
| g2_KV | 53.12% | 62.50% | 37.50% | 31.25% | 93.75% | 55.62% |
| g2_mask | 65.62% | 62.50% | 43.75% | 34.38% | 100.00% | 61.25% |
| g3_K | 56.25% | 59.38% | 40.62% | 37.50% | 93.75% | 57.50% |
| g3_V | 53.12% | 56.25% | 43.75% | 31.25% | 96.88% | 56.25% |
| g3_KV | 53.12% | 59.38% | 43.75% | 37.50% | 96.88% | 58.13% |
| g3_mask | 46.88% | 50.00% | 37.50% | 31.25% | 96.88% | 52.50% |
| g4_K | 68.75% | 71.88% | 40.62% | 37.50% | 100.00% | 63.75% |
| g4_V | 62.50% | 53.12% | 37.50% | 37.50% | 100.00% | 58.13% |
| g4_KV | 65.62% | 65.62% | 37.50% | 43.75% | 100.00% | 62.50% |
| g4_mask | 28.12% | 40.62% | 34.38% | 25.00% | 87.50% | 43.12% |
| g5_K | 68.75% | 78.12% | 43.75% | 37.50% | 93.75% | 64.38% |
| g5_V | 65.62% | 62.50% | 37.50% | 28.12% | 100.00% | 58.75% |
| g5_KV | 71.88% | 78.12% | 43.75% | 43.75% | 96.88% | 66.88% |
| g5_mask | 15.62% | 21.88% | 37.50% | 6.25% | 21.88% | 20.62% |
| g6_K | 59.38% | 62.50% | 34.38% | 25.00% | 96.88% | 55.62% |
| g6_V | 53.12% | 56.25% | 40.62% | 31.25% | 93.75% | 55.00% |
| g6_KV | 59.38% | 59.38% | 37.50% | 31.25% | 96.88% | 56.88% |
| g6_mask | 59.38% | 62.50% | 31.25% | 28.12% | 93.75% | 55.00% |

- Keep existing full-prompt Full-Sync transfer interface, byte-identical prompt, token grids, suffix, option scoring/normalization and numeric answer extraction. Freeze existing multi-task Stage-A and residual-only Stage-B; no Reader or training.
- Intervene on every translated Prompt body position after the immutable native anchor, including Question/options. This differs from the previous Hotpot evidence-only intervention scope and must not be conflated with it.
- Native and translated K/V are mixed only after cache assembly in physical receiver post-k_norm/post-RoPE coordinates at identical token/cache positions. No rotation is added during replacement.
- Six receiver groups correspond to human layers1-6,7-12,13-18,19-24,25-30,31-36. Each group has K-only,V-only,KV and attention-access mask. Three baselines: translated,native,all-transferred-body masked.27 configurations,32 questions/task,160 questions,4320 sample/configuration evaluations.
- Masking forbids direct attention to original transferred-body positions in selected layers during suffix/candidate/generation forwards. It does not zero values, remove positions or recompute the cached body. Native header/anchor and new suffix/generated states remain accessible; this is not a literal no-information baseline.
- Historical32/task are sampled using seed1234 before new interventions, without outcome/length filtering. These are discovery samples, not a fresh untouched test. Existing remaining IDs reserved but not evaluated. Original GSM8K subset has only32; any independent confirmation requires new disjoint examples.
- Stage-B training tasks: OBQA/ARC/GSM8K/MMLU-Pro; HellaSwag/SciQ not in Stage-B training. This is not proof of absence from Stage-A calibration or pretraining. MMLU-Pro keeps the original custom split.
- Oracle Native KV requires raw Prompt at the Receiver and is an offline diagnostic, not a deployable or prefill-free method. Group choices and confidence intervals across many configurations are exploratory; no formal claim from per-dataset best arms.
- Inspect per-task accuracy, paired intervals, rescued/harmed questions, replacement-versus-mask, generation caps and equal-weight macro accuracy. No automatic second-phase layer selection, confirmation or repair training is launched.
- GSM8K was skipped at the user request before formal evaluation. Five-task macro excludes GSM8K; it is not scored as zero. Original manifest/config retained, effective task override recorded in controls/skip_gsm8k.json.
