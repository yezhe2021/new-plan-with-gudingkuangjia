# Averaging Two Frozen Translators Beats Each of Them, and Still Misses Native

No training, no learned projection, no Reader. For every already-translated Qwen slot, layer and
head, the two independently trained branches are averaged in FP32 and cast once to the cache dtype:

```
K = (K_llama.float() + K_gemma.float()) / 2
V = (V_llama.float() + V_gemma.float()) / 2
```

Both branches produce `[36, T, 8, 128]` on identical Qwen token positions, so the mean is
element-wise with no concatenation along any axis. Qwen target RoPE is applied exactly once after
fusion; the native anchor KV is kept through the established Full-Sync interface. Everything is
frozen — no LoRA wrapper, no gradient step anywhere in this experiment.

Six fixed test subsets, unchanged prompts, candidates and the GSM greedy 4096-token cap. The
single-branch and native columns are reused after identity and alignment checks, so all five columns
score the same rows.

| Task | n | qwen_native | gemma_native | llama_ab | gemma_ab | equal50_fusion |
|---|---:|---:|---:|---:|---:|---:|
| openbookqa | 128 | 80.47 | 75.78 | 68.75 | 58.59 | 66.41 |
| arc_challenge | 128 | 86.72 | 78.91 | 64.06 | 55.47 | 63.28 |
| hellaswag | 64 | 35.94 | 32.81 | 34.38 | 31.25 | 35.94 |
| gsm8k | 32 | 96.88 | 93.75 | 31.25 | 68.75 | 62.50 |
| mmlu_pro | 128 | 40.62 | 29.69 | 25.78 | 28.91 | 28.91 |
| sciq | 64 | 89.06 | 89.06 | 87.50 | 81.25 | 92.19 |
| **Pooled** | 544 | **69.30** | **63.24** | **53.49** | **50.92** | **56.07** |

Paired item counts, fusion against each single branch (`fusion_only / single_only`):

| Task | vs llama_ab | vs gemma_ab |
|---|---|---|
| openbookqa | 5 / 8 | 14 / 4 |
| arc_challenge | 6 / 7 | 14 / 4 |
| hellaswag | 4 / 3 | 3 / 0 |
| gsm8k | 10 / 0 | 3 / 5 |
| mmlu_pro | 7 / 3 | 10 / 10 |
| sciq | 3 / 0 | 7 / 0 |

## Reading

Averaging is a real, if modest, gain over either branch alone: pooled 56.07 against 53.49 for
`llama_ab` (+2.58) and 50.92 for `gemma_ab` (+5.15). Against the Gemma branch the direction is
consistent — 51 items gained, 23 lost across 544 — and on sciq the fusion is the only arm that beats
Qwen native (92.19 vs 89.06).

Against the Llama branch the picture is mixed: it gains 35 and loses 21, but the pooled edge is
carried almost entirely by GSM8K (62.50 vs 31.25), where the Llama branch collapses and the average
inherits the Gemma branch's ability. That is not fusion doing work; it is the better parent showing
through.

What the mean cannot fix is the gap to the receiver's own native KV. 56.07 is 13.2 pp under
69.30, and no task closes it except sciq. Two independently trained translators carry different
errors, and averaging cancels some of the variance but none of the systematic bias relative to
native.

Caveats that limit how far this can be read:

- the two branches were trained on **different Stage-A data mixtures**; this reuses them as-is
  rather than controlling the mixture;
- MMLU-Pro is the existing custom split, not the released test set;
- this is a practical equal-mean test, not an equal-budget architectural ablation, and there is no
  prefill-free latency claim here — the Full-Sync accuracy interface only.
