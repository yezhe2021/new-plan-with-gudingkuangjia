# The Grouped Translator Cares Which Sender It Gets; the Reader Does Not

Gemma-3-4B-IT as sender → Qwen3-4B as receiver, through the same grouped Stage A + rank-64
residual Stage B + rank-16 answer-KL Reader pipeline previously run with Llama3.2-3B as sender.
Everything frozen at inference; no Reader retraining was introduced for this comparison.

Architecture: 34 source layers / 4 source heads / head dim 256 into 36 target layers / 8 target
heads / head dim 128, mixed through 4 groups into hidden 1024, Stage A+B at 146,800,640 parameters.
The Reader is rank 16, alpha 16, on `q_proj` and `o_proj` of all 36 layers. Stage A uses the same
four-task train-512-each and validation-64-each rows as B and the Reader, with no answers or CoT in
the input.

Six fixed subsets, 544 rows total, identical prompts, candidates and scoring to the Llama run — so
the two senders are directly comparable column by column.

| Task | n | qwen_native | gemma_native | grouped_residual | grouped_residual_reader |
|---|---:|---:|---:|---:|---:|
| openbookqa | 128 | 80.47 | 75.78 | 58.59 | 64.06 |
| arc_challenge | 128 | 86.72 | 78.91 | 55.47 | 60.16 |
| hellaswag | 64 | 35.94 | 32.81 | 31.25 | 32.81 |
| gsm8k | 32 | 96.88 | 93.75 | 68.75 | 65.62 |
| mmlu_pro | 128 | 40.62 | 29.69 | 28.91 | 30.47 |
| sciq | 64 | 89.06 | 89.06 | 81.25 | 81.25 |
| **Pooled** | 544 | **69.30** | **63.24** | **50.92** | **53.68** |

## Same recipe, two senders

`grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234` is the Llama version of exactly
this pipeline on the same 544 rows:

| | grouped_residual | + Reader | Δ |
|---|---:|---:|---:|
| Llama3.2-3B sender | 53.49 | 56.25 | **+2.76** |
| Gemma-3-4B sender | 50.92 | 53.68 | **+2.76** |

The Reader contributes +2.76 pp in both runs, to the hundredth. That is the most stable number in
this whole line of work: a rank-16 answer-KL adapter on Q/K and O projections buys about the same
amount regardless of which sender fed the cache.

The per-task breakdown is not stable, and it is informative:

| Task | Llama residual → reader | Gemma residual → reader |
|---|---|---|
| openbookqa | 68.75 → 71.88 (+3.13) | 58.59 → 64.06 (+5.47) |
| arc_challenge | 64.06 → 67.19 (+3.13) | 55.47 → 60.16 (+4.69) |
| hellaswag | 34.38 → 34.38 (0) | 31.25 → 32.81 (+1.56) |
| gsm8k | 31.25 → 43.75 (**+12.50**) | 68.75 → 65.62 (**−3.13**) |
| mmlu_pro | 25.78 → 27.34 (+1.56) | 28.91 → 30.47 (+1.56) |
| sciq | 87.50 → 89.06 (+1.56) | 81.25 → 81.25 (0) |

GSM8K is where the two senders diverge. The Gemma branch arrives with generation already intact
(68.75, against 31.25 for Llama) and the Reader costs it 3.13; the Llama branch arrives broken and
the Reader recovers 12.50. Read together, the Reader is a repair mechanism that pays most where the
translation is worst — which is also why its *pooled* gain can be equal in two runs whose per-task
shapes are opposite.

## Reading

The translator is the sender-sensitive part. Gemma sender lands 2.57 pp under Llama sender pooled
(53.68 vs 56.25), and the gap is structural rather than uniform: −7.82 on openbookqa, −7.03 on
arc_challenge, −7.81 on sciq, but **+21.87 on gsm8k** (Reader arms, same 544 rows). A sender with
4 heads of dim 256 mapped into 8 heads of dim 128 loses on single-span multiple choice and retains
far more on multi-step generation.

Both senders sit far under Qwen native: 53.68 against 69.30. Gemma native itself is 63.24, so the
translation loses about 9.6 pp relative to the sender it came from, and the remaining 15.6 pp gap to
native is the receiver's own knowledge that no cache transfer recovers.

Caveats that limit the sender comparison:

- this is **not a sender-only ablation** — the Stage-A data mixture differs between the Llama and
  Gemma runs, along with layer count, head count and head dim;
- Gemma native is scored as a native likelihood evaluation (FP32 on V100, after the earlier FP16
  audit produced non-finite states), which is the same policy used for the transfer arms but is not
  an answer-generation accuracy;
- MMLU-Pro is the pre-existing custom split of the official test, not a leaderboard number;
- Full-Sync accuracy interface only — no prefill-free latency claim is made here.
