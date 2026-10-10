# Strict multi-query audit V2

This follow-up implements the supplied review without retraining. Historical
plaintext and hybrid-chat runs remain unchanged. V2 is a distinct segmented
chat protocol, not a single-variable ablation of V1 and not a full benchmark.

## Sampling and immutable reuse

- QASPER official dev, seed 1234, 16 distinct papers, 3 distinct questions each.
- Full title/abstract/section text, 512–3072 receiver body tokens, no truncation.
- Paper sampling does not use answer correctness or force a type balance.
- One Sender capture per paper shared by both frozen Writer groups; one Stage-A
  mapping per group and one Stage-B residual application per group, per paper.
- Each of the six cache conditions is built once and read by all three queries.
  The Reader restores a fresh CUDA cache from immutable CPU tensors. Whole-cache
  hashes before and after the three reads must agree. Hashes and counts are saved.

## Strict state boundary

The target wrapper ends at `Paper:\n` and is computed without the paper. Paper
tokens are separately encoded into a fixed receiver grid. EVERY paper KV slot
is replaced; there is no native evidence or boundary fallback. Sender uses its
own wrapper and sees evidence only, never a Question. Native reference KV is
computed in the audit branch only and is not used to build Writer caches.

Reader inputs are immutable cache tensors and question/chat-tail IDs only. It
does not receive paper text, original paper IDs, or a full evidence Prompt.
This isolates the interface; it is NOT a formal privacy guarantee. The length
grid, model/tokenizer access at memory construction, and possible recoverability
of information from KV are not claimed to be private.

Segmented tokenization can differ at BPE boundaries from tokenizing one rendered
string. Complete Native and every cache condition use exactly the same segmented
sequence. This intentional change is recorded, never silently treated as V1.

## Conditions and measurements

- Complete Qwen Native, Native cache, pre-RoPE rebuilt Native cache.
- Frozen MCQ and GSM8K Writer Stage-A / Stage-B (four conditions).
- All 48 queries receive native first-logit, argmax, and complete generation
  comparisons. FP16 differences are recorded, not rounded away or called exact.
- Official raw stripped Answer F1, evaluator type scores, first-annotation type
  strata with counts, EOS rate, Unanswerable rate, generation cap, think markers
  and repeated 4-gram diagnostics. An always-Unanswerable baseline is included.
- Build costs and all three read costs are recorded separately and totaled per
  paper. Warm-model inference only; not model-load or disk-cache speed claims.
  Native reference/rebuild and integrity-audit overheads are kept separate.

`run_multiquery_strict.sh` runs CPU protocol tests, a one-paper/three-query smoke,
then the 16-paper audit. A code error or smoke first-token native parity failure
stops the pipeline. Results are versioned under `results_multiquery_strict_v2/`;
smoke results under `results_multiquery_strict_smoke_v2/`.

No Attention Calibration, new loss, Evidence-Only training or test tuning is
introduced. Those are deferred until this protocol audit has been reviewed.
