# QASPER Hub audit, native chat repair, and frozen Writer transfer

All runs use the same sixteen QASPER dev questions; no QASPER training. The
following follow-ups preserve the historical plaintext audit in `results/`.

| Repaired chat condition | Official Answer F1 (0–100) | EOS rate |
|---|---:|---:|
| Qwen complete native / native cache reuse | 44.99 / 44.99 | 100% / 100% |
| Frozen MCQ Stage-A / Stage-B | 1.33 / 2.87 | 0% / 6.25% |
| Frozen GSM8K Stage-A / Stage-B | 27.78 / 40.22 | 100% / 93.75% |

- `native_chat_fix.py`, `results_native_chat_v1/`: Qwen chat template, no-thinking,
  concise-answer instructions; full-prefill/cache generation agrees on 16/16.
  This is a bundled protocol repair, not a single-variable ablation. See
  `NATIVE_CHAT_RESULTS.md` for details.
- `writer_chat_transfer.py`, `results_chat_transfer_v1/`: four frozen old Writers,
  each model's own chat template, Full-Sync copy/drop on paper-body tokens only.
  Qwen wrapper/boundary tokens remain native; Question is processed natively.
  Two native pre-RoPE reconstruction smoke cases passed exact generation parity.
- Each result directory includes raw predictions and per-sample outputs; repaired
  runs also include config. Logs are in `logs/`. Greedy generation cap is 128.
- GSM8K Stage-B's 40.22 total F1 must not be interpreted as near-native evidence
  understanding: extractive F1 is 12.28 versus native 59.70, while unanswerable
  F1 is 100 versus native 75. These are small, type-balanced dev diagnostics.
- Dependencies are in sibling `lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234`.
  Model weights, Writer checkpoints, and KV caches are not included.

## Historical plaintext experiment 0

Server B only. No training or checkpoint updates. This is a new interface
diagnostic, not an official lm-eval task or a full QASPER benchmark score.

- Official QASPER v0.3 dev data, seed1234, sixteen questions (four per first
  annotation answer type), evidence512–3072 Qwen tokens, no truncation.
- Input: instruction + complete title/abstract/section text, then Question and
  Answer. No chat template, no few-shot examples, greedy generation128 tokens,
  EOS stopping. Tables/figures are not independently rendered.
- Full Native computes the complete input. Native Cache prefills the evidence
  once and continues the exact same token sequence with the boundary token and question. Original
  token positions and masks are preserved; tokenizer-boundary mismatch excludes
  a sample explicitly. Same-paper caches are reused on CPU without disk KV files.
- Compare first-answer logits, argmax, and entire generated token sequences.
  Small FP16 differences are reported rather than hidden.
- Evaluate raw stripped answer text with the supplied official Answer F1
  evaluator and all annotations. Evidence extraction/F1 is out of scope.
- Diagnose both frozen legacy MCQ and GSM8K Stage-A/B checkpoints. Translate all
  evidence tokens except native Qwen token0 and the last boundary token; Qwen
  processes the boundary token and question natively. The last boundary token is
  deferred to the reader so BPE merging with the separator cannot change the full
  token sequence. No text is discarded. Assert the target cache grid exactly.
- Old writers were not trained on this evidence-first protocol or QASPER: their
  scores cannot judge a newly trained Hub method. No test data used for selection.

Run: `/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u experiment.py`.
Outputs: results/manifest.json, per_sample.json, summary.json, predictions.
The script imports fixed modules/checkpoints from the sibling legacy reproduction
directory; it is not standalone without that dependency. Checkpoints not published.
