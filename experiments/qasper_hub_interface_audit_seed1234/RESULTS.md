# Experiment 0: QASPER evidence-prefix interface audit

Completed on server B. Sixteen official dev questions, four per first annotation
answer type; full paper text, evidence512–3072 Qwen tokens; seed1234. No training.
Plain-text instruction, no chat template or few-shot, greedy128-token generation,
EOS stopping. This is a diagnostic pilot, not a full benchmark result.

| Condition | Official Answer F1 (0–100 scale) | Reached128-token limit |
|---|---:|---:|
| Qwen complete prefill | 11.0410 | 16/16 |
| Qwen native evidence cache then native question | 11.0533 | 16/16 |
| Frozen MCQ Stage-A | 1.4497 | 16/16 |
| Frozen MCQ Stage-B | 1.4180 | 16/16 |
| Frozen GSM8K Stage-A | 1.7371 | 16/16 |
| Frozen GSM8K Stage-B | 5.7536 | 16/16 |

These are answer-token F1 scores, not exact-match accuracies. All annotations are
used by the supplied official qasper_evaluator.py. Evidence extraction is not
predicted or evaluated. Raw stripped generated text is used as the answer; no
custom post-hoc extraction is used to improve scores.

## Interface agreement

- Full input token sequence and positions are identical in both native paths.
  The final evidence boundary token is processed by the native reader along with
  the question; no text is omitted. Evidence caches are reusable by paper.
- FP16 first-answer argmax agreement:16/16.
- FP16 complete128-token output agreement:11/16 (68.75%).
- Maximum first-answer logit absolute difference:0.236328125.
- Mean Answer F1 differs by0.0124 points on the0–100 scale.
- Separate FP32 control on two discrepant questions: first-answer logit maximum
  errors7.4863e-5 and9.7036e-5; first32 generated tokens match exactly for both.
  This is NOT a complete128-token FP32 verification. Extending that control was
  attempted but script transfer was blocked by repeated SSH connection closures;
  do not claim the longer control passed.

## Interpretation

No catastrophic cache-layout failure is evident: both native paths have nearly
identical task scores and match initial decisions. Exact FP16 generation parity
is not achieved and must not be reported as fully passed. The FP32 short control
supports a rounding-related explanation but does not prove every long divergence.

Both full native and cached native hit the output limit on every question and
have low Answer F1. Inspect instruction/answer format and termination before
training a new writer; low native performance alone is not evidence of Hub error.
The historical writers have not been trained on QASPER or evidence-first native
question readout, so these zero-shot scores are migration diagnostics only.

## Files

Remote root:
`/hy-tmp/yezhe/异构模型/qasper_hub_interface_audit_seed1234`

Raw results remain in results/manifest.json, per_sample.json, summary.json,
precision_control.json and per-condition prediction JSON. Scripts and this report
are in the local project directory. Download of raw results to Windows also met
SSH connection closures; do not assume a local raw-result copy exists.
