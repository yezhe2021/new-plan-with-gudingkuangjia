# Repairing the Memory Bank Repairs Nothing: 29 pp of Long-Context Loss Sits Outside the 32 Slots

The gated dynamic memory in this family carries the document in 32 slots, and those slots are built
from the **translated** body, not the native one. That looked like the obvious place to attack the
long-context deficit: if the bank is assembled from a lossy KV cache, repair the bank.

This experiment tests that hypothesis twice, on the same frozen DST body.

- **Oracle swap.** Build the bank from the native prefill instead of the translated body, then answer
  exactly as before. If bank construction is the loss, this is the ceiling: what a perfect
  bank-repair could buy.
- **Trained recovery.** Four rank-16 residual adapters after layers 6, 12, 18 and 24 of the memory
  slots (327,680 parameters, output layer zero-initialised so the start state is **exactly** the
  parent bank — `exact_zero_adapter_bank: true`), trained for 1,536 updates to push the
  translated-bank hidden states toward the native ones.

The parent is `dst_rank64_gated_dynamic_m32_a4_server_a_seed1234`, frozen at `m32.pt 4547b751…`,
32 slots, 36 gates. Llama3.2-3B → Qwen3-4B, rank-64 body `aaae0362…`, answer body is the translated
rank-64 KV. Batch schedules and the parent manifest are asserted equal, so both arms see the same
data in the same order. Training objective is the parent's `CE + 0.5 · Full-KL` plus
`0.1 · relative hidden MSE` against the native-built bank at those four layers.

## The oracle ceiling is zero

192 rows, 64 per L-Eval source, nothing clipped (`truncation: false`), 16-token generation cap.
`sfiction` is scored with the original-document judgment (the sibling run's `loyalty_correct`);
`dst bank` here is bit-identical to that run's `gated_dynamic` arm on all three sources
(46.88 / 35.94 / 56.25), which is what makes the other columns readable.

| Source | Metric | native | dst bank | native-built bank (oracle) | trained recovery |
|---|---|---:|---:|---:|---:|
| quality | correct | 73.44 | 46.88 | 45.31 | 45.31 |
| coursera | correct | 67.19 | 35.94 | 37.50 | 35.94 |
| sfiction | loyalty_correct | 85.94 | 56.25 | 56.25 | 56.25 |
| **Mean** | | **75.52** | **46.36** | **46.36** | **45.83** |

Paired against the dst bank, 64 paired rows per source:

| Source | oracle Δ (pp) | 95% interval | recovery Δ (pp) | 95% interval |
|---|---:|---|---:|---|
| quality | −1.56 | [−4.69, +0.00] | −1.56 | [−4.69, +0.00] |
| coursera | +1.56 | [+0.00, +4.69] | +0.00 | [+0.00, +0.00] |
| sfiction | +0.00 | [+0.00, +0.00] | +0.00 | [+0.00, +0.00] |

A 29 pp gap to native, and the perfect bank moves the mean by **0.00**. It is not a wash in the
sense of "gains and losses cancelling over many rows": across 192 items the oracle flips **two**
(one up on coursera, one down on quality) and the trained adapters flip **one** (the same quality
row). Everything else is bit-identical output.

## The same story on the eight short tasks

512 rows, 64 per task, same test set as the rest of the family.

| Task | Metric | gated_dynamic (m32) | + recovery | Δ (pp) | 95% interval |
|---|---|---:|---:|---:|---|
| arc | correct | 67.19 | 70.31 | +3.13 | [+0.00, +7.81] |
| hotpotqa | em | 46.88 | 46.88 | +0.00 | [+0.00, +0.00] |
| hotpotqa | f1 | 60.00 | 60.00 | +0.00 | [+0.00, +0.00] |
| 2wiki | em | 25.00 | 25.00 | +0.00 | [+0.00, +0.00] |
| 2wiki | f1 | 34.56 | 34.49 | −0.07 | [−1.56, +1.34] |
| musique | em | 21.88 | 21.88 | +0.00 | [+0.00, +0.00] |
| musique | f1 | 31.58 | 32.38 | +0.80 | [−1.34, +3.75] |
| sciq | correct | 100.00 | 100.00 | +0.00 | [+0.00, +0.00] |
| mmlu_pro | correct | 26.56 | 26.56 | +0.00 | [+0.00, +0.00] |
| hellaswag | correct | 64.06 | 62.50 | −1.56 | [−4.69, +0.00] |
| gsm8k | correct | 75.00 | 76.56 | +1.56 | [+0.00, +4.69] |
| **Macro** | | **53.32** | **53.71** | **+0.39** | |

Four of 512 answers changed: +2 on arc, +1 on gsm8k, −1 on hellaswag. The macro moves +0.39 pp,
which is exactly two net items. Every other task is bit-identical, which is why seven of the eleven
intervals are degenerate `[0, 0]` — those are not "no significant difference", they are "no
different prediction on any of 64 rows".

## Why nothing moved

The adapters are not degenerate — their output-layer Frobenius norms end at 0.96 / 0.91 / 0.80 /
0.75 for layers 6 / 12 / 18 / 24, from exactly zero at initialisation, so the slots really are
being rewritten. What they are rewriting was already close:

| Check | Relative hidden MSE (4 layers, mean) |
|---|---:|
| zero adapter, longest training row | 0.0437 |
| after 1,536 updates, epoch 4 batch 384 | 0.0297 |

The translated-bank hidden states sit ~4.4% off the native ones in relative MSE, and training
carries that to ~3.0%. There was never much bank error to recover: the slots already agree with
the native slots to 96% of their energy. Recovering the last 1.4 points then changes two answers
in 192, because those answers were never riding on that residual.

Read the other way, this also bounds how much the bank contributes at all: substituting a bank
built from the *native* cache — a strictly better input — changes nothing measurable either. Both
readings retire the same hypothesis, that the long-context loss is a bank-construction loss.

## Training curve

| Epoch | loss | task_loss | full_kl |
|---|---:|---:|---:|
| 1 (step 384) | 1.0666 | 0.7050 | 0.7136 |
| 2 (768) | 1.0682 | 0.6936 | 0.7400 |
| 3 (1152) | 1.0708 | 0.6869 | 0.7580 |
| 4 (1536) | 1.0923 | 0.6946 | 0.7866 |

The reconstruction term pulls one way and the KL term the other: task cross-entropy improves
slightly (0.7050 → 0.6946) while Full-KL rises 10% (0.7136 → 0.7866), driven by arc
(1.29 → 1.50). At `lambda_hidden 0.1` against `lambda_kl 0.5` the hidden term is a weak pressure,
and what it buys in hidden agreement it pays for in divergence from the native distribution. The
end state is a slightly worse objective with a slightly better bank — and identical answers.

## Caveats

- 64 rows per source. Paired intervals are ±4.7 pp at best, so a real effect under ~5 pp is not
  excluded by the tables. The item counts are the tighter bound: 2 / 192 and 4 / 512 changed.
- `sfiction` is scored on the original-document judgment only, not the family's composite. Do not
  compare this column against `correct` from `long_qa_quality_leval_full64_seed1234`
  (60.16 for the same arm); compare against its `loyalty_correct` (56.25), which matches exactly.
- The hidden-MSE pair above is two single-point checks on **different** batches: `smoke/` records
  the zero-adapter value on the longest training row, `checks/latest_hidden_loss.json` records the
  last batch of epoch 4 and is overwritten every step. Treat the direction as reliable, the exact
  numbers as indicative. A per-epoch fixed-batch probe would have made this a real curve.
- `coursera` hits the 16-token cap on 7/64 rows at the dst bank, 4/64 at the oracle and 6/64 at
  recovery. Small, but it means a few rows are scored on a truncated answer.
- This tests bank repair only. The answer still reads the translated body, so it says nothing
  about repairing the body itself — which is where the remaining 29 pp has to live.
