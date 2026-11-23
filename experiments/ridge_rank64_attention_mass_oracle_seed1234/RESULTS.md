# Forcing the Native Attention Mass onto the Translated Body Costs 3.85 Nats, So Misallocated Body Mass Is Not the Deficit

The Ridge + Rank-64 arm (`dst_kvridge_fullbody_v1`) sits 7.8 macro points under native, and its
value cache is the obvious suspect (V NMSE 0.48, V probe R² 0.28). But there is a second reading of
the same deficit: if the translated keys are wrong, the reader's attention over the body is
*pointing at the wrong positions*, and no amount of value fidelity helps. This experiment tests that
reading directly, and it is the strongest available form of the test: it does not estimate the
attention mass, it **installs the native one**.

## The counterfactual

Frozen cache, no training, no estimator. For every query and every head, at all 36 receiver layers,
the body slice of the attention scores is shifted by

```
shift = logsumexp(native K · q · scale) − logsumexp(translated K · q · scale)
```

so the total probability mass placed on the body becomes *exactly* the native value, while the
within-body distribution and all value vectors stay translated. The correction is verified per row:
`max |realized − target|` mass is 1.25e-6 on the 128-row set, 1.13e-6 on the choice set, 1.43e-6 on
the field cohort. `native` here means the native-K reference computed inside the same forward pass,
so this is an oracle in the strict sense — nothing about it could be shipped.

The hypothesis is not trivially false: the translated cache really does misplace mass. Mean absolute
mass gap over all rows, heads and layers is **0.112** (0.109 on the choice set, 0.106 on the field
cohort) — roughly 11 pp of body probability in the wrong place on every query.

| Arm | What it is |
|---|---|
| `native` | native K/V cache (ceiling) |
| `translated` | Ridge front-end + pinned Rank-64 stage-B, unchanged |
| `mass_oracle` | translated, with body attention mass forced to the native value |

## Every metric gets worse, on every task

128 rows, 8 tasks × 16:

| Task | native NLL | translated NLL | mass_oracle NLL |
|---|---:|---:|---:|
| arc | 19.52 | 1.93 | 5.42 |
| hotpotqa | 1.61 | 0.50 | 0.75 |
| 2wiki | 8.28 | 2.13 | 3.92 |
| musique | 4.51 | 1.50 | 2.57 |
| sciq | 17.03 | 0.21 | 3.22 |
| mmlu_pro | 18.02 | 2.32 | 8.43 |
| hellaswag | 6.12 | 3.58 | 4.92 |
| gsm8k | 24.63 | 6.65 | 20.35 |
| **All 128** | **12.47** | **2.35** | **6.20** |

Paired over the 128 rows, `mass_oracle` against `translated`: NLL **−3.85** [−4.73, −3.04],
first-token NLL **−5.49** [−6.67, −4.45], where a negative number means the oracle is worse. Not one
task moves the other way. The oracle closes **none** of the gap — it opens a new one, of the same
order as the deficit it was supposed to explain.

Choice accuracy over the same 64 rows:

| Arm | correct | NLL | first-token NLL |
|---|---:|---:|---:|
| native | 73.44 | 15.18 | 19.46 |
| translated | 59.38 | 2.01 | 5.55 |
| mass_oracle | 57.81 | 5.50 | 10.47 |
| paired oracle − translated | −1.56 [−9.38, +4.69] | — | — |

The field cohort gives the same verdict with one exception worth keeping:

| Arm | NLL | first-token NLL |
|---|---:|---:|
| native | 0.505 | 0.0003 |
| translated | 0.659 | 0.182 |
| mass_oracle | 1.114 | **0.040** |
| paired oracle − translated | −0.455 [−0.631, −0.299] | **+0.142 [+0.030, +0.299]** |

`fraction_of_native_nll_gap_closed` is **−2.96**: instead of closing the 0.154-nat gap to native,
the oracle widens it threefold. The first token is the one place the oracle helps — 0.182 → 0.040,
against native 0.0003 — which says the mass error is real and concentrated at the first generated
token, and that fixing it buys a better *first* token and a worse *rest* of the sequence.

## Why the fix makes it worse: the values are the thing that is wrong

The translated arm's NLL is **five times lower than native** (2.35 against 12.47) while its accuracy
is 14 points *below* native (59.4 against 73.4). Token likelihood and choice accuracy point in
opposite directions on this arm, which is itself the most useful observation in the run: the reader
is confident on the gold continuation and still picks the wrong candidate.

Given that, an oracle that pulls the attention distribution *toward the native one* is being asked
to place native-like probability on translated value vectors. It does so, and the probability of the
gold tokens collapses — by 3.85 nats. The mass was never the binding constraint; putting it right
reveals that the values underneath it cannot carry the answer. This is consistent with the value-side
diagnostics in `dst_kvridge_fullbody_v1` (V NMSE 0.48, V probe R² 0.277 against K 0.552) and it
closes the alternative explanation: **the reader is not failing to find the body.**

## What this rules out, precisely

- It rules out any method whose only action is to change how much probability mass lands on the body
  — including estimators that would infer that mass from the sender side, since those can only be
  further from native than this oracle is.
- It does **not** rule out correcting *which* positions inside the body get attention. The oracle
  here holds the within-body distribution fixed and moves only the total, so a redistribution inside
  the body is untested.
- It does not test the field cohort's generated content, only likelihood; `herb` has 0/20 exact on
  every arm here, so accuracy is uninformative on that split.

## Caveats

- **Oracle, not a method.** The shift is computed from the native keys during the same forward pass.
  The result is an upper bound for mass correction, and it is a *negative* upper bound.
- **16 rows per task, 64 for accuracy.** The accuracy interval is wide (−9.4 to +4.7) and cannot
  distinguish the arms; the NLL intervals can, and they are large relative to the gap.
- **One seed**, `seed1234`, one sample of the test rows (`random.Random(1234)` over the pinned
  counts).
- **NLL is teacher-forced on gold tokens**, so "translated NLL 2.35" measures likelihood of the
  correct answer under a cache that scores 59 % on choices. The two are not comparable quantities
  and the mixture is why `fraction_of_native_nll_gap_closed` is null on the main splits and −2.96 on
  `herb` (where the denominator is positive).
- **`herb` first-token NLL for native is 0.0003**, i.e. essentially the correct token at
  probability 1. Ratios against it are meaningless; only the translated-versus-oracle comparison
  should be read.
