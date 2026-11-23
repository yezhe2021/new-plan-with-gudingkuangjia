# A Closed-Form Wiener Correction on the Full-Sync Stage-A Values Fixes the Metric It Optimises and Nothing Else

The `fullsync_ab` writer scores 17.19 macro and answers 0 of 20 field questions, and the previous
entry in this family located the failure in the writer rather than the scorer. Its values are the
suspicious half: keys are largely a linear function of the sender body (probe R² 0.55) while values
are not (0.28), and the delivered value cache is the worse of the two (V NMSE 0.48 against K 0.22).

This experiment asks the cheapest possible version of the follow-up question: **with the writer
frozen, can a closed-form per-head correction of its value outputs recover anything?** No gradient
steps anywhere. The correction is a Wiener shrinkage per head and role,
`Σ (Σ + λR)⁻¹`, with `Σ` the native value covariance and `R` the measured error covariance — three
variants: low-rank prior with fixed noise, diagonal prior with bucketed noise, and low-rank prior
with bucketed noise. Rank 16, four noise buckets, twelve bucket edges, strengths
`{0.01, 0.03, 0.1, 0.3, 1.0}`.

| Split | Rows | Purpose |
|---|---:|---|
| native covariance | 64 | `Σ` per head and role |
| residual calibration | 64 | `R`, bucketed by the gate score |
| strength validation | 20 | pick λ |
| 8-task evaluation | 128 (16/task) | NLL + readout NMSE |
| choice evaluation | 64 (4 tasks) | accuracy |
| `herb` field cohort | 20 | exact / set_f1 / recall / capped |

All three designs selected **λ = 1.0, the top of the grid**, on all three criteria. There is no
interior optimum to be found: the correction always wants to be as aggressive as allowed. That is
already a warning about the validation split, not about the method.

## It improves the value readout and degrades everything else

128 rows, 8 tasks × 16. `value_readout_nmse` is the readout of native values from the corrected
cache; `value_nmse` is the cache itself; `full_readout_nmse` reads back the whole K/V block.

| Arm | value_readout_nmse | value_nmse | full_readout_nmse | NLL |
|---|---:|---:|---:|---:|
| mlp (frozen writer) | 0.5509 | 0.6168 | 0.06262 | 17.31 |
| + fixed_lowrank | **0.5080** | 0.6462 | 0.06567 | 18.95 |
| + adaptive_diagonal | **0.5076** | 0.6585 | 0.06569 | 19.10 |
| + adaptive_lowrank | **0.5126** | 0.6556 | 0.06605 | 18.95 |
| native | — | — | — | 12.47 |

Paired, 128 rows, corrected minus frozen:

| Variant | value_readout_nmse | value_nmse | full_readout_nmse | NLL |
|---|---|---|---|---|
| fixed_lowrank | −0.0429 [−0.0633, −0.0227] | +0.0295 [+0.0267, +0.0323] | +0.00305 [+0.00206, +0.00412] | +1.640 [+0.429, +2.827] |
| adaptive_diagonal | −0.0433 [−0.0629, −0.0240] | +0.0418 [+0.0391, +0.0446] | +0.00307 [+0.00212, +0.00411] | +1.786 [+0.549, +2.989] |
| adaptive_lowrank | −0.0383 [−0.0597, −0.0174] | +0.0388 [+0.0358, +0.0420] | +0.00343 [+0.00235, +0.00459] | +1.636 [+0.433, +2.813] |

Every interval excludes zero, and they point in opposite directions: the quantity the solve
optimises improves by 7–8 % relative, the cache error grows by 5–7 %, the full-block readout gets
worse by 5 %, and per-token NLL gets worse by 1.6–1.8 nats. The correction does exactly what it was
asked to do and moves the arm away from native behaviour on everything else.

**This is the second misleading proxy in two experiments.** The previous entry found that KV NMSE
against native does not predict accuracy; here `value_readout_nmse` does not either — it is a
least-squares objective that the closed-form solve minimises by construction, so a drop in it is
evidence that the algebra ran, not that the writer improved.

## Accuracy on the choice tasks is a wash

64 rows, 16 per task, `dst_choice` scorer.

| Task | mlp | fixed_lowrank | adaptive_diagonal | adaptive_lowrank | native |
|---|---:|---:|---:|---:|---:|
| arc | 43.75 | 37.50 | 37.50 | 37.50 | 87.50 |
| sciq | 56.25 | 43.75 | 56.25 | 43.75 | 100.00 |
| mmlu_pro | 31.25 | 25.00 | 37.50 | 37.50 | 62.50 |
| hellaswag | 25.00 | 56.25 | 50.00 | 50.00 | 43.75 |
| **All 64** | **39.06** | **40.63** | **45.31** | **42.19** | **73.44** |
| paired vs mlp | — | +1.56 [−12.50, +15.63] | +6.25 [−7.81, +21.88] | +3.13 [−10.94, +18.75] | — |

The macro is nominally up by 1.6–6.3 pp and **no interval excludes zero**. With 16 rows per task the
individual cells swing by 30 pp (hellaswag 25 → 56, sciq 56 → 44), so the ordering of the four arms
here is noise. NLL on the same rows moves the other way: 16.73 → 21.96 / 22.29 / 21.95, against
native 15.18.

## The field cohort changes its failure mode without answering a single question

| Arm | exact | set_f1 | recall | capped |
|---|---:|---:|---:|---:|
| native | 0.800 | 0.9373 | 0.938 | 0.00 |
| mlp (frozen writer) | 0.000 | 0.0000 | 0.000 | **0.80** |
| + fixed_lowrank | 0.000 | 0.0000 | 0.000 | **0.00** |
| + adaptive_diagonal | 0.000 | 0.0000 | 0.000 | **0.00** |
| + adaptive_lowrank | 0.000 | 0.0000 | 0.000 | **0.00** |

The frozen writer hits the token cap on 16 of 20 rows and emits nothing gradeable. After the
correction, **capped falls from 0.80 to 0.00 on all three variants** — the degenerate loop is gone,
generation terminates normally — and `recall` is still **0.000**, i.e. it terminates by saying
something that contains none of the required IDs. Paired: capped −0.80 [−0.95, −0.60], recall
0.00 [0.00, 0.00], exact 0.00 [0.00, 0.00].

That is worth recording separately from the accuracy table: **breaking the repetition loop and
producing a correct field answer are independent events.** A monitor that watches `capped` would
read this as a large improvement; the task metric says nothing changed. The two previous arms that
reduced `capped` (the Ridge at 0.40, the Rank-64 arm at 0.60) show the same decoupling, only less
cleanly.

## What the noise model contributed: nothing

The residual calibration does show the error growing with the gate score —
mean relative MSE by bucket 0.882 / 0.947 / 1.024 / 1.153, bucket occupancy 0.256 / 0.252 / 0.248 /
0.243 — which is exactly the structure that motivates prediction-dependent noise. But the bucketed
variances buy nothing measurable: `adaptive_lowrank` against `fixed_lowrank` is −0.005 NLL
[−0.036, +0.032], and the two differ by 0.0004 in `full_readout_nmse` [+0.0003, +0.0005]. On this
data a single per-head scalar is as good as four buckets plus regularisation.

## Caveats

- **λ sits on the grid boundary.** All three designs choose 1.0; the optimum is either at or beyond
  the end of the sweep, and λ was chosen on 20 rows. The reported arm is the best of the grid, not a
  validated setting.
- **This is an upper bound for this family, not for the writer.** The correction is linear,
  per-head, and training-free by construction. A trained writer is not bounded by this result.
- **16 rows per task.** Individual task cells are ±12 pp; only the 64- and 128-row aggregates carry
  signal, and those are the ones the intervals are computed on.
- **One seed.** `seed1234` throughout, single calibration split.
- **The arm is the archived writer.** `mlp` is the stage-A checkpoint pinned as
  `PINS["fullsync_ab"]["stage_a"]`, the same one that scores 17.19 macro as `historical_mlp_stage_a`
  in `dst_kvridge_fullbody_v1`. The numbers here are a second look at a writer already known to be
  broken, which is why the null result is informative about the *value path* and not about DST.
