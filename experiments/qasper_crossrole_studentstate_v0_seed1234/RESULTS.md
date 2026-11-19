# Cross-Role Residuals and Student-State Training: V0, 25 Dev Questions, No Verdict

QASPER, frozen Stage-A512, established evidence/message/token alignment and post-`k_norm` pre-RoPE
protocol, no native evidence fallback and no truncation. This is the explicitly recommended V0
round, not a paper-scale sweep: A3/A4/B2/B3, external baselines and official test are listed as
subsequent gated experiments and are not claimed here.

Two independent blocks:

- **A** — rank-32 residuals (9,437,184 parameters each), 128 updates, lr 1e-4, 8-paper
  accumulation, original NMSE + token-vector cosine, three nearest source depth indices clipped with
  duplicates at boundaries, separate K/V/head/token branches, zero up-weights. `A_self` trains on the
  same-role pairing, `A_cross` on the cross-role pairing.
- **B** — fresh rank-64 `B_native_matched` and `B_student`, 4 epochs / 316 updates, same
  initialization, data, order and optimizer. `B_student` refreshes the actual student query/suffix
  state at every training query and detaches it. Loss = attention KL + joint projected-output NMSE +
  0.01 Stage-A anchor. Checkpoint selected on internal representation/functional loss over 16
  unused official train papers, never on dev answers.

Dev is the original 8-paper / 25-question split (22 answerable) and is **report-only**.

## A block — rank-32 residuals

| Arm | Answer F1 | Answerable F1 | Retention | Recon total | K NMSE | V NMSE |
|---|---:|---:|---:|---:|---:|---:|
| A0_original512 (reference) | 20.55 | 12.75 | 41.67 | 0.7001 | 0.0653 | 0.3555 |
| A_self | 18.44 | 11.87 | 41.67 | **0.6913** | **0.0624** | 0.3544 |
| A_cross | 18.44 | 11.87 | 41.67 | 0.6961 | 0.0650 | 0.3541 |

Both rank-32 variants reconstruct slightly better than the reference (total 0.6913 / 0.6961 against
0.7001) and both answer slightly worse (20.55 → 18.44). Same-role wins on K, cross-role wins
marginally on V, and the difference in total (0.0048) is far smaller than the gap between either and
the reference.

**`A_self` and `A_cross` produce identical predictions on all 25 dev questions.** Not similar —
identical strings, verified item by item, despite different residuals and different reconstruction
metrics. The dev split cannot separate same-role from cross-role at this scale; only the
representation numbers can, and there the ordering is K-by-self / V-by-cross, i.e. no clean winner.

## B block — rank-64 Stage B, native vs student query state

| Arm | Answer F1 | Answerable F1 | Retention | route_kl | output_nmse | anchor_nmse |
|---|---:|---:|---:|---:|---:|---:|
| A0_original512 (report-only) | 20.55 | 12.75 | 41.67 | 0.4947 | 0.1753 | 0.0000 |
| B0_original_nativeQ (report-only) | 30.70 | 24.29 | 58.33 | 0.3271 | 0.0804 | 0.0564 |
| B_native_matched | 33.53 | 24.47 | 58.33 | **0.3308** | **0.0752** | 0.0586 |
| B_student | **34.23** | **25.26** | 58.33 | 0.3328 | 0.0769 | 0.0602 |
| native_cache = native_full | 43.05 | 39.82 | 100.00 | — | — | — |
| query_only | 20.32 | 16.28 | 33.33 | — | — | — |

## Reading

The matched control is the point of this run, and it separates two things that the old reference
conflated. `B0_original_nativeQ` is 30.70 / 24.29 at route_kl 0.3271; the fresh matched pair both beat
it on answers (33.53 and 34.23) while being *worse* on route_kl (0.3308 / 0.3328). The old 43.05 /
30.70 line in the README is an external reference with a different split, output loss, anchor policy
and dev-based selection — it is not a control, and comparing against it would have credited the
matched retrain with a gain that is mostly protocol.

Within the matched pair, `B_student` beats `B_native_matched` by +0.70 F1 / +0.80 answerable F1, and
the representation metrics move the *other* way: route_kl 0.3308 → 0.3328, output_nmse 0.0752 →
0.0769, anchor_nmse 0.0586 → 0.0602. Refreshing the student's own query state during training
therefore does not improve the objective it is trained on. On 25 questions, one flipped answer moves
mean F1 by 4 points, so +0.70 is well inside a single question's worth of partial credit — this is
not evidence for the mechanism, and it should not be quoted as a gain.

Everything here is far from native: 34.23 against 43.05 F1, retention 58.33 % against 100 %, with
`query_only` at 20.32 confirming that the question alone carries little of the answer.

What V0 does establish is that the A/B machinery is live and gated: exact zero-residual identity,
self/cross routing, nonzero finite adapter gradients with no base-model gradient, actual student
attention replay (replay NMSE matches the reported output NMSE to 5 decimals), suffix/query detach
and a real miniature optimization step all pass. That is the claim this round supports — the
protocol works and the comparison is now well-defined — not that cross-role reconstruction or
student-state training helps.

Next rounds would need: more than 25 dev questions, the A3/A4 and B2/B3 variants, multiple seeds,
and official held-out test. None of those are in this run.
