# The Ridge and Recolor Translators Share Their Residual Subspace Almost Exactly, and That Shared Residual Still Cannot Be Predicted From the Input

The previous experiments in this family treat the transfer residual as noise to be averaged out by
more calibration data. This one asks whether it is instead a *structure*: if two very different
closed-form translators leave essentially the same residual, that residual is a property of the
transfer problem rather than of the estimator, and it should be learnable once and applied
everywhere.

It is shared. It is not learnable. The two halves separate cleanly.

## Setup

Legacy Full-Sync alignment, 128-token bodies, 36 receiver layers × 8 heads evaluated, keys taken in
`post-k_norm, pre-RoPE` coordinates. Documents come from the calibration stream and are split **by
document before any fitting**:

| Split | Documents | Used for |
|---|---:|---|
| base fit | 80 | the Ridge / Recolor map `x → y` |
| residual fit | 64 | the residual probes |
| test | 48 | everything reported |

Rank 16, λ = 0.01, and 128 random quadratic features per head for the nonlinear probes. The 32
documents that selected the layer taps are excluded, as are none by alignment failure
(`skipped_alignment` is empty). 288 head-layer measurements per role, but only 48 independent
documents, so all intervals are over documents.

Every number below is a **relative energy**: `‖residual‖² / ‖y‖²` per document-head, lower is
better. `base` is the uncorrected map. The probes never see the test split.

## The residual is shared: the two estimators agree on its directions

Average squared canonical correlation between the rank-16 residual subspaces of the Ridge and the
Recolor map, per layer, then averaged:

| Role | mean | median | p10 | p90 | random baseline |
|---|---:|---:|---:|---:|---:|
| key | **0.916** | 0.920 | 0.848 | 0.975 | 0.125 |
| value | **0.807** | 0.816 | 0.713 | 0.890 | 0.125 |

A random rank-16 subspace in 128 dimensions sits at 0.125 by construction. At 0.916 the two
estimators' key residuals are nearly the *same* subspace — mean |cos| ≈ 0.96 — and at 0.807 the
value residuals are not far behind (≈ 0.90). Recolor is an optimal-transport-style rotation fitted
on second moments, Ridge is a least-squares solve; they agree about where the error lives.

So the first half of the hypothesis holds, and it is the strongest version of it: the residual is
not estimator-specific.

## It is not predictable from the body

If the shared residual were a function of the input, a rank-16 projection of a probe fitted on 64
documents should remove some of it. It does not — for the arm that matters, it makes things worse.

| Role | Metric | Ridge | Recolor | Δ vs base, Ridge | Δ vs base, Recolor |
|---|---|---:|---:|---|---|
| key | base | 0.1575 | 0.2015 | — | — |
| key | mean only | 0.1576 | 0.2016 | +0.0001 | +0.0001 |
| key | shared linear | 0.1625 | 0.1887 | +0.0050 | −0.0128 |
| key | shared quadratic | 0.1632 | 0.1894 | **+0.0057** [+0.0052, +0.0061] | −0.0120 [−0.0126, −0.0116] |
| key | own linear | 0.1594 | 0.1847 | +0.0019 [+0.0016, +0.0023] | −0.0168 [−0.0175, −0.0162] |
| key | own quadratic | 0.1601 | 0.1854 | +0.0026 [+0.0022, +0.0030] | −0.0161 [−0.0168, −0.0155] |
| value | base | 0.6132 | 0.8618 | — | — |
| value | mean only | 0.6135 | 0.8622 | +0.0003 | +0.0004 |
| value | shared linear | 0.6291 | 0.8132 | +0.0159 | −0.0486 |
| value | shared quadratic | 0.6311 | 0.8153 | **+0.0179** [+0.0167, +0.0192] | −0.0465 [−0.0480, −0.0449] |
| value | own linear | 0.6192 | 0.7975 | +0.0059 [+0.0049, +0.0070] | −0.0643 [−0.0665, −0.0623] |
| value | own quadratic | 0.6213 | 0.7995 | +0.0080 [+0.0069, +0.0092] | −0.0623 [−0.0644, −0.0602] |

Read the two Δ columns differently:

- **On the Ridge arm, every probe is a regression.** Shared quadratic costs +0.0057 on keys and
  +0.0179 on values, with intervals that exclude zero. Removing the per-head mean does nothing at
  all (+0.0001 / +0.0003), so the residual is not a constant offset either. The probes are fitting
  64 documents of a 128-dim, 8-head error and generalising to none of it.
- **On the Recolor arm the probes help** — 6–8 % of its own error on keys, 5–8 % on values — but
  Recolor starts 28 % (key) and 41 % (value) worse and ends 17 % / 30 % worse than the *uncorrected*
  Ridge. The correction is repairing part of Recolor's own deficiency, not exposing a transferable
  residual. If the shared residual were the exploitable object, correcting it on Recolor should
  carry over to Ridge; it goes the other way.
- **Quadratic buys nothing over linear** — −0.0007 on keys, −0.0020 on values, both from the
  `shared_linear − shared_quadratic` pair. Whatever remains is not a smooth nonlinear function of
  the body either.

## The geometry says the same thing: the rank-16 subspace is not special

Fraction of residual energy lying inside a rank-16 subspace out of 128 dimensions:

| Role | Estimator | own basis | other's basis | common basis |
|---|---|---:|---:|---:|
| key | ridge | 0.4552 | 0.4513 | 0.4547 |
| key | recolor | 0.4441 | 0.4404 | 0.4436 |
| value | ridge | 0.2413 | 0.2365 | 0.2409 |
| value | recolor | 0.2484 | 0.2410 | 0.2466 |

12.5 % of the dimensions carry 45 % of the key error and 24 % of the value error, and the three
columns are within 1 % of each other. A subspace that captures the same energy whichever basis you
choose is not a set of privileged directions; it is what an isotropic-ish error looks like when you
project it onto 16 of 128 axes. The high canonical correlation above is therefore a statement about
*both* estimators being dominated by the same broad error mass, not about a low-dimensional
structure that could be corrected.

That also explains the key/value asymmetry: keys are twice as concentrated (0.455 against 0.241) and
their relative error is 3.9× smaller (0.158 against 0.613), which matches the probe R² of 0.552 for K
and 0.277 for V measured in `dst_kvridge_fullbody_v1`.

## What actually moves the number: more data on the estimator, not on the residual

Refitting the base map on all 144 documents (80 + 64) instead of 80, and discarding the residual
probes entirely:

| Role | base (80 docs) | pooled refit (144 docs) | relative change |
|---|---:|---:|---:|
| key | 0.1575 | 0.1523 | −3.3 % |
| value | 0.6132 | 0.5913 | −3.6 % |

Adding the quadratic augmentation to that pooled refit gives 0.1521 / 0.5930 — i.e. the nonlinearity
is worth 0.0002, nothing. **The only lever in this experiment is calibration sample size on the
estimator.** Spending the same 64 documents on modelling the residual returns less than zero on the
Ridge arm and, on Recolor, less than the same documents would return if simply added to the base fit.

## Caveats

- **48 test documents.** The 288 head-layer cells are not independent; all intervals are computed
  over documents, and 48 is what they rest on.
- **The pooled refit is not a strict control.** It changes the estimator rather than adding a probe,
  so it is the honest reference for "what would the same data buy elsewhere", not a matched
  comparison of corrections.
- **Residual probes are rank-16 linear (or 128-feature quadratic) maps from the input.** A different
  correction family — per-token, attention-conditioned, or trained — is not covered. What is
  covered is the natural closed-form version of the idea.
- **Legacy Full-Sync alignment only**, `post-k_norm, pre-RoPE` keys, one calibration stream, seed
  1234, no gradient training anywhere.
- **`own_subspace` and `other_subspace` use different normalisers** (own divides by the error
  energy, `common_subspace` likewise), so the three columns are comparable to each other but not to
  the `base` ratios above.
