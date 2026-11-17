# Folded Recolor Affine and Fused DST Readout

Follow-up to the batch-2 latency sweep in the parent directory. Same frozen checkpoints
(`14a4c5f7…` recolor_c, `4547b751…` gated dynamic), same nested-prefix QuALITY prompts,
batch 2, chunk 2048, 3 trials × 2 batch slots = 6 timings per arm, arm order reshuffled every
trial. One V100-SXM2-32GB.

Two code-level changes are isolated, each against its own untouched original:

- **`recolor_folded`** — the recolor affine is pre-folded into a single weight/bias pair, so K/V
  no longer round-trips through an intermediate fp16 tensor between the two matmuls.
- **`dst_fused`** — the residual readout `x + up(gelu(down(x)))` becomes an in-place
  `addmm_` accumulate, so the residual stream is written once instead of twice.

The Sender weights are unloaded during measurement in this run (`sender_weights_resident_during_measurement:
false`), so peaks here are 3–4 GiB below the parent run and the absolute times are not
comparable across the two directories. All comparisons below are within-run and paired.

## Handoff latency

Medians over 6 timings; translation and post-upload are phase medians over the 2 profiled
batches, milliseconds.

| Body tokens | Arm | Handoff p50 | Handoff p95 | Translation p50 | Post-upload p50 | Peak (GiB) |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | recolor_original | 1705 | 1846 | 782 | 1010 | 12.63 |
| 4096 | recolor_folded | 1556 | 1821 | 748 | 969 | 12.61 |
| 4096 | dst_original | 1923 | 2005 | 532 | 943 | 12.58 |
| 4096 | dst_fused | 1728 | 1958 | 502 | 910 | 12.55 |
| 16384 | recolor_original | 6941 | 7219 | 3171 | 3991 | 16.00 |
| 16384 | recolor_folded | 6189 | 7164 | 2990 | 3816 | 15.99 |
| 16384 | dst_original | 7122 | 7446 | 2060 | 3329 | 15.95 |
| 16384 | dst_fused | 6932 | 7880 | 1992 | 3245 | 15.92 |
| 32768 | recolor_original | 13081 | 13234 | 6309 | 7887 | 20.50 |
| 32768 | recolor_folded | 12709 | 13090 | 5936 | 7521 | 20.49 |
| 32768 | dst_original | 13304 | 13703 | 4131 | 6497 | 20.45 |
| 32768 | dst_fused | 13175 | 13598 | 3997 | 6356 | 20.42 |

## Paired deltas

Mean of per-trial differences with a 4000-sample bootstrap interval on the paired deltas.

| Body tokens | Change | Δ handoff (ms) | 95% interval | Relative |
|---:|---|---:|---|---:|
| 4096 | recolor_folded − recolor_original | −81 | [−220, +66] | −4.7% |
| 16384 | recolor_folded − recolor_original | **−559** | **[−736, −321]** | **−8.1%** |
| 32768 | recolor_folded − recolor_original | −269 | [−512, +5] | −2.1% |
| 4096 | dst_fused − dst_original | −88 | [−291, +118] | −4.8% |
| 16384 | dst_fused − dst_original | −40 | [−436, +329] | −0.6% |
| 32768 | dst_fused − dst_original | −130 | [−463, +164] | −1.0% |

Only one cell clears zero: recolor folding at 16384 saves 0.56 s (−8.1%). At 32k the same
change is −2.1% with an interval that just touches zero, and at 4k it is inside the noise.
DST fusion is a consistent direction (−0.6% to −4.8%) that never separates from zero at n=6.
Both changes are cheap in engineering and neither moves peak memory (≤0.08 GiB), so the honest
reading is: worth keeping for the rounding behaviour, not worth claiming as a latency win.

## What the two changes do to the numbers

`cache_parity_4k.json` compares the produced cache against the original path (relative L2):

| Method | Key | Value |
|---|---:|---:|
| recolor | 5.21e-4 | 3.73e-4 |
| dst | 0.0 | 0.0 |

DST fusion is bit-exact, which follows from the algebra: `addmm_` with the same fp16 operands
performs the identical multiply-accumulate, only the destination differs. Recolor folding is
*not* exact — it deliberately removes an intermediate fp16 materialisation, so the cache it
produces is a slightly different (more precise) tensor, 5e-4 relative.

Downstream, that drift never changes a token. Over both batches at every length:
first-token relative logit L2 is 2.5e-3 to 6.1e-3 for recolor and exactly 0.0 for dst, while
both arms match the original on 8 of 8 greedy tokens and both first tokens.

So the recolor saving at 16k is buying back time that the original path spends shrinking a
tensor to fp16 and reading it back, and the accuracy cost is below the resolution of greedy
decoding on this prompt set. It is not free in the strict sense, and it should not be described
as parity.

## Caveats

- n=6 per arm. The bootstrap interval is a within-run resampling, not a hardware-noise model;
  a 5% effect at one length is at the edge of what this design can resolve.
- Translation is 3.2 s of 6.9 s at 16k for recolor and 2.1 s for dst, so folding can only
  touch part of the budget. Upload and assembly still dominate, as in the parent experiment.
- These arms were measured with Sender weights unloaded. Do not compare their peaks or absolute
  times to the parent directory's table, where both models stay resident.
