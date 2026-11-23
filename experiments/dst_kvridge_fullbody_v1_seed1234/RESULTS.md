# The Full-Body Ridge Loses to the 64-Token Bottleneck on Its Own and Wins Once Rank-64 Cleans It Up

DST carries the sender content in 64 latent slots. Every deficit in this family is therefore a
candidate bottleneck deficit. This experiment removes the bottleneck rather than widening it: the
receiver reads **raw K/V at every token position from all 28 sender layers**, mapped into its own
36-layer raw K/V space by a closed-form Ridge, with the pinned Rank-64 stage-B code then applied
unchanged on top.

The result is a clean 2×2, and the two halves point in opposite directions.

| Writer front-end | no stage-B | + pinned Rank-64 stage-B |
|---|---:|---:|
| DST latent slots (archived stage 1) | 41.21 | 51.37 |
| **Raw-body Ridge** | **38.09** | **53.32** |

The Ridge is the *worse* writer before correction (−3.12) and the *better* one after (+1.95). Both
front-ends gain from stage-B, but the Ridge gains more: +15.23 against +10.16.

## What was actually built

**Stage A — Ridge calibration, no gradient steps.** 500 FineWeb-Edu documents at 1024 tokens,
stride 4 over uniquely matching character spans, giving 123,827 aligned (sender position, receiver
position) pairs. The first 32 documents only select taps: for each receiver layer, the top 4 sender
layers by linear probe R². Those 36 tap sets are frozen, then one covariance solve per layer and
role at λ = 0.01 produces FP32 coefficients over the concatenation of 4 × (8 heads × 128) features
→ 1024. Keys and values are handled separately, in `pre-k_norm, pre-RoPE` space. FP16 capture, FP32
solve, 1.2 GB of coefficients. Fit 20:18:09 → 20:25:40.

**The probe is the interesting part.** Best-tap R² per receiver layer:

| | mean | min | max |
|---|---:|---:|---:|
| K | 0.552 | 0.477 | 0.793 |
| V | 0.277 | 0.176 | 0.448 |

Keys are largely a linear function of the sender body; values are not. That asymmetry survives into
the delivered cache (K NMSE 0.22 against V 0.48, below), and it is why the whole approach is bounded
by the value side no matter how the map is fitted.

**Stage B — the pinned code, unchanged.** `dst_stage1_rank64_then_static_m32_a4_seed1234` train and
runtime modules are imported as-is; only `Engine.source` is replaced by the Ridge. Rank 64,
9,437,184 parameters, residual output zero-initialised, so the start state is *exactly* the Ridge
(`checks/gpu_completed.json` asserts Ridge/native raw-cache KL = 3.15e-07 and residual-identity
parity). 1,536 updates over 4 epochs on arc + hotpotqa + gsm8k × 1024, objective `CE + 0.5 · Full-KL`
against native. Training 20:25:57 → 01:40:59, 5 h 15 m. Final step loss 0.8851 (task 0.5530,
full_kl 0.6642), `amp_retries` 0 throughout, peak reserved 22.05 GiB.

## Test scores

8 tasks × 64, `dst-fullbody-chat-v1`, same scorer for every arm. `historical_mlp_stage_a` is the
archived writer carried along as a control (see below), not a baseline for this comparison.

| Task | Metric | native | dst_stage1 | ridge_stage_a | dst_rank64 | **ridge_rank64** | historical MLP |
|---|---|---:|---:|---:|---:|---:|---:|
| arc | correct | 87.50 | 62.50 | 65.62 | 67.19 | **70.31** | 48.44 |
| hotpotqa | em | 56.25 | 26.56 | 14.06 | 45.31 | **45.31** | 0.00 |
| 2wiki | em | 32.81 | 9.38 | 12.50 | 23.44 | **29.69** | 0.00 |
| musique | em | 25.00 | 3.12 | 1.56 | 21.88 | **26.56** | 0.00 |
| sciq | correct | 100.00 | 98.44 | 95.31 | 98.44 | **98.44** | 42.19 |
| mmlu_pro | correct | 53.12 | 20.31 | 28.12 | 20.31 | **28.12** | 15.62 |
| hellaswag | correct | 45.31 | 46.88 | 50.00 | 60.94 | **57.81** | 31.25 |
| gsm8k | correct | 89.06 | 62.50 | 37.50 | 73.44 | **70.31** | 0.00 |
| **Macro** | | **61.13** | **41.21** | **38.09** | **51.37** | **53.32** | **17.19** |

Paired, 64 rows per task, ridge_rank64 against each arm (`mean` is the Ridge arm's gain):

| Task | vs dst_rank64 | 95% interval | vs dst_stage1 | 95% interval |
|---|---:|---|---:|---|
| arc | +3.12 | [−4.69, +10.94] | +7.81 | [+0.00, +17.19] |
| hotpotqa | +0.00 | [−10.94, +10.94] | +18.75 | [+6.25, +31.25] |
| 2wiki | +6.25 | [−4.69, +17.19] | +20.31 | [+7.81, +32.81] |
| musique | +4.69 | [−4.69, +14.06] | +23.44 | [+14.06, +34.38] |
| sciq | +0.00 | [+0.00, +0.00] | +0.00 | [+0.00, +0.00] |
| mmlu_pro | +7.81 | [−6.25, +21.88] | +7.81 | [−6.25, +20.31] |
| hellaswag | −3.12 | [−10.94, +4.69] | +10.94 | [−1.56, +21.88] |
| gsm8k | −3.12 | [−12.50, +4.69] | +7.81 | [−3.12, +20.31] |

**The +1.95 macro is diffuse, not a headline win.** Five tasks move up or hold, two move down by
exactly the same 3.12, and **no per-task interval against `dst_rank64` excludes zero**. On this
sample size the honest statement is "the Ridge front-end is not worse, and the aggregate is
1.95 pp ahead" — not "the Ridge fixes 2wiki".

The comparison against `dst_stage1` (the bottleneck with no stage-B) is the one with real signal, and
it is large: +7.8 to +23.4 on five tasks. That is mostly stage-B's contribution, not the Ridge's,
which is exactly why the 2×2 above is the right way to read this.

## The Ridge is a worse writer by the metric we would have used to judge it

KV reconstruction against native, 64 validation rows:

| Writer | K NMSE | V NMSE | KV NMSE |
|---|---:|---:|---:|
| dst_stage1 (latent slots) | 0.116 | 0.317 | 0.216 |
| ridge_stage_a (raw-body Ridge) | 0.221 | 0.482 | 0.352 |

The Ridge is 1.9× worse on keys and 1.5× worse on values, and yet its corrected output is the
better one. So NMSE-against-native is **not** the quantity that predicts downstream accuracy here.
The likely reading is that the Ridge leaves a smaller *structured, linear* residual for Rank-64 to
absorb, while the latent bottleneck leaves a residual that is already nonlinear — but this is a
post-hoc story and the experiment does not isolate it. What the experiment does establish is the
negative: a lower KV NMSE would have led you to the wrong arm.

## The 20-task field cohort still fails, and the Ridge makes it slightly worse

`herb` = the frozen 20 official-field questions, existing Evidence→Question prompt, standard
Full-Sync, no field prefill.

| Arm | exact | set_f1 | recall | capped |
|---|---:|---:|---:|---:|
| native | 0.800 | 0.9373 | 0.938 | 0.00 |
| dst_stage1 | 0.000 | 0.0591 | 0.185 | 0.50 |
| ridge_stage_a | 0.000 | 0.1137 | 0.341 | 0.60 |
| dst_rank64 | 0.000 | 0.0957 | 0.254 | 0.60 |
| **ridge_rank64** | **0.000** | **0.0552** | **0.164** | **0.40** |
| historical MLP | 0.000 | 0.0000 | 0.000 | 0.80 |

Every DST-family arm scores 0/20 exact. The new arm is the **worst of the live arms** on set_f1
(0.055 against 0.096) and recall (0.164 against 0.254). It does stop looping more often — capped
drops 0.60 → 0.40 — but it terminates with fewer correct IDs, so the truncation relief buys nothing.
Paired against `dst_stage1`: set_f1 −0.004 [−0.037, +0.029], recall −0.021 [−0.164, +0.109]. On 20
rows this cannot separate the arms; it can only say the Ridge does not repair the field task, which
is the same wall the whole family hits.

## The archived writer control collapses, which explains the earlier `fullsync_ab` zero

`historical_mlp_stage_a` is the archived Sender-prefix / post-k_norm cache writer, loaded from the
exact stage-A checkpoint that the herb experiments pin as `PINS["fullsync_ab"]["stage_a"]` and scored
here by the same scorer as everything else. It lands at **17.19 macro, 0.00 on four of eight tasks,
80 % capped on the field cohort**.

That is a direct control on the previous investigation: when the `fullsync_ab` arm scored 0.000 with
20/20 two-token loops, the suspicion was a broken scorer or a broken prompt. This run puts the same
writer through the standard scorer across nine tasks and it fails the same way. **The failure is in
the post-k_norm writer path, not in the field-task scorer.**

## Caveats

- **One seed.** `seed1234`, and the stage-B comparison inherits the parent's seed.
- **64 rows per task.** The resolution is roughly ±6–10 pp; macro intervals overlap and the +1.95
  is not individually significant on any task.
- **1.2 GB of FP32 Ridge.** This is a diagnostic about where the loss lives, not a deployment
  proposal; there is no compression in the front-end at all.
- **The stage-B gain is confounded with initialization.** The Ridge arm starts at a different point
  and gains more (+15.23 vs +10.16 in 1,536 updates). Whether it would keep gaining, or whether the
  latent arm is simply slower, is not tested.
- **`status.json` is stale.** It still reads `"status": "running", "phase": "validation"` with
  `time 2026-11-23 01:40:58`, while `evaluation/summary.json` reports `"status": "completed"` at
  03:42 and `runs/stage2/completed.json` records `finished_at 01:40:59`. The pipeline finished; the
  final status write did not happen. Any monitor keyed on `status.json` would report this run as
  still going.
- **`herb` is 20 rows.** ±16 pp on recall. Treat the ordering of the four DST arms there as noise.
