# Parameter-matched all-heads wiring inside the RGST sender lands within 2.7 macro points of the local-head control, so neither head mixing nor rank is where the missing eleven points live

The control run of this family (`fullsync_rgst_r16_r32_dst_protocol`) trained a residual correction on top
of the frozen Full-Sync Ridge, inside a rank-16 and a rank-32 subspace, and landed on top of DST Rank-64
within noise while staying ten points under native. One obvious explanation was architectural: the writer
was wired per head, so an output head could only ever see its own input head and could not exploit the
cross-head structure that attention heads are supposed to carry. This experiment removes that restriction
and trains the same writer a second way, at the same parameter count.

**Two wirings, identical tensors.** Both variants keep the control's tensor shapes exactly — `down`
`(576, 512, 128)` and `up` `(576, 128, rank)` over 2 roles × 36 layers × 8 heads — so the parameter counts
are 39,011,328 (rank 16) and 40,200,192 (rank 32) in either wiring. The *only* difference is connectivity:
the control's **local-head** writer gives each output head 128 hidden units driven by its own input head,
while **all-heads** reshapes the same weights into (output head × input head × 16 hidden units), so each
output head reads all eight input heads through 16 units apiece, and the hidden groups are mixed by the
original output layer. Training, statistics, data and schedules are the control's, unchanged: stage 1 is 4
epochs at 3e-4 on mean per-layer K/V normalized NMSE in native pre-k_norm/pre-RoPE space with full-body
tokens, stage 2 is 4 epochs at 1e-4 with `lambda_kl = 0.5`, 1,536 updates per rank per stage, best epoch 2
for both ranks. Stage 1 ran 11:26→13:50, stage 2 13:50→20:43, evaluation finished 21:05.

The local-head arms in this run are the control's frozen checkpoints re-scored, and they reproduce the
control's numbers on the shared items exactly (macro 58.76 and 54.69), so the comparison below is
like-for-like on the same 128 items (16 per task) rather than a cross-run comparison.

| Arm | 2wiki | arc | gsm8k | hellaswag | hotpotqa | mmlu_pro | musique | sciq | **macro** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all-heads, rank 16, final | 41.93 | 43.75 | 81.25 | 50.00 | 68.69 | 43.75 | 25.00 | 93.75 | **56.02** |
| all-heads, rank 32, final | 38.81 | 37.50 | 75.00 | 62.50 | 75.77 | 31.25 | 31.25 | 100.00 | **56.51** |
| local-head, rank 16 (control) | 53.66 | 43.75 | 81.25 | 56.25 | 67.44 | 43.75 | 30.21 | 93.75 | **58.76** |
| local-head, rank 32 (control) | 38.81 | 43.75 | 75.00 | 62.50 | 73.69 | 37.50 | 6.25 | 100.00 | **54.69** |
| all-heads, rank 16, best epoch | 45.06 | 50.00 | 75.00 | 56.25 | 68.07 | 43.75 | 20.31 | 100.00 | **57.30** |
| all-heads, rank 32, best epoch | 45.06 | 37.50 | 75.00 | 62.50 | 68.69 | 43.75 | 21.88 | 100.00 | **56.80** |

Reference arms scored by the control on the same 128 items, for scale: native **70.31**, rank-32 oracle
**67.36**, rank-16 oracle **57.52**, DST Rank-64 **55.00**, frozen Ridge alone **34.80**. The oracle arms are
`ridge + P_r(native − ridge)` — the residual projected onto the same frozen rank-`r` basis — so they mark
what a correction of that width could reach if the residual were known rather than predicted.

Paired differences with a stratified bootstrap (4,000 resamples, resampling inside each task and averaging
task means), in points:

| Contrast | choice (5 tasks) | multi-hop (3 tasks) | macro (8 tasks) |
|---|---:|---:|---:|
| all-heads − local-head, rank 16 | −1.25 [−6.25, +3.75] | −5.23 [−14.44, +3.32] | −2.74 [−7.18, +1.88] |
| all-heads − local-head, rank 32 | −2.50 [−7.50, +2.50] | +9.03 [−1.39, +19.79] | +1.82 [−3.12, +6.64] |
| rank 32 − rank 16, all-heads | — | — | +0.49 [−5.16, +6.09] |
| rank 32 − rank 16, local-head | — | — | −4.07 [−9.44, +1.36] |

**Head mixing does nothing measurable, and the sign is not even stable.** At rank 16 all-heads is 2.74
points below the local-head control; at rank 32 it is 1.82 above; both intervals straddle zero and the two
multi-hop intervals point in opposite directions with ±10-point widths. The honest reading is a bound: with
128 items this design can resolve about ±5 points on the macro, so any benefit or harm from cross-head
mixing is smaller than that. Rank follows the same pattern (+0.49 [−5.16, +6.09] inside all-heads), which
matches the control's own `rgst32 − rgst16` of −0.004 on the 256-item choice suite.

**The four RGST variants occupy a 4.1-point band, 11.5 under native and 10.9 under the rank-32 oracle.**
That is the more useful number here. Capacity was never the binding constraint in this family — the same
tensor budget, spent either locally or across heads, buys the same 55–59 macro. Because `oracle32` shows a
rank-32 correction reaching 67.36 when the residual is handed to it, the ten points between RGST and that
oracle are a *prediction* problem (the residual must be inferred from sender features) and not a bandwidth
or wiring problem. Whatever the translated cache loses on its way to the receiver is not recoverable by
giving the writer a wider receptive field on the sender side, which is consistent with two earlier results
in this line: restoring native attention mass on the translated cache *costs* 3.85 nats, and the per-head
ridge probe explains only 0.277 of the value residual against 0.552 for the key.

**Caveats that limit what can be claimed.** The 128-item suite is 16 items per task, so single-task cells
move by one item ≈ 6 points: the +9.03 at rank 32 is almost entirely `musique`, where the local-head rank-32
arm answered 1 of 16 (6.25 F1) against 5 of 16 for all-heads — one task's instability, not evidence for
mixing. The suite is also not a random subset of the control's: on the shared items `hotpotqa` reads 67.44
for the local-head rank-16 arm against 60.52 over the control's 64, while `musique` reads 6.25 for rank 32
against 28.11 over 64, so absolute levels here must not be compared against the control experiment's
published numbers. Only the within-run, within-item comparisons in the table above are load-bearing.

Archived: code, manifest with checkpoint and control hashes, `per_sample/` for 128 items × 6 arms, the run
log and per-stage completion records. `*.pt` (stage1/stage2 `last.pt` at 907 MB plus four rank checkpoints)
and `run.pid` are excluded per archive policy; 4.8 GB on disk reduces to 2.1 MB here.
