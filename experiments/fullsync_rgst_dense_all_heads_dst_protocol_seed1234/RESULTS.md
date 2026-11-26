# A dense all-heads writer with 7.6× the parameters of the grouped one loses 2.3 macro points and 6.2 multi-hop F1, and the arm that fits the KV better is the arm that answers worse

`fullsync_rgst_all_heads_dst_protocol` ended with a bound rather than a gain: head-mixing the RGST writer
at a fixed tensor budget moved the macro by −2.7 and +1.8 points at rank 16 and 32, both intervals
straddling zero. The obvious next objection is that the budget was the problem — that 16 hidden units per
output head is too thin to exploit cross-head structure, and the wiring was never really tested. This run
removes the budget constraint and hands the same protocol a dense writer.

**Architecture.** `dense_r32` keeps one dense layer per (role, layer): the input is the concatenation of
all four taps × 8 heads × 128 channels (4,096 wide), projected to 1,024 hidden units, reshaped to 8 heads ×
128, and each head then up-projected to 32 residual coordinates. That is 304,441,344 trainable parameters
against 40,200,192 for `grouped_r32` — **7.6×** — with the same optimiser, data, statistics, rank and
schedule. `grouped_r32` is not retrained here: it is the frozen `stage2/final_r32.pt` from
`fullsync_rgst_all_heads_dst_protocol`, loaded strictly and re-scored on the same items, so this is a
like-for-like, within-item contrast on the fixed 512-item suite (64 per task).

Both stages ran as before: stage 1 is 4 epochs at 3e-4 on mean per-layer K/V normalized NMSE in native
pre-k_norm/pre-RoPE space over full-body tokens, stage 2 is 4 epochs at 1e-4 with `lambda_kl = 0.5`, 1,536
updates per stage, effective batch 8. The dense run started 22:11 and finished stage 1 at 01:55, stage 2 at
07:58, evaluation at 08:31 (10 h 20 min wall clock, 14 sender layers offloaded to CPU). Greedy decoding
through the frozen suite; the longest generation here is 512 tokens and `reached_cap` is 0 in every cell
except `gsm8k` (1.6 % against 3.1 %), so unlike several earlier RGST runs neither arm degenerates into a
loop.

| Arm | 2wiki | arc | gsm8k | hellaswag | hotpotqa | mmlu_pro | musique | sciq | **macro** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense, rank 32, 304.4 M params | 34.70 | 60.94 | 78.12 | 53.12 | 53.64 | 29.69 | 22.01 | 98.44 | **53.83** |
| grouped, rank 32, 40.2 M params | 32.95 | 62.50 | 76.56 | 54.69 | 59.67 | 28.12 | 36.39 | 98.44 | **56.17** |

Paired differences with a stratified bootstrap (5,000 resamples inside each task, task means averaged), in
points:

| Contrast | choice (5 tasks) | multi-hop (3 tasks) | macro (8 tasks) |
|---|---:|---:|---:|
| dense − grouped, rank 32 | +0.00 [−3.44, +3.44] | −6.22 [−12.20, −0.19] | −2.33 [−5.52, +0.72] |

**The whole deficit is multi-hop, and almost all of it is one task.** On the five single-hop choice tasks
the two wirings are indistinguishable — the difference is exactly 0.00 points with a ±3.4 interval. The
multi-hop drop is real at the suite level (−6.22 with the interval excluding zero) but it is carried by
`musique`, where exact match falls from 17/64 to 10/64 and F1 from 36.39 to 22.01; `hotpotqa` contributes
−6.03 F1 with an interval that straddles zero, and `2wiki` moves the other way (+1.75). Read strictly, this
is one task out of eight: a −14.4-point `musique` cell plus noise elsewhere.

**Capacitance is not the constraint, and KV fidelity is the wrong proxy for it.** The dense arm reaches a
lower stage-1 training NMSE than the grouped arm — 0.2359 against 0.2538 at update 1,536 — and its
held-out KV validation at the end of stage 1 reads key 0.0507 / value 0.4544 full-space NMSE, degrading to
0.1444 / 0.5474 after stage 2 (the KL term trades KV fidelity for teacher agreement, as in every run of
this family). So the arm with 7.6× the parameters is the better KV regressor and the worse reader. The
stage-2 endpoints split in the same direction: at update 1,536 dense reads loss 0.543 (CE 0.268, KL 0.550)
against grouped's 0.480 (CE 0.173, KL 0.614) — dense imitates the teacher *better* (lower KL) while fitting
the gold continuation worse, and the gold continuation is what the reader is scored on. This is the third
independent instance of the same pattern in
this line — `dst_kvridge_fullbody_v1` had a translated arm at one-fifth of native's NLL and 14 points less
accuracy, restoring native attention mass on the translated cache *costs* 3.85 nats, and the per-head ridge
probe explains 0.277 of the value residual against 0.552 for the key. Lower KV error does not point toward
a better cache.

**What this rules out.** Combined with the previous run, the RGST family now spans 39–304 M trainable
parameters and sixteen hidden units to a full 1,024-wide dense layer, and every variant lands in a
53.8–58.8 macro band. The ten-point gap to the rank-32 oracle (67.36 on the 128-item suite, where the
residual is *given* rather than predicted) is therefore not addressable by width, by connectivity, or by
fitting the K/V tensors more closely — the residual has to be inferred from sender-side features, and that
inference is what is failing.

**Caveats.** One checkpoint per arm, no seed replication, so the −2.33 macro is a single draw; the interval
[−5.52, +0.72] does not exclude zero and I would not defend the sign of the macro on its own. The 512
items are 64 per task, which buys about ±5 points of resolution on the macro and ±1.6 points per cell, so
the `musique` gap (−14.4, 7 items of exact match) is the only cell strong enough to carry an argument. The
two arms were also trained at different times from the same frozen Ridge and statistics; the hashes in
`manifest.json` pin the control, the grouped checkpoint and the PCA basis, which makes the re-score exact
but does not make the two training runs paired.

Archived: code, manifest with control/grouped/final hashes, `per_sample/` for 512 items × 2 arms, run log,
stage completion records and KV validation. `*.pt` (stage1/stage2 `last.pt` at 3.5 GB plus five 1.2 GB rank
checkpoints each) and `run.pid`, `run.lock` are excluded per archive policy; 19 GB on disk reduces to
314 KB here.
