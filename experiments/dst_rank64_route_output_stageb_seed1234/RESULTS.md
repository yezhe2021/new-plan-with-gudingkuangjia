# DST Stage B Retrain with Route and Output Distillation

Llama3.2-3B → Qwen3-4B, frozen rank-64 body (`aaae0362…`, the stage-2 checkpoint of
`dst_stage1_rank64_then_static_m32_a4_seed1234`), original Stage-B training data, 1,536 updates
at effective batch 8, learning rate 1e-4, `ce_weight 1.0`, `vocab_kl_weight 0.5`. Trainable
parameters 9,437,184 — the body stays frozen, only the head is retrained.

What is new is two auxiliary heads supervised by the teacher at sparse probe positions:

- **route** — fixed-query attention distillation, at most 4 probe queries per layer, matching the
  teacher's routing rather than only its output.
- **output** — NMSE on the output path, normalised against V (`output_to_v`).

Prefix quotas walk the depth axis from teacher-fed to student-fed:
(4 gold / 4 native / 0 student) → (3 / 3 / 2) → (2 / 2 / 4) → (2 / 2 / 4).
`aux_gradient_target` is 0.1 with `max_smoke_gradient_share` 0.2; the measured pooled shares are
route 0.0756 and output 0.0283, i.e. about a tenth of the gradient, as intended.

## Short tasks, 512 rows, same test set as the rest of the family

`test_sha256 de54be31…`, 64 rows per task. `B_rank64` is the original `dst_rank64` and `C` is
`recolor_c`; both reproduce their published numbers exactly, so the new column is directly
comparable.

| Task | Metric | native | B_rank64 | C | route_output |
|---|---|---:|---:|---:|---:|
| arc | correct | 87.50 | 67.19 | 67.19 | 62.50 |
| hotpotqa | em | 56.25 | 45.31 | 29.69 | 39.06 |
| 2wiki | em | 32.81 | 23.44 | 9.38 | 21.88 |
| musique | em | 25.00 | 21.88 | 1.56 | 15.62 |
| sciq | correct | 100.00 | 98.44 | 98.44 | 100.00 |
| mmlu_pro | correct | 53.12 | 21.88 | 25.00 | 29.69 |
| hellaswag | correct | 45.31 | 60.94 | 57.81 | 56.25 |
| gsm8k | correct | 89.06 | 73.44 | 70.31 | 75.00 |
| **Macro** | | **61.13** | **51.56** | **44.92** | **50.00** |

Paired against `B_rank64`, 64 rows each:

| Task | Metric | Δ (pp) | 95% interval |
|---|---|---:|---|
| mmlu_pro | correct | **+7.81** | **[+1.56, +15.62]** |
| sciq | correct | +1.56 | [+0.00, +4.69] |
| gsm8k | correct | +1.56 | [−4.69, +7.81] |
| 2wiki | em | −1.56 | [−10.94, +7.81] |
| arc | correct | −4.69 | [−10.94, +1.56] |
| hellaswag | correct | −4.69 | [−14.06, +4.69] |
| hotpotqa | em | −6.25 | [−15.62, +3.12] |
| musique | em | −6.25 | [−14.06, +0.00] |
| musique | f1 | **−7.83** | **[−14.96, −1.58]** |

## Long documents, same L-Eval cohort as `long_qa_quality_leval_full64`

192 rows, 64 per source, no truncation. All five previous arms reproduce to the point, so only
the last column is new.

| Source | Metric | native | dst_rank64 | gated_dynamic | heterofold_r64 | recolor_c | route_output |
|---|---|---:|---:|---:|---:|---:|---:|
| quality | correct | 73.44 | 45.31 | 46.88 | 59.38 | 65.62 | 51.56 |
| coursera | correct | 67.19 | 32.81 | 35.94 | 42.19 | 48.44 | 31.25 |
| sfiction | correct | 78.91 | 57.03 | 60.16 | 59.38 | 59.38 | 48.44 |
| sfiction | loyalty_correct | 85.94 | 56.25 | 56.25 | 56.25 | 54.69 | 46.88 |
| sfiction | fact_correct | 71.88 | 57.81 | 64.06 | 62.50 | 64.06 | 50.00 |
| **Mean correct** | | **73.18** | **45.05** | **47.66** | **53.65** | **57.81** | **43.75** |

Paired against `dst_rank64`: quality +6.25 [−3.12, +15.62], coursera −1.56 [−6.25, +3.12],
sfiction **−8.59 [−15.62, −2.34]**, loyalty −9.38 [−20.31, +0.00], fact −7.81 [−17.19, +1.56].

## Reading

The auxiliary heads do exactly what they were asked to and the model gets slightly worse.

Everything that can be answered from a single span or by choosing among options improves or holds:
mmlu_pro +7.81, sciq +1.56, gsm8k +1.56, quality correct +6.25. Everything that requires carrying
evidence out of a long passage falls: hotpotqa em −6.25, musique f1 −7.83, sfiction correct −8.59,
and sfiction loyalty drops from 56.25 to 46.88 — now the lowest of six arms, against 85.94 for
native. Routing distillation pushes the receiver to attend where the teacher attends, and on this
evidence the student's own routing was more useful for faithfulness than the teacher's is.

Net: 51.56 → 50.00 on the short cohort and 45.05 → 43.75 on the long one. Attaching two
well-behaved auxiliary losses with a 10% gradient share and 1,536 updates does not buy a result.

## The validation curve says the same thing

The per-epoch validation set is 192 rows over arc, hotpotqa and gsm8k. `route_kl` and
`output_nmse` fall monotonically to the last epoch, which is what a converged auxiliary loss looks
like. The objectives that matter do not.

| Epoch | Step | task_loss | full_kl | route_kl | output_nmse |
|---:|---:|---:|---:|---:|---:|
| 1 | 384 | 0.7770 | **0.6668** | 0.1151 | 0.0886 |
| 2 | 768 | 0.6793 | 0.7774 | 0.1062 | 0.0866 |
| 3 | 1152 | **0.6479** | 0.8482 | 0.1020 | 0.0855 |
| 4 | 1536 | 0.6769 | 0.7130 | **0.0994** | **0.0844** |

`full_kl` against the teacher is best at epoch 1 and never recovers to it. `task_loss` bottoms at
epoch 3 and turns up at epoch 4 — the shipped checkpoint is not the validation best on either
metric. The run was 1,536 updates because that is the family convention, not because the curve
asked for it.

## Attempt history

`attempts/01_initial_aux_cap/` holds the first launch: same manifest, stopped at step 4 of 1,536
while the auxiliary-gradient smoke phase was being calibrated (`checks/weights.json` is the
calibration that shipped: route 18.24 / output 7.66 raw, pooled shares 0.0756 / 0.0283). It is
kept because the smoke numbers are the only record of how large the auxiliary gradients are
before they are rescaled.

## Caveats

- Comparisons above are paired on 64 rows per task. Only mmlu_pro (+7.81) and musique f1 (−7.83)
  clear zero on the short cohort; the macro move of −1.56 pp does not.
- The short and long cohorts are the same ones used by the rest of the family, and the short one
  is the development cohort, so this arm had no held-out advantage to begin with.
- Weights are excluded from the repository (`*.pt`). The manifest pins
  `checkpoint_sha256 201a85e5…` for the evaluated checkpoint, which is `runs/epoch4_step1536.pt`
  on the training host.
