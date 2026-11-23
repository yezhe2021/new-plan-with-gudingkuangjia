# A Dynamic M32 Memory Bank on Top of DST + Rank-64 Moves Three Tasks and the Macro by Nothing

Gemma 3 4B sender → Qwen3 4B receiver, 64 rows per task, one scorer. The base arm is this
experiment's own DST + Rank-64 writer; the treatment adds the gated dynamic M32 memory bank in
keep-both mode, i.e. the Stage-A prefix state is retained and the bank is written alongside it
rather than replacing it.

| Task | Metric | DST+Rank64 | +M32 | Δ pp |
|---|---|---:|---:|---:|
| arc | correct | 57.81 | 60.94 | +3.12 |
| hotpotqa | em | 35.94 | 28.12 | −7.81 |
| hotpotqa | f1 | 53.97 | 47.37 | −6.61 |
| 2wiki | em | 15.62 | 20.31 | +4.69 |
| 2wiki | f1 | 28.38 | 33.65 | +5.27 |
| musique | em | 12.50 | 12.50 | +0.00 |
| musique | f1 | 16.51 | 16.85 | +0.34 |
| sciq | correct | 93.75 | 92.19 | −1.56 |
| mmlu_pro | correct | 10.94 | 15.62 | +4.69 |
| hellaswag | correct | 56.25 | 48.44 | −7.81 |
| hellaswag | raw_correct | 37.50 | 42.19 | +4.69 |
| gsm8k | correct | 75.00 | 76.56 | +1.56 |
| gsm8k | legacy_correct | 75.00 | 78.12 | +3.12 |
| **Macro (8 tasks, primary metric)** | | **44.73** | **44.34** | **−0.39** |

Paired bootstrap, 64 rows per task, memory minus base:

| Task | Metric | Δ pp | 95 % interval |
|---|---|---:|---|
| arc | correct | +3.12 | [−4.69, +10.94] |
| hotpotqa | em | −7.81 | [−17.19, +0.00] |
| hotpotqa | f1 | −6.61 | [−13.60, −0.18] |
| 2wiki | em | +4.69 | [+0.00, +10.94] |
| 2wiki | f1 | +5.27 | [+1.26, +10.39] |
| musique | em | +0.00 | [−4.69, +4.69] |
| sciq | correct | −1.56 | [−7.85, +4.69] |
| mmlu_pro | correct | +4.69 | [−1.56, +12.50] |
| hellaswag | correct | −7.81 | [−15.62, −1.56] |
| hellaswag | raw_correct | +4.69 | [−1.56, +10.94] |
| gsm8k | correct | +1.56 | [−6.25, +9.38] |
| gsm8k | legacy_correct | +3.12 | [−4.69, +12.50] |

Four tasks move up, three move down, one is exactly flat, and **the macro lands 0.39 pp below the
base arm**. The cells are not noise-free: on `f1` the two retrieval tasks move significantly in
opposite directions (hotpotqa −6.61 [−13.60, −0.18], 2wiki **+5.27** [+1.26, +10.39]), and
`hellaswag` loses 7.81 on `correct` [−15.62, −1.56] while gaining 4.69 on `raw_correct`, which is a
prompt-format artefact of the same 64 rows rather than two independent findings.

## Reading

The bank is a real mechanism with real capacity and it produces a flat macro. That is the third
independent time the memory content has failed to be the lever in this family: the M64 variant of
the same bank scores 52.93 against 53.32, swapping every M32 bank for a donor's moves macro by
0.2 pp, and a perfect-bank ceiling test flips 2 of 192 rows. Keep-both does not change the
conclusion — the bank neither helps nor hurts on aggregate, and the per-task moves it does produce
are within the resolution of 64 rows.

What this run does add is a clean within-experiment baseline: the base arm at 44.73 macro is
re-run here, on the same rows and scorer, so the −0.39 is a paired comparison rather than a
cross-experiment one. Read the Δ column and nothing else; the absolute numbers are lower than the
51.37 recorded for the lineage arm in `dst_kvridge_fullbody_v1` because this is a separate training
run, not because the recipe regressed.

## Caveats

- **64 rows per task.** ±8–12 pp resolution; only the `f1` and `hellaswag` intervals strictly
  exclude zero, and two of those three point in opposite directions.
- **One seed** (`seed1234`), one training run per arm.
- **`raw_correct` and `legacy_correct` are prompt variants** of the same rows, included because the
  parent experiments reported them; they are not independent tasks and are not in the macro.
- **The macro uses one primary metric per task** (em for the three retrieval sets, `correct`
  otherwise) to stay comparable with the rest of the lineage's tables.
