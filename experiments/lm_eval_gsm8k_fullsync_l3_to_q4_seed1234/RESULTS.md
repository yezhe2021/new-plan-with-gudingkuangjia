# Results

The correctness pilot completed on 2026-09-22 with `lm_eval==0.4.12`, eager
attention, float16 weights, greedy decoding, a 256-token generation budget,
zero few-shot examples, and the first 10 official GSM8K test rows. The local
task YAML reproduces the stock GSM8K v3 prompt, generation settings, filters,
and metrics; only the dataset loader points to the server's official JSONL.

## Main pilot

| Condition | Strict-match EM | Flexible-extract EM |
|---|---:|---:|
| Llama3.2-3B native | 0/10 (0%) | 4/10 (40%) |
| Qwen3-4B native (HF backend) | 0/10 (0%) | 6/10 (60%) |
| Qwen3-4B native (custom backend audit) | 0/10 (0%) | 6/10 (60%) |
| Llama to Qwen Full-Sync Generation Stage-B | 2/10 (20%) | 1/10 (10%) |

Strict and flexible extraction are independent filters, so one is not required
to be an upper bound on the other for an individual output.

## Backend correctness

After matching eager attention and honoring both Qwen EOS token IDs from its
generation config, the official HF Qwen backend and custom native-Qwen mode
produced exact-identical continuation text for all 10 requests. lm-eval logs
each request once per filter, giving an audit result of 20/20 exact records.

This rules out prompt construction, tokenization, position IDs, greedy decode,
EOS handling, and task stop strings as explanations for the transfer gap.

## Full-Sync diagnostics

| Diagnostic | Value |
|---|---:|
| Requests | 10 |
| Mean Sender prefix tokens | 62.8 |
| Sender token range | 29--111 |
| Mean Receiver prefix tokens | 64.4 |
| Receiver token range | 28--113 |
| Mean translated tokens | 63.4 |
| Mean generated tokens | 256 |
| Reached max generation budget | 10/10 |
| Stopped by EOS/task stop | 0/10 |

The transfer model recovered non-zero correctness under the standard harness,
but all outputs exhausted the generation budget. Its weak termination behavior
is therefore a primary diagnostic result of this pilot.

The repository contains raw lm-eval result JSON, per-sample logs, native-backend
comparisons, alignment diagnostics, and pipeline logs. Models and checkpoints
are referenced by path only and are not included.

## Standard 5-shot / 32-sample evaluation

This follow-up uses the same GSM8K v3 task definition and eager float16
backends, with `num_fewshot=5`, the first 32 official test rows, greedy
decoding, and a 256-token generation budget.

| Condition | Strict-match EM | Flexible-extract EM |
|---|---:|---:|
| Llama3.2-3B native | 18/32 (56.25%) | 19/32 (59.38%) |
| Qwen3-4B native (HF backend) | 26/32 (81.25%) | 27/32 (84.38%) |
| Llama to Qwen Full-Sync Generation Stage-B | 2/32 (6.25%) | 0/32 (0%) |

The strict/flexible inversion for Full-Sync is preserved exactly as reported
by lm-eval's two independent extraction filters. It indicates a generation
format/extraction failure mode and should not be read as a monotonic pair of
metrics. The flattened result directory includes all three aggregate JSON
files, all 96 per-sample generation records, the Full-Sync backend audit, and
the complete execution log.

## Stage-A-only generation

This ablation uses the same first 32 GSM8K test rows and generation protocol,
but bypasses the Generation Stage-B residual adapter. The transferred cache is
therefore produced only by the frozen Stage-A Full-Sync translator. Audit rows
record `generation_stage_b_checkpoint: null`.

| Condition | Strict-match EM | Flexible-extract EM |
|---|---:|---:|
| Stage-A only, 0-shot | 0/32 (0%) | 1/32 (3.12%) |
| Stage-A only, 5-shot | 0/32 (0%) | 0/32 (0%) |

The repository includes the backend change adding the explicit
`fullsync_stagea` mode, the combined runner, both aggregate result files, all
64 per-sample generations, both backend audits, and the complete run log.
