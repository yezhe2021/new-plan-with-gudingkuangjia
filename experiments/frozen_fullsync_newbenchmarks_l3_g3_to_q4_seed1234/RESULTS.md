# Results

All figures use 128 deterministic test examples. Writers and Stage-B adapters
are frozen checkpoints trained only on OpenBookQA, ARC-Challenge and MMLU-Pro.

## Llama3.2-3B to Qwen3-4B

| Dataset | Sender native | Qwen native | Native oracle | Stage-A | Stage-B |
|---|---:|---:|---:|---:|---:|
| HellaSwag | 64.06% | 79.69% | 79.69% | 39.84% | **63.28%** |
| GSM8K-MC | 25.00% | 45.31% | 45.31% | 30.47% | **28.91%** |
| LongBench v2 | 23.44% | 29.69% | 29.69% | 27.34% | **26.56%** |

## Gemma3-4B to Qwen3-4B

| Dataset | Sender native | Qwen native | Native oracle | Stage-A | Stage-B |
|---|---:|---:|---:|---:|---:|
| HellaSwag | 55.47% | 79.69% | 79.69% | 24.22% | **65.63%** |
| GSM8K-MC | 28.13% | 45.31% | 45.31% | 27.34% | **27.34%** |
| LongBench v2 | 24.22% | 28.13% | 27.34% | 18.75% | **26.56%** |

The GSM8K-MC row is retained only as a historical diagnostic and is not the
formal GSM8K result. Formal unchanged free-generation results are written to
`runs/gsm8k_generation/` by `run_gsm8k_generation.sh`. All 128 LongBench v2 examples required symmetric context
truncation to the experiment's historical 2,048-body-token ceiling. Exact
per-sample predictions, choice logits, common/Writer-only/Oracle-only correct
counts, native agreement, choice KL and KV metrics are in `runs/`.

## Standard GSM8K quick generation

To keep turnaround short, the formal generation audit uses the first 32
official test rows and only the Llama3.2-3B Sender Native and frozen Stage-B
conditions. It uses `Question:\n...\n\nAnswer:`, greedy generation up to 384
tokens, and the existing numeric extraction logic; no choices are constructed.

| Condition | Exact match | Contains gold | Mean generated tokens |
|---|---:|---:|---:|
| Llama Sender Native | 11/32 (34.38%) | 56.25% | 115.38 |
| Llama→Qwen Stage-B | 0/32 (0.00%) | 0.00% | 264.19 |

The native-text/native-cache equivalence smoke test reached exact generation
agreement 1.0 before this run, so the result is not explained by cache
construction, RoPE positions, or the generation loop.
