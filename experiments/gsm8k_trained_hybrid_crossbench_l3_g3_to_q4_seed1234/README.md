# GSM8K-trained Hybrid Writers: four-dataset frozen evaluation

Seed 1234. Freeze the existing GSM8K Llama3.2-3B→Qwen3-4B and
Gemma3-4B→Qwen3-4B Stage-A+Hybrid checkpoints and evaluate 128 fixed test
questions each from OpenBookQA, ARC-Challenge, MMLU-Pro and HellaSwag.
Each receiver uses native Qwen token0 followed by translated full
Question+Options KV and native `Answer:`; this is **not** the older
receiver-native Question plus translated Options protocol.

No training occurs in this archived directory. Raw per-sample outputs, manifest
audits, and summaries are under `runs/hybrid/{llama,gemma}`. Model checkpoints
and KV caches are intentionally excluded.
