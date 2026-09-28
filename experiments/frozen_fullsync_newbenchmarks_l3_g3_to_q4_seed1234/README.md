# Frozen Full-Sync Writers on HellaSwag, GSM8K and LongBench v2

This experiment performs no training. It evaluates the existing freshly
trained Full-Sync Stage-A and Stage-B checkpoints for both Llama3.2-3B to
Qwen3-4B and Gemma3-4B to Qwen3-4B on 128 deterministic test examples from
each benchmark.

- HellaSwag uses its native four endings as A-D choices.
- The original `GSM8K-MC` files are retained only as a historical diagnostic.
  The formal GSM8K evaluation is `evaluate_gsm8k_generation.py`: unchanged
  free generation from `Question: ...\nAnswer:`, up to 384 tokens, followed by
  the existing numeric exact-match extraction. No choices are constructed.
- LongBench v2 uses its native A-D task. Context is symmetrically head/tail
  truncated when required so all tokenizers remain within the historical
  2,048-body-token ceiling; every truncation is recorded per sample.

Each result reports sender full-native, Qwen full-native, Qwen Full-Sync
native oracle, frozen Stage-A and frozen Stage-B. It also records native
agreement, choice KL, KV reconstruction metrics, jointly correct counts and
Writer-only/Oracle-only correct counts. Native KVs exist only in CPU RAM for
the current dataset and are never persisted.

Run with `bash run_pipeline.sh`; inspect with `tail -f logs/pipeline.log`.
Run the formal GSM8K test with `bash run_gsm8k_generation.sh`; inspect with
`tail -f logs/gsm8k_generation.log`.
