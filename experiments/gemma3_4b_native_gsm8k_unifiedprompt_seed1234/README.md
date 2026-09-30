# Gemma3-4B Native GSM8K unified-prompt evaluation

- Model: Gemma3-4B-IT, text tower only, FP32.
- Data: first 128 official GSM8K test rows.
- Prompt/evaluator: identical to `unified_fullbody_preanswer_hub_l3_to_q4_v1`.
- Decoding: greedy, at most 256 generated tokens.
- Outputs are flushed after every sample so partial accuracy remains observable.
