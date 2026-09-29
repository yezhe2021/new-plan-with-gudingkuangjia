# Cross-Sender Functional Equivalence Audit

This is a frozen-checkpoint audit on the same 128 GSM8K test rows. It compares
Qwen-native, Llama→Qwen Hybrid4ep, and Gemma→Qwen Hybrid4ep KV states.

It reports:

- pairwise K/V cosine and symmetric NMSE, overall and per Qwen layer;
- full-vocabulary KL and JS under native-, Llama-, and Gemma-rollout prefixes;
- Llama↔Gemma linear interpolation behavior along the native trajectory;
- per-sample representation/functional distances and their correlation;
- the already-observed joint answer correctness for the exact same IDs.

Sender states are captured sequentially. Temporary KV tensors are removed after
successful audit; checkpoints are never modified and no training is performed.
