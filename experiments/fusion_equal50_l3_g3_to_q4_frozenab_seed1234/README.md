# Frozen Llama + Gemma equal-weight KV fusion into Qwen

Authorized on 2026-10-24: run fusion method2 first. No new training, no learned projection, no Reader.

For every existing translated Qwen slot, target layer and head, fuse the independently translated raw K and V after each sender's frozen Stage A + residual Stage B:

`K = (K_llama.float() + K_gemma.float()) / 2`

`V = (V_llama.float() + V_gemma.float()) / 2`

The FP32 arithmetic is cast once to the existing FP16 Qwen cache dtype. Both branches produce `[36,T,8,128]` on identical Qwen token positions. Do not concatenate along token/layer/head axes, add normalization, mix K with V, use the two old Readers, or apply sender RoPE. Qwen target RoPE is applied exactly once after fusion; native anchor KV is retained using the established Full-Sync interface.

All six fixed test subsets are evaluated: OBQA128, ARC128, HellaSwag64, GSM8K32, MMLU-Pro128, SciQ64. Scoring, prompt text, candidate policy and GSM greedy4096-token cap remain unchanged. Existing single Llama A+B, single Gemma A+B, Qwen Native and Gemma Native outputs are reused after identity/alignment checks. No training/test selections are made from gold answers.

`manifest.json` pins the four selected A/B checkpoints, prepared inputs, baselines and source dependencies. The two raw source caches are read from their existing experiment paths without overwriting them. Four CPU unit tests check the weighted mean, endpoints, symmetry, dtype, invalid shapes and locality. Real GPU smoke checks reproduce single-branch baseline scores and predictions, verify0/1 endpoints, preserve cache length and validate native raw-K/single-RoPE reconstruction. Every model is frozen and Qwen has no LoRA wrapper.

`experiment.py --launch` starts a detached worker. Failures stop and write failure.json. Completed per-item outputs are atomic; final summary.json includes counts, accuracies and paired correctness against both single branches. Five candidate-scoring tasks run before GSM to avoid blocking their results behind long generation.

Interpretation: the old Llama and Gemma Stage-A data mixtures differ, and this reuses their already-trained branches. MMLU-Pro is the existing custom split. This is a practical equal-mean fusion test, not an equal-budget architectural ablation or a prefill-free latency benchmark.
