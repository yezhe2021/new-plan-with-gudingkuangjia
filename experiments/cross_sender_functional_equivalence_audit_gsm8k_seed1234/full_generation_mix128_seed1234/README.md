# Frozen GSM8K Llama/Gemma KV mixing: full independent generation

Server A only. Follow-up to the 128-example eight-token cross-sender audit.
No training, new loss, new checkpoint selection or prompt modification.
Reuses the exact two original Stage-A + Hybrid2048x4 frozen Writers.

Pure Gemma, 50/50 Gemma/Llama and pure Llama each independently generate a full
answer (greedy, 256-token maximum), without Native teacher-forced continuations.
Use the old `generation_common.greedy_generate` and `generation_record` directly:
EOS or Question:/</s>/<|im_end|> stop; primary legacy Flexible extraction.
Keep native Qwen token0; mix all aligned body K/V in post-k_norm/pre-RoPE space.
Sender question input is the same `Question: {question}\n`; receiver reads Answer:.
Gemma Sender stays FP32; Llama/Qwen FP16; frozen Writers keep their old arithmetic.

Two-question smoke checks endpoint tensor identity, endpoint eight-token probe
consistency, repeat-run generation determinism and fresh-cache isolation.
Then run 128 test rows, preserving all per-arm text, tokens count, stop reason,
parsed answers and strict/flexible correctness. Compare endpoints to original
saved outputs; report any reproduction disagreement rather than overwriting it.

Summary includes paired correctness, both/only/neither counts, endpoint-oracle
union and mixed success in both-correct/Llama-only/Gemma-only/neither groups.
All groups use endpoints rerun in this experiment, separately reporting old counts.
This is legacy-protocol mechanism analysis, not official lm-eval scoring.

Models load in separate processes/stages. Own temporary captures permit resume;
only those temporary tensor files are removed after all 128 results are saved.
Original checkpoints, old results, scripts and caches are untouched. No monitoring
automation is created. Pipeline exits on code, protocol, memory or CUDA errors.

Run `bash run.sh`; watch `tail -f logs/pipeline.log`.
