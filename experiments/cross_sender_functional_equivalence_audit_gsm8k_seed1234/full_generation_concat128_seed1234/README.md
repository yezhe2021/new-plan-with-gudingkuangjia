# Frozen Llama/Gemma KV concatenation, 128 GSM8K examples

Server A only, no training. Same questions, checkpoint identities, Prompt,
Full-Sync alignment, model/Writers arithmetic, 256-token greedy generation,
legacy stops and Flexible parser as the immediately preceding mix128 run.

New arms: L+L, G+G, L+G, G+L. Each has one native Qwen token0 plus two
N-token translated body blocks; no separator, source ID or extra native
question is introduced. Both blocks describe the SAME question. Matched
L+L/G+G control doubled memory length and repeated content. They do not
guarantee an exactly equivalent readout to single-source memory.

Always concatenate post-k_norm, pre-RoPE K, then call the unchanged native
cache constructor with continuous positions. No duplicate token0, no second
key normalization, no concatenation of already rotated keys. The second block
was encoded independently; it is not a fresh prefill conditioned on block one.
V has no RoPE. Answer suffix begins after both blocks.

Reuse the three completed baseline outputs (L, G, averaged 50/50) as archived
reference results, not newly generated arms. Verify exact 128 IDs/questions,
old manifest source hashes and parser reconstruction. Two-question smoke also
reruns endpoint generations for exact baseline reproduction, validates shape,
block identity and re-positioned cache values/keys, and repeats one concatenated
generation to detect mutable-cache contamination. This is a smoke gate, not a
guarantee of bitwise endpoint reproduction on all 128 examples.

Formal results retain per-arm complete text, parser outputs, counts and stop
reasons. Summary includes every pair among seven arms, paired bootstrap intervals
for accuracy differences (GPU, 10000 resamples), and each arm's retention in the
four endpoint correctness groups. Interpretation must prioritize L+G/G+L versus
same-length G+G/L+L, and disclose order sensitivity and all four arms without
choosing a test-optimal direction as an unbiased new method.

`capture_driver.py` is an unchanged copy of the preceding run's capture helper;
only its capture routines are used here. It resolves original asset paths from
the parent audit config and writes intermediates under this experiment only.
Models are loaded in separate stages, Gemma FP32 and Llama/Qwen FP16. CPU math
threads limited to one. Own temporary tensors are removed only after successful
formal result saving; original data/checkpoints and prior outputs stay intact.

Run `bash run.sh`; inspect `tail -f logs/pipeline.log`.
