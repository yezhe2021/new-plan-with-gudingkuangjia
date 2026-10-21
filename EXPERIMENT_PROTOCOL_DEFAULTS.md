# Default inference and evaluation protocol

User decision: 2026-10-19. Future KV Writer evaluations default to the pinned
HeteroFold reproduction's task-specific prompt, chat rendering, generation,
stopping and scoring definitions, not merely its chat-template flag.

Reference: `heterofold_official_l3_to_q4_seed20260825` on server B and its
`heterofold/evaluation/{short_context,runtime}.py` and `heterofold/modeling.py`.
The frozen copies in the new template experiment record their SHA256 identities.
This is the HeteroFold repository protocol, not an assertion of lm-eval parity.

For GSM8K:

- User content: `Solve the following math problem carefully. Show your reasoning
  and finish with the final numeric answer in \boxed{...}.\n\nProblem: {question}`.
- Each model uses its own tokenizer template, `add_generation_prompt=True`.
  Qwen uses `enable_thinking=False`. No additional `Answer:` assistant prefill.
- Greedy decoding, maximum 4096 new tokens. Native uses the reference
  `greedy_decode`; transferred caches use the reference `generate` with
  temperature 0. No added `Question:` or `</s>` string stopping rule.
- Primary metric: reference `extract_numeric_answer`: last boxed answer, then
  last #### answer, then last numeric match; reference numeric normalization.
- V100 execution uses the established FP16 hardware adaptation for Llama/Qwen,
  recorded explicitly; this does not authorize reducing Gemma precision.
- Other tasks retain their own HeteroFold settings; do not apply GSM8K's
  4096-token limit or numeric parser to unrelated tasks.

Keep the requested Writer weights, Full-Sync alignment and KV interface fixed
unless the user requests a change. Raw-vs-chat experiments are explicit template
ablations, with all other settings fixed within the comparison. Historical
results remain historical and must not be silently overwritten or relabeled.
Do not automatically expand a simple verification into a four-way template
matrix or multiple Oracle/Native arms: the current user preference is Qwen
Native Chat versus the requested Writer with both ends using the HeteroFold
configuration, unless additional controls are explicitly requested.
Use the same held-out samples across compared conditions; do not tune on test
results. Record exact rendered prompts/IDs, stopping, dtype, backend, source
hashes, checkpoint identities and per-sample outputs.
