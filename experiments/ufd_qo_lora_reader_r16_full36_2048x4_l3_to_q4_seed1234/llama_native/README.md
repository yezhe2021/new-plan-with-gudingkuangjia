# Llama Native completion of the six-task UFD / QO-Reader comparison

No training. Load only frozen Llama-3.2-3B-Instruct on server B, FP16 V100.
Reuse exactly the same544 test rows: OBQA128, ARC-C128, HellaSwag64,
custom MMLU-Pro128, SciQ64 and GSM8K32. No facts/support are added.

Use the existing recorded Llama Sender chat template and its pinned template
date, complete user prompt, native assistant suffix and same task answer cue.
Tokenize complete rendered text without duplicated special tokens. Do not use
the Qwen template. Native Llama sees the FULL task context: no translation,
token selection, Full-Sync mapping, external memory, Stage-A/B or Reader.

Call pinned HeteroFold native_candidate_scores for labels and HellaSwag
length-normalized continuations. MMLU-Pro retains the previously disclosed
custom split and A-J label scoring, not official5-shot-CoT. GSM8K uses pinned
greedy_decode, complete boxed math instruction in native Llama chat, max4096,
reference stopping and numeric parser. Do not introduce legacy stop strings.

Smoke checks: exact rendering/token IDs; official candidate scoring versus
selected-position manual LM-head scoring on each of the five MCQ/continuation
tasks; Native first-token logit/argmax parity for GSM8K. Do not select/tune on
test labels. Save full Native generations, scores, rendered prompts and IDs.

Reuse the four existing Reader experiment arms unchanged; add llama_native.
Automatically save all five arms and every pair's correctness counts in
summary.json, plus ../summary_with_llama_native.json. Original parent summaries
are preserved. Append the completed follow-up to the shared results document.
One model load, no repeated baseline generation, no new persistent KV cache.
