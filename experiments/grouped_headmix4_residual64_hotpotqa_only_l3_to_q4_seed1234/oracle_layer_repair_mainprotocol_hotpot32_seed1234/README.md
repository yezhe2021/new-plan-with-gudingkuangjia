# Oracle repair under the original full-prompt main protocol

Preserve the exact parent prepared/test.jsonl rows (first32): original prompt string, Llama and Qwen frame/body IDs, complete Question in the transferred prompt, continuation suffix IDs, fixed Stage A/B checkpoints, no Reader, greedy64 and answer EM/F1. Do not insert evidence wrappers, move instructions, or natively reprocess Question. All32 translated and full-native generated strings/predictions must exactly match the saved main evaluation or stop.

Use tokenizer offsets on the existing rendered full prompt to identify only fully-contained tokens in the original Passage1..N evidence span. Keep instruction, Question and boundary-crossing tokens untouched. Replace K/V only after physical cache assembly, at those columns and selected layers. Mask those exact columns during suffix/generation/NLL in the selected layers. Keep causal masks and absolute positions.

Four baselines: translated; original full native; all evidence positions masked; all evidence positions replaced with native. Full native and all-evidence-native are distinct because the latter retains translated Question/instruction KV. Masking direct evidence access cannot erase evidence already encoded in Question KV, so mask-all is not a true no-evidence/question-only baseline.

Six groups of6 layers, each K-only/V-only/KV and same-location mask.28 configurations x32 =896 generations, plus answer-token-mean Gold NLL. Startup checks: historical baseline parity, native cached/full prefill parity, masked-content invariance, K/V role isolation and zero changes to non-evidence columns. Keep the previous evidence-only experiment as a separate record.

Next128 IDs reserved without inference. No training, head search, automatic confirmation, monitoring, retries or GitHub upload. Current32 remain exploratory, with uncorrected comparisons.

Launch: `/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python run_pipeline.py launch`

Log: `tail -f logs/pipeline.log`
