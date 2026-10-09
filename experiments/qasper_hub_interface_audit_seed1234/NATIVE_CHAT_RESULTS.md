# QASPER Native Chat Repair v1

Server B completed the native-only run on the same16 dev questions as experiment0.
No training, checkpoint update, test tuning, truncation or answer-parser changes.

Changes are a bundled repair, not an isolated chat-template ablation: Qwen native
chat template, explicit system answer-only instructions, enable_thinking=False.
Same128-token cap, greedy decoding, native EOS, FP16 eager model, official Answer
F1 on raw stripped output. This is a dev pilot, not full benchmark performance.

| Condition | Answer F1 (0–100) | EOS rate | Mean generated tokens | Limit-hit rate |
|---|---:|---:|---:|---:|
| Previous plaintext complete prefill | 11.04 | 0% | 128 | 100% |
| Native chat complete prefill | 44.99 | 100% | 17 | 0% |
| Native chat evidence-cache reuse | 44.99 | 100% | 17 | 0% |

Full vs cached generated token sequences agree exactly on16/16 questions. Cache
split is computed on the rendered chat prompt's actual full tokenization, before
the question; cache contains system/user prefix and paper, and reader computes
the question and assistant generation prefix. Position/mask handling is delegated
to HF generate. No independently tokenized fragments are concatenated.

Full native Answer F1 by evaluator best-reference type: extractive59.70,
abstractive18.23, boolean16.67, unanswerable75.00. The selected four-per-first-type
manifest and the evaluator's best-reference type need not coincide for questions
with disagreeing annotations; these small category scores are diagnostic only.

The improved native protocol supports usable answer behavior and native-cache
reuse on this pilot. It does not prove full-length/general-model Hub compatibility
or writer performance under chat wrappers. Historical plaintext results remain
unchanged. Do not attribute the whole improvement to one component of the bundle.

Remote raw configuration, prompts, generated IDs, answers, predictions and summary:
`/hy-tmp/yezhe/异构模型/qasper_hub_interface_audit_seed1234/results_native_chat_v1/`.
