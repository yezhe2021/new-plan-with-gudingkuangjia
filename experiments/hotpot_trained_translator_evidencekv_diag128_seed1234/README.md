# Hotpot-trained Translator evidence-KV diagnostic (128 questions)

Purpose: determine whether a HotpotQA-trained Stage A+B improves evidence-only KV utility and increases the room for accept/reject selection. Both LLMs and the translator are frozen; no Q/O Reader. No selector fitting, stability probes, or new LLM training.

Reuse the first four complete, already shuffled donor blocks of the overnight experiment's test split. Selection does not inspect predictions, answers, or utilities. Verify source IDs and normalized questions against the new checkpoint's 800 training and 200 validation rows. Preserve the exact historical tokens, evidence, positions, donor IDs, three corruption conditions and generation settings. These questions have been evaluated before; this is an exploratory paired diagnostic, not a new confirmatory evaluation. A later formal selector test requires a fresh, disjoint split.

New checkpoints: `grouped_headmix4_residual64_hotpotqa_only_l3_to_q4_seed1234/runs/stage_a/best.pt` and `runs/stage_b/last.pt`. Check their declared hashes, parent manifest identities and Stage-B's frozen Stage-A identity. Copy the old per-question predictions as immutable, hashed references. No checkpoint is selected using this diagnostic.

Generate compact question-only, position-preserving blocked evidence, native evidence, clean translated evidence, wrong-donor evidence, and 50% replaced evidence. The Receiver always reads the real question natively. Verify masked-cache content invariance, native cached/full-prefill agreement and frozen parameters before evaluation. Evaluate all 128 questions without outcome-based stopping. Save per-question outputs atomically and resume completed questions after an explicitly reviewed execution failure.

Report EM/F1, helpful/harmful cases, paired checkpoint differences and gold oracle accept/reject headroom above the best fixed policy. Bootstrap original questions, keeping corruption variants together, 5000 times. Oracle selection is diagnostic only. Include the historical question-gain > 0 policy without fitting. Inference features never use gold; gold is used for scoring and oracle diagnostics only. Balanced synthetic corruption prevalence is not a natural deployment distribution. The full-prompt-trained checkpoint still faces an evidence-only interface shift.

Run on server B with the attnkv Python environment: `python prepare.py` then `python run_pipeline.py`. Outputs: `logs/pipeline.log`, `status.json`, `summary.json`, `RESULTS.md`, and `evaluation/*.json`. No persistent KV tensors or automatic GitHub upload.
