# All-layer Keep-both: inference-only mask intervention

User selected: modify only the existing Keep-both inference mask and evaluate the original seven tasks. Do not retrain and do not run GSM8K.

Use the same independently trained Keep-both final epoch4/step1024 checkpoint. Answer-time layers1-36 now see both translated body and32 Memory KV; the original version exposes memory only in19-30. Body remains visible in every layer in both versions. Public header and causal answer/suffix/generated tokens remain readable. There is no native evidence prefill, learned gate, cache overwrite or output fusion.

Construction, four adapters, memory tokens, normalization/RoPE, cached positions and answer offset remain unchanged. A temporary context changes only `routing.replace`, restores it after every score, and asserts restoration. On the first example of every task, reproduce the original partial-layer Keep-both scores/generation, verify bitwise identical memory construction under the new setting, check all36 causal masks, and reuse the new-arm output in evaluation.

The seven original fixed64-question cohorts are copied byte-for-byte: ARC, HotpotQA,2Wiki,MuSiQue,SciQ,MMLU-Pro,HellaSwag. Reuse existing label/ending-likelihood and64-token QA scorers. Reuse source-hashed per-question C,original RN-KV,partial-layer Keep-both and Native baselines. Save new predictions, per-task paired bootstrap differences, rescue/harm IDs, caps and complete text.

This is a train/test routing intervention: weights were trained with answer memory visible only at19-30. Its outcome must not be described as all-layer Keep-both training. All parameters remain frozen, parameter versions and absence of native student prefills are checked after every sample.

The pipeline runs detached under an exclusive lock. A pause flag stops at sample boundaries. It does not resume any other paused experiment or upload anything automatically.
