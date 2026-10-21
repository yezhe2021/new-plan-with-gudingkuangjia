# Full36 Q/O-LoRA Reader pilot on frozen UFD Writer

One Reader-only run on server B: rank16, alpha16, dropout0, zero-up initialization
on Qwen3-4B q_proj and o_proj in ALL36 layers. Exactly7,667,712 trainable parameters.
Freeze Llama3.2-3B-Instruct, existing MCQ Stage-A, UFD Stage-B step1024 and all
original Qwen weights. No independent Cross-Attention, new memory interface,
Writer update, KV modification, template change, truncation or position change.

The Reader is disabled while constructing Native body KV, native prefix anchors,
Native greedy prefixes and Teacher logits. It is enabled ONLY when continuing
from the physically injected translated cache: native suffix, candidate scoring
and generated tokens. The fixed transmitted K/V remain bitwise unchanged.
Subsequent decoded tokens can create different future KV indirectly because their
hidden states change; this is expected and is not rewriting the transmitted KV.

Reuse the exact audited parent UFD dataset rows/order, source states, native
trajectory token IDs, positions and supervision. OBQA/ARC/GSM8K/MMLU-Pro512 each,
4 epochs, effective batch8, 1024 updates. Native greedy prefixes in epoch1;
exactly50% current Writer+Reader Student prefixes in epochs2-4. Detached sampling,
same exact Native/Student scoring prefix, full-vocabulary KL(Native||Student),
temperature1, FP32 loss, up to8 unique valid positions, training rollout cap64.
No Gold CE, choice KL, attention or KV loss. No new Stage-A or Writer training.
Qwen frozen base executes FP16 on V100. LoRA master weights AND small low-rank
matmuls use FP32, with delta cast to the original projection output dtype. This
avoids generation/autocast dtype mismatch without changing model/cache precision.

## Controlled tests and checkpoints

CPU test: zero initialization equals base; finite/nonzero adapter gradient;
base no gradient; disabling a changed adapter exactly restores the base.
GPU smoke on one example/task: bitwise zero-init UFD logits/top1 identity;
injected prefix KV unchanged; Native AND online Writer+Reader prefix backward;
both Q and O receive finite nonzero gradients; only LoRA parameters trainable;
Native Teacher remains bitwise identical after smoke updates. Smoke weights
are discarded; formal training uses a fresh zero-up Reader.

AdamW lr1e-4, wd0, clip30, GradScaler. Resumable checkpoints every64 updates;
validation every256 updates, retain all four epoch checkpoints. Primary fixed
epoch4/step1024. Best validation macro-KL checkpoint is diagnostic only. No test
selection. Fixed16 validation rows/task: KL and same-prefix next-token agreement,
plus label accuracy on the three MCQ tasks. Full GSM8K task accuracy is measured
in final free-generation testing, not replaced with token agreement.

## Final evaluation

Exactly the same six parent subsets, rendered prompts/IDs, scoring and parser:
OBQA128, ARC128, HellaSwag64, custom MMLU-Pro128, SciQ64, GSM8K32. Reuse checked
parent Qwen Native / old MCQ Writer / UFD Writer per-sample outputs, and compute
ONLY the new Writer+Reader condition. In particular, do not regenerate old MCQ
Writer's expensive/degenerated GSM8K outputs. HellaSwag remains normalized
continuation likelihood. GSM8K remains pinned HeteroFold boxed math instruction,
thinking-disabled Qwen chat, greedy, max4096, reference stopping/numeric parser.
MMLU-Pro remains the approved CUSTOM split and ARC-style A-J label-scoring
extension; NOT an untouched official-test/5-shot-CoT benchmark. No fact-only
inputs or Reader memory branch are introduced. Save full generations, scores,
correctness, generation lengths/caps and paired correct/only-correct counts.

The final pipeline automatically saves summary.json and appends to the shared
../EXPERIMENT_RESULTS.md. Failed/incomplete training is also recorded honestly.
CUDA allocator peak allocated/reserved memory is recorded separately from
whole-device/process physical nvidia-smi memory; never conflate those measures.

## Ordered launch

run_pipeline.py first performs the CPU/data/hash audit. Then it waits for the
parent UFD experiment's completed marker AND process exit, plus20GiB free and
CUDA availability, checking every5minutes. No GPU model is loaded while queued.
It never stops/preempts other jobs. On readiness: smoke -> train -> six Reader
evaluations -> automatic summary. Lock prevents duplicate pipelines. No ongoing
chat monitor or additional training ablations are scheduled. Reuse existing
source states; do not capture or retain a second full KV cache dataset.
