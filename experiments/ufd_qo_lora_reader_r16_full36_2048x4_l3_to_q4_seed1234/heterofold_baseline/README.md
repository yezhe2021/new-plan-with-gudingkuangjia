# Complete frozen HeteroFold baseline on the same six subsets

No training or new map. Reuse original folded HeteroFold SHA dc2ea778...718e.
Newly score OpenBookQA128, approved custom MMLU-Pro128 and SciQ64. Exact existing
prompts, candidates, labels, numeric parser and held-out rows remain unchanged.
Reuse verified ARC128/HellaSwag64/GSM8K32 historical HeteroFold per-sample results;
do not substitute GSM8K128 scores. Historical GSM8K32 Native outputs are checked
bitwise against current saved Native generations, predictions and token counts.

Preserve ORIGINAL HeteroFold semantics: RAW Llama prompt without chat/BOS;
capture_sender pre-norm states; original folded depth-stencil/recolor map;
build_receiver_cache with native Qwen header and original k_norm/RoPE handling;
native assistant suffix plus same task-specific cue. Do NOT run this map through
our Full-Sync, native first-content-token anchor or post-k_norm Stage-A builder.
This is a comparison of existing complete methods, NOT an equal-training-budget
architecture ablation. MMLU-Pro is the disclosed custom split and A-J label
extension; OBQA and SciQ use the same ARC-style task formatter as all other arms.

Both FP16 Llama/Qwen can reside on this available V100; per-row raw Sender KV
are staged to CPU RAM and discarded, no persistent KV files. Require20GiB free
before loading both, never preempt other jobs. Before new results: replay two
known ARC and two HellaSwag examples, require same prediction and score tolerance;
smoke all3 new tasks and verify Receiver native context IDs remain identical.

The six-arm comparison includes Llama Native, Qwen Native, old MCQ Writer, UFD,
UFD+Q/O-Reader and HeteroFold. Save full raw outputs and all pairwise correct/only
correct counts. Final summary.json and ../summary_with_llama_native_and_heterofold.json
are automatically generated, with a completed entry in the shared results file.
Original earlier summaries are preserved. No new GSM8K generation is required.
