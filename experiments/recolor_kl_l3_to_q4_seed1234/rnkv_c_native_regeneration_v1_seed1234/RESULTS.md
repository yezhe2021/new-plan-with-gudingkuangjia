# RN-KV v1: fixed epoch4/step1024

| Method | ARC64 accuracy | Hotpot64 EM | Hotpot64 F1 |
|---|---:|---:|---:|
| C | 67.19% | 29.69% | 46.53% |
| rnkv | 64.06% | 35.94% | 49.80% |
| native | 87.50% | 56.25% | 68.52% |

- Original C epoch4/step800, not the queued dependency's fine-tuned weights. Sender, Recolor, all C corrections and Qwen weights are frozen. Only32 prompt embeddings and4 memory-only Rank16 adapters train (409600 parameters).
- Construction permits all translated body plus causal memory positions. Answer queries in19-30 forbid translated body and read memory; other layers forbid memory and retain body. Header and causal suffix/generated tokens remain visible everywhere. No blending, output fusion, physical cache replacement, or post-RoPE relocation.
- Memory K/V are captured from Qwen's actual native projection/cache update at each layer's input state. Adapters after block l can affect later layers' memory states, never retroactively rebuild layer l K/V. Memory positions L..L+31; answer suffix starts L+32.
- Task loss always uses gold context: ARC negative log likelihood of the gold option-label tokens with the official label cue (full-vocabulary CE, not option-normalized classification); Hotpot all gold answer tokens plus one im_end EOS. Generated wrong prefixes are used for KL only, not paired with misaligned gold CE.
- KL compares Native and Student on the identical semantic prompt and generated/gold prefix, full vocabulary, at most8 of64 prefix positions. Native teacher has original positions; Student's suffix is necessarily shifted by32. No artificial memory or positional holes are added to Native/C baselines.
- Train ARC official train1024 and Hotpot train1024; ARC validation from official dev and Hotpot validation from a disjoint train subset. Exclude parent calibration and all parent/dependency test questions/available source IDs. Preserve all distractor passages; no length-based selection or truncation. Fresh test ARC64/Hotpot64 exactly reuse dependency's fixed input rows for within-question comparisons.
- Full Prompt transfer includes the question and ARC choices. This does not establish an evidence-only private communication protocol or generalization to tasks outside the supervised ARC/Hotpot mixture.
- CPU-only data preparation/tests may run while dependency occupies GPU. GPU smoke and training wait for dependency completed markers, successful runner exit and GPU release. Dependency failure/pause stops the queue. Other paused experiments stay paused; no automatic upload or ongoing notifications.
