# RN-KV adapter-off inference ablation

| Method | ARC64 accuracy | Hotpot64 EM | Hotpot64 F1 |
|---|---:|---:|---:|
| C | 67.19% | 29.69% | 46.53% |
| rnkv | 64.06% | 35.94% | 49.80% |
| adapter_off | 26.56% | 3.12% | 6.40% |
| native | 87.50% | 56.25% | 68.52% |

- Inference-time ablation of the existing jointly trained checkpoint. This does not measure a model trained without adapters, or isolate the total value of adapters during learning.
- Adapters are disabled only by exact bypass, with no zeroing, refitting, merging or checkpoint mutation. Learned memory embeddings remain unchanged. Every construction must bypass exactly one adapter at each of blocks6/12/18/24.
- Same128 fixed test rows and original protocol as the completed RN-KV experiment. Reuse its immutable C/full-RN-KV/Native per-question baselines, with enabled-model parity checked before ablation. No GSM8K, new data, retraining or outcome-based checkpoint selection.
- Memory still contains32 tokens at positions L..L+31; suffix starts L+32. Construction reads translated body; answer layers19-30 use memory and other layers retain body. No native body KV or hidden state enters student inference.
- Paired intervals/rescued-harmed cases use whole questions, n64/task. Parent result files and weights remain unchanged. Detached fail-stop process; other paused experiments stay paused; no automatic upload or continuous monitoring.
