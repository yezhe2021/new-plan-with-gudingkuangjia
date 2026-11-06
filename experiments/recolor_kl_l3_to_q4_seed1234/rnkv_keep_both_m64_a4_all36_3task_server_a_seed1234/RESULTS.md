# M64 vs M32, four Rank16 adapters, all-layer Keep-both, three tasks

| Task | Metric | C | M32+A4 | M64+A4 | Native |
|---|---|---:|---:|---:|---:|
| arc | correct | 67.19% | 70.31% | 70.31% | 87.50% |
| hotpotqa | em | 29.69% | 46.88% | 45.31% | 56.25% |
| hotpotqa | f1 | 46.53% | 61.09% | 58.85% | 68.52% |
| 2wiki | em | 9.38% | 32.81% | 32.81% | 32.81% |
| 2wiki | f1 | 22.05% | 43.95% | 43.63% | 43.45% |
| musique | em | 1.56% | 21.88% | 20.31% | 25.00% |
| musique | f1 | 7.64% | 27.79% | 30.83% | 33.07% |
| sciq | correct | 98.44% | 98.44% | 98.44% | 100.00% |
| mmlu_pro | correct | 25.00% | 21.88% | 32.81% | 53.12% |
| hellaswag | correct | 57.81% | 60.94% | 59.38% | 45.31% |
| gsm8k | correct | 70.31% | 75.00% | 71.88% | 89.06% |
| gsm8k | legacy_correct | 70.31% | 75.00% | 71.88% | 89.06% |

Execution history: CPU saved-tensor/native-KV offload was used through accepted step24; from step25 onward GPU-resident execution was used at the user request, preserving checkpoint/optimizer/scaler/RNG. See controls/GPU_RESIDENT.md and controls/use_gpu_resident.json. Configuration caveats below describe the initial execution plan.

- OnlyM64+A4 arm is authorized. All36 answer layers read translated body+Memory; adapterafter6/12/18/24,rank16. Compare with fixed M32+A4 baseline on same3072train/192val/512test and4epochs/1536steps.
- Freeze existing Llama3.2-3B→Qwen3-4B Recolor+C, same RAW/token/layer alignment and pre-k_norm/pre-RoPE conventions. No translator retraining or weight quantization.
- 64 Memory KV occupy positionsL..L+63; suffix startsL+64. No fixed32 offsets anywhere in this arm. Prompt first32 rows and four adapter initial states are exactly matched to the original M32 initialization; additional32 prompt rows use independent fixedseed7634.
- Cache contains original ExampleBuilder.recolor_outputs roundedFP16, identical to original C.Engine.base. Cache source, weights, rows and coordinates pinned. Source and Qwen stages never coexist on GPU. Native body is teacher/baseline only and never supplies student Memory.
- Offloading preserves gradients through frozen Qwen; native KV is moved to CUDA only for no-grad teacher/prefix forwards, released before student CE/KL backward. CPU tests compare execution ordering loss/gradients; real GPU smoke includes longest samples and actual consecutive batches.
- If current shared GPU cannot fit the capped probe, retain cache and wait for more free memory; never skip/truncate examples, shrink Memory or kill external workloads. Queue continues detached after success without local computer online.
- Full gold CE +.5full-vocab KL, same3-task quotas, batch8,lr1e-4,4epochs/1536 accepted updates, same-batch AMP overflow retry. Host transfers may slow throughput; total runtime requires measurement.
- Eight fixed64-question tasks,QA64/GSM512, both GSM numeric metrics. Reuse pinned C/Native/M32 predictions only after same-model parity checks. Historical test cohorts have been inspected; one seed is exploratory.
