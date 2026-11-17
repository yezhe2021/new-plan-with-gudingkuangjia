# DST Gated-Dynamic Handoff Latency, Batch 2, 4k → 32k

Llama3.2-3B → Qwen3-4B, DST rank-64 body, gated dynamic memory (M=32, 36 gates) from
`dst_rank64_gated_joint_m32_a4_seed1234/initial/runs/epoch4_step1536.pt`
(`4547b751…`). One V100-SXM2-32GB, torch 2.5.1+cu121, batch 2, chunk 2048, 3 trials per
batch slot (n=6 timings per cell), 32 decode steps.

Prompts are nested prefixes of four disjoint concatenations of QuALITY dev articles, so the
4096 / 16384 / 32768 rows read the same text and differ only in how much of it is handed over.
The timer starts with the CPU-side Sender K/V already materialised. It covers Full-Sync
(alignment, gather, GPU upload, translation, cache finalise, memory construct) plus the
receiver suffix and first token. The Sender forward is outside the timed region, and both
frozen models stay resident on the GPU.

## Handoff latency and decode

| Body tokens | Arm | TTFT p50 (ms) | TTFT p95 (ms) | Decode p50 (ms/step) | Peak allocated (GiB) |
|---:|---|---:|---:|---:|---:|
| 4096 | native | 2717 | 2736 | 104.4 | 15.71 |
| 4096 | recolor_c | 1812 | 2138 | 104.7 | 17.69 |
| 4096 | **gated dynamic** | **1730** | 2180 | **64.4** | 16.84 |
| 16384 | native | 18019 | 18104 | 326.9 | 20.91 |
| 16384 | recolor_c | 7467 | 8497 | 326.3 | 21.06 |
| 16384 | **gated dynamic** | **6475** | 6947 | **155.3** | 20.21 |
| 32768 | native | 57802 | 58158 | 610.5 | 21.84 |
| 32768 | recolor_c | 13879 | 14776 | 609.4 | 25.56 |
| 32768 | **gated dynamic** | **13813** | 16406 | **288.4** | 24.71 |

Relative to native prefill, gated dynamic costs 0.64× the TTFT at 4k, 0.36× at 16k and 0.24×
at 32k: the gap widens with length while the absolute cost grows only 8× for an 8× longer body.
Decode is 0.62× / 0.48× / 0.47× of native. Native prefilled in one shot at every length
(`prefill_blocks = [null, null]`), so the 57.8 s at 32k is not a blocked-prefill artefact.

Gated dynamic and recolor_c are level on TTFT at 32k (13813 vs 13879, inside the spread) but
gated dynamic holds a 2.1× decode lead. Native keeps the smallest peak at 32k (21.84 GiB):
both DST arms pay roughly 3 GiB extra to hold the translated cache next to the raw body.

## Where the 13.8 s at 32k goes

Phase medians for the gated dynamic arm, milliseconds. `gather_and_upload` moves the translated
K/V from host to device; nothing else is within an order of magnitude.

| Phase | 4096 | 16384 | 32768 |
|---|---:|---:|---:|
| alignment | 227.8 | 1397.7 | 2482.6 |
| cache_allocate | 2.3 | 2.3 | 4.9 |
| gather_and_upload | 982.2 | 3429.1 | **8068.1** |
| translation | 234.4 | 942.7 | 1909.2 |
| cache_finalize | 53.6 | 218.4 | 440.5 |
| cache_handle | 0.7 | 0.8 | 1.8 |
| memory_construct | 118.8 | 223.2 | 385.6 |
| receiver_suffix_and_head | 84.4 | 171.8 | 319.7 |
| sum of phases | 1704 | 6386 | 13612 |
| measured TTFT p50 | 1730 | 6475 | 13813 |

`gather_and_upload` is 57.6% of TTFT at 4k, 53.7% at 16k and 59.3% at 32k. The tower is
transfer-bound, not translation-bound: the 36-gate memory adds 119 / 223 / 386 ms and the
rank-64 projection adds 234 / 943 / 1909 ms, together under a fifth of the total. Any further
speedup has to come from moving fewer bytes or moving them asynchronously.

Receiver cache bytes grow 1.13 → 4.50 → 9.00 GiB across the three lengths while the memory bank
stays at 9 MiB, i.e. the bank is 0.1% of the handoff at 32k.

## Follow-up: affine folding and fused readout

`batch2_quality/affine_folding/` runs two code-level changes against their own originals —
a pre-folded recolor affine and an in-place `addmm_` residual readout — on the same prompts and
checkpoints, four arms interleaved per trial. Recolor folding saves 0.56 s at 16k
(−8.1%, interval [−736, −321] ms) and is inside the noise elsewhere; DST fusion is bit-exact
(0.0 logit drift) but never separates from zero at n=6. Neither moves peak memory. See
`batch2_quality/affine_folding/RESULTS.md`; that run unloads the Sender, so its peaks and
absolute times are not comparable to the table above.

## Caveats

- The 32k column runs with the raw body in file-backed CPU memory. Per-trial storage reads were
  2.77 GB / 57 KB / 823 MB / 700 KB / 0 / 1.81 GB with 10893 / 1 / 3494 / 7 / 0 / 7208 major page
  faults, so the p50 is a warm-page-cache number; a cold trial is what produces the 16406 ms p95
  (+19%). Treat 32k as "warm host, uncontended card".
- This is a latency experiment only. It re-uses the frozen gated-dynamic checkpoint and reads no
  task accuracy; nothing here revises the +1.76 pp on 8-task mean or the 4% gate magnitudes
  reported in `dst_rank64_gated_dynamic_m32_a4_server_a_seed1234`.
- Batching does not perturb the handoff numerically: max relative L2 between the batch-2 and
  batch-1 memory K/V is 0.00259 (`batch2_quality/batch_parity.json`).

## Batch-1 reference (8k, two articles)

`summary.json` at the root of this directory is the earlier batch-1 run on the 8k E4 workload:
handoff TTFT p50 1403.5 ms (p95 1760.0), text TTFT p50 2889.1 ms (p95 3170.3), decode
67.7 ms/token (p95 70.6), peak 19.14 GiB, translated K/V 1.21 GB, memory bank 4.5 MiB,
construct peak delta 4.56 GB. The batch-2 4k row above is not comparable to it: different
prompt set, different batch width.

## Attempt history

`batch2_quality/attempts/` keeps the three configurations that did not ship.

| Attempt | Outcome |
|---|---|
| `initial` | died mid-measure at 4096, batch 1 |
| `baselines_shared_oom` | `Native prefill or decode could not fit with the smallest block` — native, recolor_c and gated dynamic could not be resident together with both models |
| `resume_manifest_type` | `AssertionError` while re-deriving the manifest after a resume |
| `sender_unloaded` | ran with the Sender unloaded and capture outside the timer |

The shipped configuration is the one in `batch2_quality/manifest.json`: both frozen models
resident, Sender capture outside the timed region, per-length prompt set pinned by hash.
