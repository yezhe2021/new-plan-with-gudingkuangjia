# Qwen3-4B → Gemma3-4B: Full-Sync DST with a Rank-64 Stage 2

| Task | Metric | Gemma Native | DST Stage 1 | DST+Rank64 | Rank64 − Stage1 | Rank64 − Native |
|---|---|---:|---:|---:|---:|---:|
| arc | correct | 75.00% | 79.69% | 79.69% | +0.00 | +4.69 |
| hotpotqa | em | 53.12% | 17.19% | 48.44% | +31.25 | −4.69 |
| hotpotqa | f1 | 69.61% | 26.86% | 62.25% | +35.40 | −7.35 |
| 2wiki | em | 26.56% | 10.94% | 21.88% | +10.94 | −4.69 |
| 2wiki | f1 | 37.00% | 20.62% | 34.32% | +13.70 | −2.68 |
| musique | em | 29.69% | 6.25% | 10.94% | +4.69 | −18.75 |
| musique | f1 | 38.21% | 9.90% | 18.74% | +8.84 | −19.47 |
| sciq | correct | 100.00% | 96.88% | 100.00% | +3.12 | +0.00 |
| mmlu_pro | correct | 35.94% | 32.81% | 34.38% | +1.56 | −1.56 |
| hellaswag | correct | 45.31% | 51.56% | 56.25% | +4.69 | +10.94 |
| gsm8k | correct | 75.00% | 0.00% | 53.12% | +53.12 | −21.88 |
| gsm8k | legacy_correct | 76.56% | 7.81% | 56.25% | +48.44 | −20.31 |
| **mean** | | **55.08** | **36.91** | **50.59** | **+13.67** | **−4.49** |

Paired over all 512 test rows with 4,000 bootstrap resamples: Rank64 − Stage1
+13.67 [+9.96, +17.38], Rank64 − native −4.49 [−8.59, −0.39]. Per task, the two
intervals that exclude zero against native are musique −18.75 [−31.25, −6.25] and
gsm8k −21.88 [−34.38, −9.38]; hellaswag +10.94 [+1.56, +21.88] is the only one
above native. Recovered 43 rows against native, broke 66, net −23.

- Direction is the reverse of `gemma_dst_rank64_stage12_server_a_seed1234`: 36
  Qwen3-4B layers feed 34 Gemma-3-4B-it layers through 34 taps (sender layers 9
  and 26 are skipped), 256-wide carried state, 1024 source and target width.
- Stage 1 is 111,515,648 parameters, trained 4 epochs at 3e-4 on mean per-layer
  K/V normalized MSE in native pre-k_norm/pre-RoPE space, full-body tokens.
- Stage 2 freezes Stage 1 epoch 4 and trains 8,912,896 parameters, 34 pairs of
  bias-free 1024→64→1024 GELU residuals on K and V, 4 epochs at 1e-4 with
  lambda_kl 0.5.
- Training rows are ARC/HotpotQA/GSM8K at 1024 each, the same raw rows as
  `rnkv_keep_both_all36_arc_hotpot_gsm_seed1234`; validation is 64 per task and
  diagnostic only. Fixed final epoch 4 at each stage, no test-time selection.
- No memory and no Recolor arm in this directory. Recolor and HeteroFold R16 on
  the same pipeline are in `heterofold_qwen_to_gemma_paper_seed20260825`, which
  pins this run's `manifest.json`, `experiment.py` and `prepared/test.jsonl`.
- Stage 1 scores 0.00 on gsm8k under the current scorer and 7.81 under
  `legacy_correct`: it emits no parseable final answer, not wrong ones. The
  Rank64 residual is what restores generation to 53.12.
