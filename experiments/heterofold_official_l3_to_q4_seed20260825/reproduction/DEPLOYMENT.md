# Server B deployment (2026-10-14)

Directory: `/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825`

Verified before formal launch:
- Official 12 CPU tests passed; 2 adaptation tests added for the final pipeline.
- Official LongBench files verified against upstream SHA256.
- Official 2000-prompt manifest digest matched upstream exactly.
- Actual models: Llama geometry 28 x 8 x 128, Qwen geometry 36 x 8 x 128.
- FP16 native-cache first-step argmax matched, logit MAE 0.0089216661.
- Tiny diagnostic calibration completed 2 updates with finite gradients;
  folded transfer produced finite logits. This is not a benchmark result.

Formal process launched using `bash reproduction/run_pipeline.sh`.
The wrapper resumes only completed official stages/epochs and holds an exclusive
lock. It exits on error rather than stopping other tasks or changing precision.

Log on server B:
```bash
tail -f /hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/logs/pipeline.log
```

Expected final artifacts: `environment.json`, `maps/llama3_to_qwen4/`,
`results/{native,recolor,heterofold}/{arc,gsm8k,qasper}/items_00000_00032.json`,
`results/summary.json`. No recurring monitor has been created.
