# Qwen 原生 KV 的 PCA 压缩对照

在服务器 B 运行，复用上一轮 PCA 的精确样本划分：OpenBookQA、ARC-Challenge、GSM8K 各 8 条校准、24 条独立分析样本。MCQ 共用一个校准基，GSM8K 单独拟合。

每层每个 KV head 分别拟合中心化 PCA。对每个外部 token 重建 `mean + (x-mean) U_k U_k^T`，k 为 16/32/64/96。分别压缩 K、V、KV，共 12 个干预条件，另有原生未压缩对照。128 维完整重建用于实现检查。所有外部位置均保留，token0 anchor、最后原生读出和后续生成缓存保持原协议。K 在原生 k_norm 后、RoPE 前压缩，不重复归一化。

仅用原生校准数据拟合基，不用测试样本选择 k，不训练模型。校准集是此前已声明的官方评测 pilot 子集，不能把本实验结果表述为完整 benchmark 或独立训练集校准结果。

使用原日志中的精确请求，调用固定 lm-eval 0.4.12 的 HFLM 生成/候选似然以及官方 filters/process_results。GSM8K 重新生成、保存完整输出并计算 strict/flexible exact match；MCQ 重新计算候选似然、acc/acc_norm。全词表首 token KL 单独作为机制指标，不能代替完整生成质量。

脚本自动执行 CPU 投影检查、原生注入检查、128 维重建检查，以及每组第一条分析样本的短生成/候选似然检查。检查通过后继续全部 72 条分析样本；逐条件写入输出，支持同一 manifest 下断点继续。只保存 PCA 均值/基/特征值和 JSON/图表，不保存原始 KV。

```bash
cd /hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/native_compression
/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_compression.py
tail -f logs/pipeline.log
```

完成后自动生成 `summary.json`、`REPORT.md`、曲线图并更新共享 `EXPERIMENT_RESULTS.md`。`metrics/*_spectra.json` 的谱由独立分析样本的原生协方差和校准投影矩阵计算，标明不含最后 FP16 舍入；逐样本 `retained_centered_energy` 则直接测量实际重建值。
