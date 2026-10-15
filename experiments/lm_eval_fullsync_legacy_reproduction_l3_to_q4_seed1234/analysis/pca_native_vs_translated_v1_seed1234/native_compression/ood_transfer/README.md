> Completed scope: HellaSwag24 + MMLU-Pro13 =37 samples. The user omitted the final psychology sample; its two partial outputs are preserved in `excluded_partial` and excluded from metrics. Original config/manifest retain the planned38-sample scope; see `user_stop_request.json` and the final report.

# 冻结 ID PCA 基的 OOD 压缩实验

服务器 B；不重新拟合 PCA，不训练模型。HellaSwag24 条使用原 OpenBookQA/ARC 校准基；MMLU-Pro14 个学科各随机1条，使用原 GSM8K 校准基。样本从已有官方 OOD pilot 日志中按 seed1234 抽取，运行前固定，不按答案/得分筛选。

每项均测试原生未压缩和 K/V/KV 各保留16/32/64/96成分，共13条件。保持 token 数量、位置、原生 anchor 和最后读出。新生成 KV 按原协议自然产生，不继续压缩。原始 KV 不落盘。

HellaSwag 调用官方候选续写似然与 acc/acc_norm；MMLU-Pro 沿用官方逐步推理生成、custom-extract 与 exact_match。Prompt/生成参数直接取父实验 qwen_native evaluation_ood 日志。MMLU-Pro 原预算为2048 token，本次不修改；ID GSM8K 为256，因此跨任务比较有格式、长度、任务难度等混杂，不能将差异全归于 OOD。

主要比较同一批样本压缩前后的分数变化，同时报告首 token KL、保留的中心化能量、逐条预测、生成长度/是否达到预算。用配对 bootstrap 描述每组下降幅度的不确定性，并与对应 ID 组比较。不是完整官方 benchmark。

CPU 计分回放通过后等待现有 ID 实验完成，再检查 GPU 资源、原生缓存注入及128维重建。生成检查仅使用32 token，不改变正式评测预算。检查通过后自动运行全部38条。

```bash
cd /hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/native_compression/ood_transfer
/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_ood.py
tail -f logs/pipeline.log
```

`manifest.json` 固定样本、原始日志、任务配置、脚本、父实验和 PCA 基的哈希；每条每条件即时保存，支持输入不变时继续。自动生成 summary/REPORT/图，并更新共享实验摘要。
