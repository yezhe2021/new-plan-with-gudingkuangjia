# KV功能敏感性机制诊断：第一阶段

冻结现有A/C/D、Recolor、Llama3.2-3B和Qwen3-4B，仅分析原ARC64、HotpotQA64，固定G1(1–6)、G4(19–24)、G5(25–30)。不训练新参数，不扩MMLU-Pro/独立128题确认集，不跑HellaSwag/GSM8K。

四种组合TT、NT、TN、NN分别表示翻译/原生K与V。主上下文是全部标准答案token；辅助上下文是未修复接收方实际greedy生成的最多64个token，在最多8个固定位置比较。每次只替换一个组、保留同题同位置的header和其他层，重新执行suffix/continuation。

诊断分四条路径：

1. 固定Native Q与非正文KV尾部，离线计算四种组合的Attention Route KL/TV、post-Wo Output NMSE和K/V交互。
2. 固定未修复Translated Receiver Q与相应尾部，重复相同计算。
3. 实际Receiver自然前向：捕获每个修复配置实际产生的Q和o_proj输出，记录Q差异及全词表KL、目标token NLL、ARC正确选项margin。
4. 在同一功能前向中将目标组的探针q_norm输出固定为Native Q，重新比较四组合，避免Query变化被解释成K的独立影响。

Q固定发生在q_norm之后、RoPE之前，之后走正常Qwen旋转；缓存替换发生在最终post-k_norm/post-RoPE空间。Question随原Prompt缓存传输，因此不会把Native-prefill Question Q冒充Translated推理时的Q。实际-Q与固定-Q指标分开解释。

自由生成准确率、Hotpot EM/F1及原选项评分从前置Oracle逐题结果复用；不以KL或NLL替代真实任务效果。Primary统计使用未修复状态的指标预测修复收益，不混合四种组合制造机械相关。按任务/方法/上下文分层，控制层组秩效应，整题cluster bootstrap1000次，提供逐题数据与组内相关。

探索性A信号：两个任务均至少2/3方法在Gold上下文中Route相比KV-L2的rho优势95%区间大于0，且Student-prefix方向一致。B信号还要求Output相比L2优势及兼容性synergy与功能synergy相关稳定。两者均不满足时报告C=当前已选样本缺乏稳定支持，不宣称机制不存在。A/B可共存；独立128题确认不自动启动。详细预注册指标、方向及限制见config.json。

已完成Wrong-Donor相位修正版将做身份核验和独立公式检查，不引用旧无相位修正的数字，也不重新启动donor实验。即便相位修正通过，长度重采样仍不是纯内容单因素实验。

启动检查覆盖固定Q因子分解恒等式、捕获/自Q固定logits逐位一致、完整注意力输出重建、原缓存与header保护、实际自生成前缀重放、最长正文与最高Gold探针成本。只保存JSON指标/Token前缀，不保存KV/Query/Attention张量。

```bash
python run_pipeline.py launch
tail -f logs/pipeline.log
```

服务器后台prepare→smoke→ARC64→HotpotQA64→统计汇总。失败停止、逐组保存。Transformer和融合任务保持暂停，不自动上传，不持续通知。
