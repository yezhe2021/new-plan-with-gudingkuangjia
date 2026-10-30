# Prefix Native Sweep：A/C/D，三任务各64题

复用已完成的分层替换实验中的同一批192题、同一顺序、同一公开header和评分接口，冻结Recolor、A/C/D及两端模型，不训练、不跑GSM8K。Transformer实验保持暂停。设计原稿保存为design.md，运行前判据和解释边界保存为config.json。

每个方法扫描前缀0、6、12、18、24、30、36层原生KV；0层是直接复用的unmasked，36层须逐候选分数或完整生成精确等于直接复用的native。576个unmasked结果和192个native结果不重算；新推理6×3×64×3=3456个配置。替换沿用旧replace()，在post-k_norm/post-RoPE缓存空间进行，仅改原传输Prompt正文KV，保留公开header、位置和新续写，不重算上方正文状态。

输出每组曲线、配对差分及区间、相邻6层的条件边际增益、单调性、原生高于翻译时的Recovery Ratio。每臂/任务独立分类：某段增益≥2倍六段中位数且≥8pp为localized候选；各段均在[-3,+8]pp且七点线性R²≥0.9为distributed候选；两项均不成立或同时成立则ambiguous。ARC/Hella主指标为accuracy，Hotpot主指标EM、F1辅助。整体只有Native高于Translated的主指标曲线标签全部一致才沿用，否则ambiguous。Hella原生不优于翻译的曲线仍完整报告，但不能解释成向原生恢复。

分类只是累积混合缓存曲线的描述，不证明误差本质集中/铺开，也不能证明可部署修复的天花板。前缀仍留下原生/翻译边界；边际不等于旧的单组独立替换效应。64题、多重比较、历史已查看测试集，需要后续独立确认。

保留prefix24与旧G4 V的逐题比较，同时补同类型G4 KV比较和(prefix24-prefix18)-(G4 KV-unmasked)配对交互分析，不增加推理。prefix24优于G4 V本身不能证明错配、也不能证明旧的全部替换结果均为低估。

启动检查不重新生成两份旧基线；prefix00缓存和logits逐位相等、复用结果逐分数相等；prefix36实际生成/打分须与旧native完全一致。逐层检查全部前缀、K/V两个分量、保护header和未命中层，自身替换不变性。原生cached/direct-prefill的一致性由身份核验后的前置smoke继承。正式评测每题每臂还检查prefix36完整结果字典相等。

```bash
python run_pipeline.py launch
tail -f logs/pipeline.log
```

后台prepare→smoke→ARC→HellaSwag→HotpotQA→summarize，错误停止并保留逐配置结果，不自动上传、不自动监控、不恢复其他暂停实验。模型权重/KV不落盘。RESULTS.md包含曲线表、边际表和预注册标签；summary.json还给出包含标签的建议提交说明，待用户明确要求后上传。
