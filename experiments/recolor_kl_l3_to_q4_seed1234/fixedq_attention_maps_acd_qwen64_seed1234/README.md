# 固定Q：A/C/D与Qwen原生注意力图关系

按用户确认，只比较固定Native Qwen Query。等待`kv_functional_sensitivity_phase1_acd64_seed1234`成功完成并释放显存后自动运行。沿用ARC64、HotpotQA64，A/C/D/Recolor和LLM冻结，不训练，不跑GSM8K/HellaSwag，不新增实际Q或四条件功能诊断。

在同一道题全部Gold答案预测位置提取Native Q；所有方法共享Query、Token位置、因果mask、Native header与continuation尾部K，仅替换正文K来源。注意力概率本身不直接使用V；因此本实验专门分析K的路由关系，不声称衡量完整KV功能。原生注意力只是参考，不等于正确证据或答题正确性。

覆盖全部36层、32个Query Head(GQA中8个KV Head)，输出逐题逐层/逐头KL(Native||method)、JS、TV、Cosine、正文Top10重合率及四方法两两关系、A/C/D误差向量方向、结构区域注意力质量。与Phase1重叠18层固定NativeQ的Gold routeKL作一致性检查，不重跑旧功能诊断。

图：

- figures/layer_curves.png及SVG：逐层KL、TV曲线，整题bootstrap95%区间。
- figures/layer_head_kl.png及SVG：36层×32个Query Head差异，共享对数色标。
- figures/pairwise_relations.png及SVG：Native/A/C/D两两JS与Cosine。
- 固定每任务第0、32条样例，第3、21、27层：四方法并排首答案Query的head×Token图，以及所有Gold探针平均的head×64正文位置bin图。样例不根据效果选择，四方法使用相同Token列与每层共享色标，保留原概率，不对选中Token重新归一化。

每题先平均头和探针，再汇总64题并按整题bootstrap1000次，不把头/Token当独立样本。任务准确率/Hotpot F1从旧逐题基线复用，注意力更接近Native不自动解释为方法更优。所有样本都是已有探索性样本，不是新的独立确认集。

```bash
python run_pipeline.py launch
tail -f logs/pipeline.log
```

后台等待成功→prepare→smoke→ARC→HotpotQA→绘图汇总。保留少量样例attention摘要JSON，不保存KV、原始Query、权重或全文T×T矩阵。失败停止，不自动上传，不持续监控；此前Transformer与融合训练仍保持暂停。
