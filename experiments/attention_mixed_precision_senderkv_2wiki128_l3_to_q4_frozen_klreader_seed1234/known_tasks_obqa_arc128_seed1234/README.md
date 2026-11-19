# Fixed OpenBookQA / ARC quantization follow-up

停止使用2Wiki作为后续量化判断任务；本补充实验使用旧KL-only Reader已评测过的固定 held-out OpenBookQA128、ARC-Challenge128。历史未量化迁移正确数分别92/128、86/128。沿用原问题、候选标签、chat模板、回答cue、Full-Sync对齐、Stage-A、Stage-B、Reader最终权重，不训练、不换checkpoint、不根据量化结果选样本。

1. 首先重放全部256题的未量化迁移基线，保存与历史候选分数/预测的差异；任一任务正确率低于50%则停止，不继续量化。第一题额外核对单次前缀forward与原候选逐项评分实现一致。
2. 单token候选标签通过审计后，一次因果forward同时评分所有选项；不自由生成答案。主指标为选项归一化标准答案NLL与选项准确率，完整词表答案标签NLL作诊断。
3. 维持FP16、全INT4、高注意力10% INT8、3次随机10% INT8、低注意力10% INT8，共7个条件。排序来自FP16 Llama原回答cue最后query，按query heads/层聚合，对全部传输提示token排序，无gold信息。每题保护相同数量位置，共用于K/V及所有层/KV头。
4. 两条读取路径：Llama原生缓存；冻结A+B→Qwen+旧KL-only Reader。复用父实验的实际INT4/INT8协议，逐题核对两条路径数据包hash相同以及混合组实际字节一致。仅缓存min/range和4/8bit码表以减少重复计算，CPU测试要求优化后的数据包与父实现逐字节相同。
5. 新捕获Sender的FP16迁移分数再次与已验证基线比较，差异超过0.1 nats立即停止。原始基线、历史结果和2Wiki实验不覆盖。按任务分别报告成对NLL差值及题级bootstrap区间。

这回答的是已适配任务上的量化策略效用，不证明跨任务泛化。历史测试集已被查看；不作为新的未触碰benchmark。保留原Full-Sync native body prefill，不声称无prefill或实际网络加速。

服务器 B，Python `/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python`：

```bash
python -m unittest -v test_codec
python run_experiment.py prepare
python run_pipeline.py --launch
```

独立进程依次执行baseline、sender、receiver、自动汇总；失败停止，不自动重试。`logs/pipeline.log`查看进度；`baseline/completed.json`记录FP16复现情况；最终`summary.json`及逐题结果在本文件夹。自动追加服务器共享`EXPERIMENT_RESULTS.md`，不自动上传GitHub。

## 2026-10-26 恢复记录

首轮在ARC第23题（arc_22）触发历史FP16分数差异检查，保留已完成的OpenBookQA和ARC前22题。`recovery/audit.json`的同题重放证明：历史Sender缓存来自SDPA，本轮用于注意力测量的缓存来自eager，两种后端分别精确复现对应缓存。剩余样本的第一层K/V均相同，累计浮点差异单独记录。旧候选评分还在前缀后附加未来候选token；优化版只算因果前缀，FP16矩阵形状变化会带来数值差异。在相同前缀上，原后缀评分函数与优化版分数完全一致；映射张量也逐位相同，见`recovery/canonical_receiver_audit.json`。

恢复过程不改原脚本、权重、位宽分配或已完成结果，只计算缺失的接收方题目。所有量化条件仍与同一eager路径自己的FP16基线配对比较；旧SDPA缓存/逐候选评分是数值复现参照，不能要求绝对分数逐位相等。原停止记录和首次额外检查失败记录均保存在`recovery/`，恢复代码和来源hash另行记录。
