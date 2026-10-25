# Cross-model attention and KV-access ablation pilot

服务器 A；Llama-3.2-3B-Instruct、Gemma-3-4B-it、Qwen3-4B；不训练任何模型或适配器。

## 可检验的问题

1. 同证据、同问题、同强制回答前缀下，注意力是否在共同文本位置上相似？
2. 一个模型的注意力排序，能否预测另一个模型屏蔽对应 KV 位置时的任务损失？

注意力分布相似不等于功能相同。匹配 Head 也不能单凭分布相似就解释为语义等价。

## 固定数据与输入

- 使用服务器已有 2WikiMultihopQA dev，seed1234 随机排序，固定接收最先通过审计的24题。
- 8题只做 Head 指派；16道不重叠题报告注意力与消融结果。不使用评测题选择 Head、层或参数。
- 每题的全部标注 supporting sentences 加8个固定干扰句，随机排列；保留原标题。
  这使用了支持句标注构造受控证据集，**不是官方完整上下文 benchmark 分数**。
- 同一证据与问题逐字符复用于三个模型，各用原生 chat template；Qwen thinking=False。
  这是控制语义任务输入，而不是声称三种原生模板或特殊 token 相同。
- 三者均强制 `The answer is:`；前缀不包含任何标准答案内容。
- 最长768 prompt tokens，三个 tokenizer 均满足才接受；包含短答案后仍短于 Gemma 的1024-token局部窗。
- 使用 fast-tokenizer Unicode 字符偏移，将 token 概率按字符交叠比例投到相同的空白分词文本单元。
  跨证据边界 token 仅分配落在证据中的概率；不比较不可对应的 special tokens。
- 保存证据总注意力质量；比较的是条件于证据的归一化分布，不能忽略很低的证据注意力。
- 投影不会消除分词方式带来的全部偏差。消融的最小单位仍是完整的模型 KV token；
  极少量跨证据边界 token 按与证据有交叠纳入候选，屏蔽时其边界外字符也一并不可直接访问。
  `prepared/questions.jsonl` 的原始 offset 和每条干预的 blocked_positions 可用于复核这点。

## 注意力测量

- 用官方 eager attention 返回的真实 softmax 概率。只保留强制回答前缀 query 行，避免长期保存整个 N×N attention。
- 两种 query：前缀最后一个 token；按前缀字符覆盖量加权的 prefix mean。
- 全模型聚合、相对深度三分层聚合，以及匹配后的单 Head 分别报告。
- 相对深度配层，不假设层编号或 Head 编号直接对应。
  每层对用8道校准题平均 JSD 的矩形 Hungarian 一对一指派；多余 Head 不匹配并报告覆盖率。
- 比较的是 query attention heads：Llama每层24、Gemma每层8、Qwen每层32；不是共享KV的头数。
  GQA、RoPE、Gemma局部/全局层等计算均保持原模型实现。
- 每个 Head 均保留原始指标：Top-K overlap（5/10/20%共同单元）、Spearman、base-2 JSD。
  常数排名的 Spearman 记 null；Top-K 使用独立有种子的破同分，另报同分条件下的期望 overlap 与随机基线。
- 置信区间以题为单位 bootstrap，不能把同题不同 Head 当成独立样本扩大 n。

## KV干预

- 先缓存原生题目 body，再从强制答案前缀开始屏蔽证据 key 列；所有层/Head 同样屏蔽。
  同一 token 的 K/V 访问因此不可见，但物理缓存、RoPE 位置与 token 顺序不变。
- 不会重算已缓存 body，其他 body token 可能仍间接包含被屏蔽信息；这是直接 KV 访问干预，**不是彻底删除文本信息**。
- 每个接收模型按其证据 KV 数量屏蔽10/20%，所有排序与随机对照使用完全相同的接收方 KV 数量。
- 排序来自自身全模型、自身三层段、其他模型的全模型、校准选定的晚层匹配 Head，以及自身对应 Head。
  Head 条件是“按该 Head 的排序选位置”，不是“删除该 Head”。
- 对照：自身低注意力位置、3次有种子随机位置、全部证据 KV 访问屏蔽。
- 主指标：完整标准短答案的条件 NLL 变化（每token和每字符）；诊断 greedy EM 与24-token截断率。
  全模型排序和对照做自由生成；各层/Head排序只算 NLL，以控制首轮预算。
- 分别报告全体题和原始答对题的损失，避免 baseline 低正确率掩盖功能下降。

## 精度、实现检查和边界

V100不支持原生BF16。本轮 Gemma 保持FP32，Llama/Qwen使用FP16；这不是精度匹配消融。
GPU smoke检查：无屏蔽 replay 一致、cached vs full prompt top1一致及logit误差、被屏蔽位置注意力精确为0、
原body cache逐tensor未改变、全参数冻结且无梯度、投影覆盖与概率有限。

这是一轮16题评测的可行性 pilot，不足以声称跨任务普遍规律、KV向量可直接互换或共享KV后性能不变。

## 执行

```bash
cd /home/yezhe/异构模型/cross_model_attention_kv_ablation_2wiki_seed1234
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -m unittest -v test_metrics
/home/yezhe/data/miniconda3/envs/attnkv/bin/python experiment.py prepare
/home/yezhe/data/miniconda3/envs/attnkv/bin/python run_pipeline.py
```

模型顺序加载，不与其他任务争抢GPU；失败停止等待检查，不自动重试。准备之后 manifest 固定代码、数据与模型身份。
查看日志：`tail -n 40 logs/pipeline.log`；状态：`cat status.json`。
最终 `summary.json` 包括功能损失、成对随机对照和注意力指标；`matching.json`记录校准指派；
`comparisons/per_head.jsonl`、`probes/`、`ablations/`保存可复核记录。
最终脚本自动追加服务器同级 `EXPERIMENT_RESULTS.md`；不自动上传GitHub。

方法动机：[Attention is not Explanation](https://aclanthology.org/N19-1357/)、
[Attention is not not Explanation](https://aclanthology.org/D19-1002/)。两篇早期研究的对象与当前LLM不同，
本实验依赖实际干预结果判断功能重要性，不将文献结论直接套用到当前模型。
