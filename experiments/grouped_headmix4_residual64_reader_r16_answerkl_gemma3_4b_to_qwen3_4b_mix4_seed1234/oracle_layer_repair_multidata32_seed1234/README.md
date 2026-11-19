# Gemma → Qwen：五数据集层组原生KV替换

用户要求优先执行此实验；之前的翻译＋原生拼接训练已保存断点暂停，不自动恢复。

复用已有 Gemma-3-4B-it → Qwen3-4B Stage-A+B。Stage-A146,800,640参数、Stage-B为rank64跨头残差；全部冻结，不加载Reader，不重新训练。复用既有Gemma源KV，保留原FP32计算/FP16存储精度，不需要加载或运行Gemma大模型。

## 与上一轮对齐

OBQA、ARC-Challenge、HellaSwag、MMLU-Pro、SciQ各32题，严格复用Llama→Qwen替换实验相同的160题ID。问题、选项、Qwen prompt/frame/body/suffix、评分及归一化逐字段相同。GSM8K不进入本实验队列、样本或推理。

Qwen36层分为G1=1–6、G2=7–12、G3=13–18、G4=19–24、G5=25–30、G6=31–36。每组分别原生替换K、V、KV，或屏蔽该组对原传输缓存位置的注意力访问。加全翻译、全原生、全传输位置屏蔽三基线，共27种配置、4320次样本配置评测。

操作在原协议传输的整个body中进行，包含问题和选项；native anchor和新处理的suffix不改。原生KV和翻译KV在缓存组装后、相同位置、RoPE后空间直接替换。屏蔽是注意力mask，不是KV置零。保留其他缓存和suffix，所以不能把全屏蔽理解为完全没有题目信息。

## 正确性和输出

使用原Gemma分组/残差类与固定权重，独立导入，不运行旧版common的全局绑定，避免把Gemma和Llama训练辅助模块混用。检查权重/manifest、Gemma源缓存身份及原行hash。每题都复现历史Gemma A+B和Qwen原生候选预测/分数。GPU smoke另检查完整原生/翻译组装、原生full/cached一致、K/V/层组隔离、anchor不变、被屏蔽内容不影响输出和所有参数冻结。

逐题、逐配置落盘，失败停止，可从已完成配置继续；后台跑完自动汇总每任务准确率、救回/损失、与翻译和同位置屏蔽的paired区间，以及五任务宏平均。summary同时引用相同样本上的Llama宏平均，便于比较；两套Stage-A训练混合不同，不能据此做单变量sender归因。

这属于有原生KV的离线Oracle探索，不是无原文可部署修复。每集32题与多配置搜索仍需独立确认。不自动上传、重训Repair或恢复暂停的拼接训练。

```bash
python run_pipeline.py launch
```

无需保持本机或SSH在线。最后结果为RESULTS.md、summary.json和evaluation/下的逐题结果。
