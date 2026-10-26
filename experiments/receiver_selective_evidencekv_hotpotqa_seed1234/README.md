# HotpotQA 接收端证据 KV 预实验

本实验用 HotpotQA bridge 问题的两段完整正确证据，判断冻结的异构 KV 路径是否能提供有效增益。只运行 32 题预实验；提前固定的独立 128 题保存在 `prepared/reserved_test.jsonl`，不自动评估。它不是完整的官方 distractor 基准。

Llama3.2-3B 只编码证据，已有 grouped Stage-A 和 rank64 residual Stage-B 转换 KV。Qwen3-4B 原生读取问题，使用关闭 thinking 的聊天模板生成短答案。所有参数冻结，不安装或加载 Q/O Reader。脚本复用 `config.json` 中 reference 的已固定模型、token 对齐和 KV 转换代码，具体检查点及哈希写入 manifest。

## 条件

- `question_only_compact`：普通无证据问题提示，不保留证据位置。
- `reject_all`：屏蔽全部传入证据 KV，保留共同 anchor 和位置；从问题第一个 token 开始屏蔽。
- `native_evidence`：Qwen 原生编码正确证据，作为正对照。
- `clean`：接收正确的翻译 KV。
- `wrong`：接收另一题的 KV，按长度匹配 donor 并重采样到相同位置数；donor 不共享证据文档标题。
- `partial_wrong`：同位置替换 50% KV，K/V 使用相同替换索引。

除 compact 外，所有条件的证据位置数、问题和答案位置相同。保留 compact 控制来区分位置变化和证据内容的影响。错误证据是合成干扰，不保证 donor 在语义上绝对无关。

## 数据与指标

从缓存的 HotpotQA dev 固定种子抽取。只按任务类型、完整段落、token 长度和已使用样本排除规则选题，不依据模型预测或是否答对选题。排除旧 HotpotQA 实验中已准备的样本与旧 pilot ID，以及当前参考检查点已知训练/验证问题；32 与 128 题的文档标题互不重叠。

报告生成答案的标准归一化整段 EM/F1、参考答案逐 token 平均 NLL、EOS 率和耗时。保存原始输出，不根据答案抽取片段。参考答案只用于单独的 teacher forcing 和离线评分，不能进入 Sender、生成或无标签特征。问题 log probability、首 token 熵和生成平均 log probability 只记录，不拟合选择器。

预实验通过标准提前写入 config：原生证据相对 compact 问题 F1 增益至少 0.10，正确翻译 KV 相对 reject-all 的 F1 增益至少 0.05 且按问题配对 bootstrap 95% 区间下界大于 0，正确 KV F1 高于错误 KV。这是继续实验的探索性标准，不是正式显著性证明。

## 执行

```bash
python run_experiment.py prepare
python run_pipeline.py --launch
tail -n 50 -f logs/pipeline.log
```

流水线：capture → translate → preflight → evaluate → summarize。Preflight 检查屏蔽后随机 KV 内容不能改变输出，以及原生 KV 续算与完整 prefill 一致。结果逐条件落盘；结束后自动生成 summary/RESULTS，并追加共享 `EXPERIMENT_RESULTS.md`，删除本实验临时 KV。不会改变旧实验或自动上传 GitHub。
