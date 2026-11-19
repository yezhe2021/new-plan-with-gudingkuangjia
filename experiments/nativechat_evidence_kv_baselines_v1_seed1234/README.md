# Native-chat evidence-only KV translation baselines V1

服务器 B。Llama3.2-3B、Gemma3-4B 分别使用自己的原生 chat template 编码 Evidence；Qwen3-4B 冻结，只读取翻译后的证据 KV、原生公共前缀和问题后缀。Sender 不接收实际问题；Receiver 的翻译条件不 prefill 证据原文。Llama 模板默认日期固定为 18 Oct 2026。所有原生基座使用 FP32/eager；V100 不使用 BF16。这是新 Evidence-only/native-template 协议适配，不是论文原模型对及数值表的原样复现。

## 固定数据与协议

官方 HotpotQA train 第一 parquet shard 中按 seed1234 选择 512 train、64 validation；官方 distractor dev 中选择 128 pilot。只选 Bridge 和两份完整 gold supporting paragraphs，保留标题及全部句子，不提供 supporting sentence 标签或答案标记。三个 split 的问题 ID 和文档标题相互隔离；pilot 排除旧64题。任何样本筛选只依赖结构、长度、正文边界及三模型 token 接口，不依赖模型回答。它是 gold-evidence 机制 pilot，不是完整 HotpotQA 检索 benchmark。

固定语义内容：通用短答案指令、Evidence 标题、文档正文、固定 `End of evidence.` 标记；Qwen 后面另接 Question/Answer 和自己的 assistant 生成前缀。Sender 也使用固定正文结束标记以避免 native template 的 strip 改写正文尾部。三种模板正文字符边界必须是 token 边界，纯 special token 不参与正文映射。接收端两段独立正文的 token IDs 必须恰好拼成联合正文 token IDs，公共前缀和问题后缀必须一致。正文超长或边界不满足要求就排除，不截断。

所有传输缓存最终使用 Qwen token 网格、post-k_norm/pre-RoPE K 和 V；统一重施 Qwen RoPE。Our Writer 映射 post-k_norm/pre-RoPE K，不二次归一化；HeteroFold 映射 raw/pre-k_norm K，在进入缓存前使用 Qwen k_norm 一次。独立块均在同一公共前缀下独立编码，拼接仅保留一份公共前缀，键在最终全局位置重施 RoPE。Native Independent 不要求等同于 Native Joint。

## 两条翻译路线

Our Writer 复制原始 Llama Full28 MLP / Gemma Full34 HeadMix256 结构及 Residual64 结构，从新初始化开始，不加载旧 MCQ/lm-eval checkpoint。Full-Sync 在共同 UTF-8/字符结束边界单元内 copy/drop，每个 Qwen 正文 token 都映射，不选 slot、不压缩。

Stage A 在每个训练问题的 E1、E2、联合 E1+E2 三种原生编码单元上拟合原生 Qwen KV，NMSE+cosine，4 epoch、AdamW1e-3、有效 batch8、clip30。Stage B 冻结 Stage A，在联合完整证据条件下做原生 Qwen 贪心轨迹的全词表 KL，2 epoch、AdamW1e-4、有效 batch4、clip30。教师最长128 token，保留自然终止，截断不强加 EOS；不使用 gold 答案筛选轨迹或答案/EOS CE。它明确是 choice-KL Writer 的 QA 蒸馏适配，不声称原始 choice loss 未变。

HeteroFold 复用 vendor/heterofold 的官方核心实现：Character-END 前向填充、相对深度三层 taps -4/0/+4、Recolor 闭式拟合及 K/V 分开的 rank16 Output-Aware Calibration。新协议下源特征只来自 Evidence，校准 query 来自 Qwen 问题位置。保持 attention KL、每 query-head key-induced post-WO MSE、value post-WO NMSE（routing detach）。Recolor 用 FP64 统计和闭式求解、最终 affine FP32，使用原官方 jitter/floor；校准4 epoch、3e-4、weight decay1e-4、有效 batch8、clip1，最后折叠进 affine map。原生 query/非证据 KV 是训练侧教师探针，推理绝不注入这些带原生证据条件的问题状态。

两路线拥有同样的512训练问题及E1/E2/Joint校准单元，但监督位置、损失和计算量不同，不能只按分数将差异归因于网络结构。所有 checkpoint 按独立验证损失选择，epoch0 也参与。

## 执行与验收

先核对 Native KV 捕获/重建、多步 logits/greedy 生成和原生 generate 的一致性，再检查两模型原生 capture/正文对齐。每个方向用16个训练问题进行32次小样本拟合更新、Writer B 和 HeteroFold K/V 的真实反向传播、冻结权重无梯度与 folding 一致性；smoke 权重丢弃。小样本损失下降只是实现检查，不视为充分拟合或泛化证明。

原生验证须相对 Question Only 的 F1 平均增益至少0.05，且2000次按问题配对 bootstrap 的95% CI下界大于0，否则停止 Translator 正式训练并报告原生 pilot 结果。四个单源 Translator 也分别按同一验证增益门槛验收，并要求正确证据平均F1高于错证据；不按pilot选超参或检查点。错证据来自当前相近长度小批中另一问题，保留完整KV、不重采样；长度并非完全匹配，因此此项只是辅助敏感性诊断。每个条件从注入前重新计算问题后缀，不保留正确证据下的问题KV。

单源阶段由一个 Sender 联合编码两份完整证据。四个 Translator 全部通过验证后，才评估 Native Joint、Native Independent，以及每种方法的 L+L、G+G、L+G、G+L，共10个多源条件（Native Joint复用）。L+G/G+L只交换来源分配，不交换E1/E2的最终顺序。另保留Question Only和四个单源联合基线，完整pilot最多15条件。

输出短答案全文直接按 HotpotQA 官方归一化计算 EM/F1；不取首行或重新选择答案片段。Qwen关闭thinking，greedy，统一128 token上限；报告原始回答、EOS/截断和长度。完整逐题输出、验证门槛、权重身份、日志及summary保存在本目录，代码自动追加共享 EXPERIMENT_RESULTS.md。

## 存储与硬件

当前实验盘和系统盘各只剩约5 GiB。为保留全部旧实验，临时优化器检查点、Recolor sufficient statistics放在 `/dev/shm/nativechat_evidence_kv_baselines_v1_seed1234`，预算60 GiB，按阶段轮换同一个恢复文件。它们是易失的；服务器重启后未完成阶段可能需要重算。最终两个 Stage-A 权重持久保存到 `/root/nativechat_evidence_kv_baselines_v1_seed1234_selected`，折叠HeteroFold及Stage-B权重持久保存到本目录 models/，完成记录保存路径和SHA256。

只允许一个冻结 backbone 在GPU上；阶段之间卸载并重载权重，模型I/O成本明确计入，不宣称推理加速。CPU8线程，原始KV只在有界4 GiB LRU和当前小批内存中，不将KV写入磁盘。数据、配置、源代码和模型原始权重身份在正式训练前冻结。失败保留日志，不抢占或停止其他GPU任务。

```bash
cd /hy-tmp/yezhe/异构模型/nativechat_evidence_kv_baselines_v1_seed1234
nohup /hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py > logs/pipeline.log 2>&1 < /dev/null &
tail -f logs/pipeline.log
```

不自动上传GitHub，不训练Receiver，不添加读取器、动态路由或新候选结构。
