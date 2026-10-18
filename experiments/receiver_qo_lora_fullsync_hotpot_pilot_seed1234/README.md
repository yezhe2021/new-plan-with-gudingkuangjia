# 冻结 Full-Sync writer 的接收方 Q/O LoRA

服务器 A；Qwen3-4B36层 Q/O rank8，共3,833,856可训练参数。Llama/Gemma源模型及原非lm-eval Full-Sync StageA+StageB均冻结，沿用父实验HEAD、原始UTF8 copy/drop、完整gold2段落和位置协议。

官方HotpotQA train中seed1234选128训练+16验证；父实验已固定dev64的前32条仅作最终评估。选择只依赖bridge类型、完整两段gold证据、既定长度/对齐和答案token长度，不依赖模型分数。不是完整benchmark。

4epoch，第一轮使用单writer联合编码，之后使用独立编码的同writer/异writer/原生混合KV。每样本答案及EOS CE + 原生Qwen教师的多步全词表KL；每4样本额外一次原生KV保持项。教师与学生共用基座，教师关闭LoRA，训练启用LoRA。batch1，累积8，AdamW1e-4。最低验证CE+KL选best，不看dev。

源模型先生成CPU KV后卸载，再单独加载writer翻译，最后加载接收方。Gemma源模型保持原float32并按需要CPU权重offload，运算仍在CUDA。CPU内存暂存KV，绝不写入磁盘；恢复时重新生成缓存。训练checkpoint仅保存LoRA/优化器/scaler和进度，不复制基础模型或writer。

最终在相同32条dev上保存LoRA关/开时的原生、单writer联合、独立组合、混合组合、问题单独、错KV和分别遮掉A/B证据的输出。遮掉证据使用attention mask保留原位置；错KV取另一条记录的KV并等距copy到相同长度，属于明确标记的机制对照。自动EM/F1沿用父parser，不代替人工审计。

```bash
cd /home/yezhe/异构模型/receiver_qo_lora_fullsync_hotpot_pilot_seed1234
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u train_receiver.py --smoke
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u train_receiver.py
tail -f logs/pipeline.log
```

Smoke用2训练+1验证+1dev，验证真实两种writer的KV、零LoRA一致性、原生缓存重建、Q/O有限非零梯度和checkpoint读写。正式运行不会加载smoke权重。原生教师prefix全词表logits仅驻CPU内存，不持久化。所有输入/权重身份、选择排除原因和范围记入manifest。
