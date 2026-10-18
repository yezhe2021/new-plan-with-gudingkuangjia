# 翻译 KV 专用接收方读取接口，服务器 B

冻结最新 student-prefix v1 的 Stage A/B、Llama3.2-3B 与 Qwen3-4B。只训练接收方新增的每层/每 query-head rank-8 query 和输出残差，以及依赖当前隐藏状态的 rank-8 门控。原生 Q/O 权重不变。

Full-Sync 原缓存布局保留：位置 0 是原生 anchor；位置 [1, prompt_length-1) 是翻译 KV；最后 prompt token 与生成 token 原生计算。各位置、RoPE、GQA、因果 mask 和官方 prompt 均保留。教师原生前向不启用读取接口。

外部注意力使用原生 query 加低秩修正，外部 scores 增加动态 gate log-prior；整个缓存统一 softmax。该计算等价于分别归一化两组注意力后按分区归一化量融合，但零初始化沿用原 eager matmul/softmax 路径，可逐位恢复旧计算。外部分支读出的向量另加低秩输出修正。外部 query 修正在 Qwen Q-normalization 后、RoPE 前加入；外部输出修正在原生 o_proj 前加入。内部/外部缓存不重排、不重新编码整个问题。

数据、损失、预算固定为前驱：两轮，每轮 128 可信参考 + 128 在线学生前缀 + 86 MCQ 回放；effective batch 4，共 172 更新，AdamW 2e-5。参考轨迹全词表 KL + 0.1 答案 CE + 0.1 真实终止 CE；学生前缀仅 KL，错误学生 token 不当作正确 CE 标签；MCQ 回放沿用原损失。训练轨迹 256，参考验证 24，固定学生前缀验证 32；直接复用前驱固定前缀，不按 benchmark 生成训练数据。

只按原独立验证混合损失选 epoch 0/1/2。零初始化 baseline 保留。当前单次预算验证接收方可训练读取，不是等参数量或等算力的全面架构比较。

启动测试：CPU FP32/GPU FP16 小模型的逐位恒等、增量 cache、因果未来隔离、masked 外部 bank 隔离、各分支梯度；真实最长训练轨迹的零初始化逐位恒等；可信参考、128-token 学生 rollout、最长 MCQ 文本视图三条反向路径、冻结参数无梯度、峰值显存。Smoke 更新丢弃。

正式评测：原协议 OBQA/ARC 各 128，MMLU 128，MMLU-Pro 128；最后者原 5-shot raw completion、2048 token、原停止和 parser。保留完整逐题输出和输入身份验证。随后做独立验证的 full/disabled/gate-only/read-only/错 KV/遮蔽 KV/选项换序诊断；后两种 ablation 只做推理开关，不伪称分别训练。错 KV 替换整个翻译 prompt（含题目），不能单独解释为证据消融。门控质量不能仅由 attention mass 大小判断。

```bash
cd /hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/receiver_gated_reader_v1_seed1234
/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py --startup-only
nohup /hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py > logs/pipeline.log 2>&1 < /dev/null &
tail -f logs/pipeline.log
```

流程有排他锁；训练每 16 更新及首更新保存恢复状态，epoch 结束保存选中结果；MMLU-Pro 逐题落盘和恢复。冻结后校验全部本实验脚本/config、prepared 数据和完整前驱 manifest。最终代码自动生成 summary.json、追加共享 EXPERIMENT_RESULTS.md，失败也保留记录。不上传 GitHub，不复制权重到本地。
