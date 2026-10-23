# Frozen grouped Stage A + residual Stage B + Reader

用户于 2026-10-23 授权暂停 A+Reader 评测并启动本实验。

## 冻结前端与训练范围

- Stage A：`grouped_headmix4_l3_to_q4_seed1234/runs/stage_a/best.pt`，134,217,728 参数。
- Stage B：`grouped_headmix4_residual64_l3_to_q4_seed1234/runs/stage_b/last.pt`，rank64 跨头残差，9,437,184 参数。
- 两份 checkpoint 的 SHA256 均固定在 config；严格加载 Stage A 和残差 state，再将整个前端设为 eval / requires_grad=False。
- Qwen3-4B 原始参数冻结。新建36层 Q/O LoRA，rank16、alpha16、7,667,712 可训练参数；不加载旧 Reader。
- 原生 body/教师关闭 Reader。迁移 KV 的 suffix/continuation 开启 Reader。
- GPU smoke 和每轮结束验证整个 A+B 的参数哈希不变，训练过程中检查冻结参数无梯度。checkpoint 只保存 Reader 和恢复所需的优化器状态，并固定两份前端权重身份。

## 训练协议

沿用最近 A+Reader 的数据、题目顺序、目标与预算，仅把冻结前端改为 A+B。

- OpenBookQA、ARC-Challenge、GSM8K、MMLU-Pro 各512训练题，共2048题。
- 各64验证题；每轮固定取16题/任务监控，不使用测试结果选模型。
- 选择题：只在固定回答前缀后预测答案标签的位置做完整词表 KL；不生成回答，不用 gold CE。用真实 tokenizer 审计训练/验证标签均为单 token。
- 数学题：最多64 token 的原生/学生前缀，最多8个监督位置。第1轮原生，第2–4轮每批两道数学题中一道用当前学生前缀。教师与学生使用相同 token 历史。
- KL(Native || Student)，温度1，FP32 loss；按样本等权。
- 4轮、有效batch8（每任务2题）、1024步、AdamW lr1e-4、weight_decay0、clip30、seed1234。
- 每64步保存；主checkpoint固定epoch4/step1024。验证最佳checkpoint仅作诊断。

## 自动评估

训练后评估六任务：OBQA128、ARC128、HellaSwag64、GSM8K32、MMLU-Pro128、SciQ64；GSM8K greedy cap4096。
只新增计算 A+B+Reader，复用已完成残差实验的 A+B、Native、旧UFD、全量StageB逐题结果。
旧 A+Reader 评测保持暂停。prepare 固定其已有逐题文件快照，只对共同样本比较，并显式报告样本数；不将部分 GSM8K 结果当成完整32题结果。

本实验增加了 Reader 训练预算，用于检查在 A+B 上追加 Reader 的收益，不是等预算对照。
MMLU-Pro 沿用从官方test中划分的自定义数据集；历史测试已被查看。准确率接口不支持宣称零prefill延迟。

## 启动与产物

服务器Python：`/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python`。

1. `python -m unittest -v test_reader.py test_queue.py`
2. `python experiment.py prepare`
3. `python run_pipeline.py --launch-if-ready`

服务器独立session顺序执行 GPU smoke、训练、评估和汇总。脱离SSH运行，本地电脑可关机；服务器重启不自动恢复。失败留下failure.json并停止，不盲目重试。
产物包括 status.json、logs/pipeline.log、smoke/completed.json、runs/reader/、evaluation/per_sample.jsonl、summary.json。
