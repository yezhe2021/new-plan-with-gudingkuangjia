# Frozen grouped Stage A + answer-position Reader

用户于 2026-10-23 确认实现并启动。只新增一个训练分支。

## 方法

- 使用 `grouped_headmix4_l3_to_q4_seed1234` 的 Stage-A `best.pt`，SHA256 固定在 config。
- 四组源层到目标层、组内跨头映射保持冻结。**不加载任何 Stage-B 残差，不加载旧 Reader。**
- Qwen3-4B 原始参数冻结；36 层 q_proj/o_proj 的 rank16、alpha16 LoRA 从零输出初始化，7,667,712 个可训练参数。
- Reader 只在迁移 KV 的 suffix/continuation 上启用；原生题目预填充和教师始终关闭 Reader。
- 全词表 forward KL(Native || Student)，温度 1，FP32 loss，按样本等权；不使用答案 CE 或奖励。
- OpenBookQA、ARC、MMLU-Pro：在固定回答 suffix 后预测选项字母的位置计算一次 KL，不生成自由回答、不监督 EOS/格式。prepare 用实际 tokenizer 核验所有训练/验证标签是互不相同的单 token；遇到多 token 拒绝运行，避免静默截断标签。
- GSM8K：沿用最多64 token、最多8个位置的同前缀蒸馏。epoch1 原生前缀，epoch2–4 每批两个数学样本中一个原生、一个当前学生前缀；教师和学生的输入 token 历史完全一致。

## 数据与预算

复用 UFD 的已固定数据及对齐 Sender 缓存。四任务各512训练题、64验证题。
每批每任务2题，有效 batch8；4 epochs、1024优化步、AdamW lr1e-4、weight decay0、clip30。
每64步保存；每epoch在固定16题/任务上监控验证KL和选择题准确率。主结果固定epoch4，不以测试集选模型。
已有原生前缀只在数学题训练中复用。

## 评估与边界

训练后自动评估六任务：OBQA128、ARC128、HellaSwag64、GSM8K32、MMLU-Pro128、SciQ64。
新增计算 Stage-A-only、Stage-A+Reader 两种条件，复用并核对逐题的 Native、旧UFD、新版全量StageB、新版残差StageB结果。
GSM8K greedy generation cap4096。HellaSwag/SciQ沿用父实验候选答案评分接口。

本轮相对残差实验同时更改训练位置和选择题监督位置，不能把收益严格归因于只更换训练位置。
MMLU-Pro来自官方test的既有自定义划分，不能宣称完整官方test从未参与训练。
历史测试子集已被查看，不据其选checkpoint。短训练前缀不代表充分训练长推理。
当前准确率接口计算原生body用于教师/anchor，不报告prefill-free速度收益。

## 执行

服务器 Python：`/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python`。

1. `python -m unittest -v test_reader.py`
2. `python experiment.py prepare`
3. `python run_pipeline.py --launch-if-ready`

Launcher 使用独立 session，无终端输入依赖；GPU smoke 通过后自动训练，训练后自动评估汇总。
若资源暂时不足，可独立运行 `queue/server_queue.py` 等待启动；不持续轮询训练结果。
失败会留下 `failure.json` 并停止，避免盲目自动重试。服务器重启不自动恢复。

GPU smoke 验证原生cache重建、零初始化等价、传输KV不被改写、72个LoRA分支梯度、教师输出不变和translator权重不变。
每epoch再次校验translator权重哈希，优化器及checkpoint只含Reader可训练参数（恢复checkpoint另含优化器状态）。

关键产物：`status.json`、`logs/pipeline.log`、`smoke/completed.json`、`runs/reader/`、`evaluation/per_sample.jsonl`、`summary.json`。
