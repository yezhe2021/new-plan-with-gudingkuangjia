# HotpotQA distractor512：HeteroFold 与三个 Reader

在标准 HotpotQA distractor dev 的 7405 题中，使用 `random.Random(1234)` 固定抽取 512 题，按原始索引排序。完整保留每题提供的所有段落及顺序，不按长度、答案或模型结果筛题，不选择金标准证据，不截断提示词。

对比原生 Qwen、已复现的官方校准 HeteroFold、固定 Stage A+B 加旧 KL / TaskCE / Hybrid 三个 Reader。全部使用现有固定检查点，不训练或校准。Llama3.2-3B-Instruct → Qwen3-4B；FP16 V100。

统一提示词要求仅输出简短答案，Qwen thinking 关闭，greedy 解码最多 64 token。完整回答去除聊天控制符和两端空白后计算 HotpotQA 答案 EM/F1；保留 yes/no/noanswer 的特殊规则。不评估 supporting facts/joint 指标。逐题保存原始生成、解析答案、得分和截断状态，并输出成对差值置信区间。

HeteroFold 沿用官方 bare-text source capture 与 native-header transfer cache；自研方法沿用 chat source capture、冻结 A+B 和 native body prefill/anchor。保持各自既有实现，不据此声称无 prefill 或速度优势。HeteroFold 曾使用 HotpotQA train 校准；Reader 没有用 HotpotQA 训练，因此不是同域训练预算匹配对比。

后台流水线：prepare → 首题与最长题 GPU 检查（包括官方 HeteroFold cache 一致性、所有 Reader 分支、切换后原生 logits 不变）→ 五条件完整评测 → summary.json / RESULTS.md。失败即停，记录 traceback，不自动重试或上传。

启动：`/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python run_pipeline.py launch`

日志：`tail -f logs/pipeline.log`
