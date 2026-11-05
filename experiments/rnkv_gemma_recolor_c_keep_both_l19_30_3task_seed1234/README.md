# Gemma → Qwen: HeteroFold Recolor+C, Keep-both19–30

按2026-11-04最终确认：只重做19–30层开放Memory，不启动全层组。旧Grouped Stage-A+B实验、其权重与其缓存都不用于本实验。

后台依赖链：旧Gemma第一组评测结束 → Llama三个机制诊断全部结束 → 本目录新建Gemma Recolor → 新建C → 冻结Recolor+C，重新训练Memory → 八任务统一评测。GPU检查在轮到本任务后执行，失败立即停止并保留诊断；排队状态不代表训练已经开始。

## 冻结翻译器的建立

使用与Llama Recolor-C相同的2000条Open-R1+Hotpot校准样本：1600训练/400验证。Recolor仅拟合训练1600条，原始RAW文本、精确token末端对齐、span1/stride4深度模板、两语料有效对齐token质量等权重。Gemma34层、4个KV头×256维映射为Qwen36层、8个KV头×128维。直接复用已固定的官方算法；Gemma采用FP32前向，QwenFP16。

统计量在GPU累加，projection轨迹移到主存减少显存；每50条断点保存，同序恢复。模型卸载后执行Recolor拟合。源K与目标K都取k_proj输出，即pre-k_norm/pre-RoPE。源不使用旧Full-Sync/chat接口。接收端只使用公共header原生KV，不添加原生正文首token锚点。

C为从零初始化的线性Rank64 K/V残差，9,437,184参数，冻结源模型/目标模型/Recolor，4epochs、有效batch8、800次更新、lr3e-4、weight_decay1e-4、clip1。与已有C一致：Native question Query上的Attention KL、key-induced per-head post-W_O MSE、detached-routing value output NMSE。只改变教师张量的存放方式（CPU列表按层回GPU），不截断问题probe、不改变损失。固定epoch4/step800权重。K残差作用于pre-k_norm/pre-RoPE，再执行一次Qwen k_norm与RoPE。

## Keep-both

冻结上述新翻译器。32个新Memory Tokens，加第6/12/18/24层后的Rank16 Memory-only adapters，共409600参数。从头初始化，ARC/Hotpot/GSM各1024训练、各64验证，4epochs、1536次更新、batch8、lr1e-4、完整gold任务CE+0.5 Native→Student全词表KL，混合前缀配额与Llama三任务版本完全相同。

构建时Memory读取完整翻译正文并使用Qwen自身投影生成KV；回答时所有层始终保留翻译正文KV，只有19–30层额外可读32个Memory位置。其他层屏蔽Memory读取；不删除缓存、不改变正文RoPE位置，答案从L+32开始。保留通过冻结Qwen到Memory/Adapter的梯度。溢出拒绝整个优化更新，恢复RNG重试同批，不跳过样本。

复制固定Llama三任务数据文件，不沿用旧Grouped Gemma为了避免其预训练数据污染而做的个别样本替换；新Recolor/C校准题与下游各split无问题重叠。八任务每个64题：ARC、Hotpot、2Wiki、MuSiQue、SciQ、MMLU-Pro、HellaSwag、GSM。QA cap64、GSM cap512，保留两个数字提取指标。Gemma新Recolor+C基线重新计算；Native按每任务真实前向parity通过后复用固定结果。

独立40GiB限额可再生缓存，只缓存同一新Recolor的FP16输出，包含实验与映射hash，反序列化后逐元素检查；未命中正常运行Gemma。不使用旧Grouped潜变量缓存，也不缓存Native测试正文供Student使用。

## 启动与日志

```sh
/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py launch
tail -f logs/pipeline.log
```

`runner_status.json`区分排队/运行/失败；`controls/pause_requested.json`可在安全边界暂停。`launch.json`已存在时禁止重复启动。队列只读Llama机制实验的成功标记，不触碰旧Gemma全层暂停控制。

CPU检查验证真实tiny Qwen的全部36层mask、长答案梯度和checkpoint梯度等价性。GPU依次检查最长校准文本统计、官方RAW缓存逐元素一致、k_norm仅一次、教师分层存储损失等价、C梯度、最长三任务Memory反向和连续批次，再进入正式训练。检查失败不会伪造完成或静默修改训练协议。

仍需披露：本实验Memory只开放19–30层，最新Llama三任务版本开放全层，因此二者并非只切换Sender的严格对照；Gemma FP32也影响耗时。既有测试集已被查看，此轮不是未触碰的确认集。Memory是现有翻译KV与可训练参数的函数，结果改善不等同于恢复已经丢失的信息。
