# Server A · Llama→Qwen · M64 + 4 adapters · all-layer Keep-both

用户2026-11-04只授权先运行M64这一组。64 Memory Tokens，4个Rank16 Adapter，位于第6/12/18/24层后。以已完成M32+A4三任务全层Keep-both为对照，冻结同一份HeteroFold Recolor+C，不重新训练翻译器。可训练参数491520。

训练集逐字节复制原基线：ARC/HotpotQA/GSM8K各1024训练，各64验证；4epochs、有效batch8、1536次更新、lr1e-4、weight_decay1e-4、clip1。完整gold任务CE+.5 Full-vocabulary Native→Student KL；gold/native/student混合前缀配额、最多64 rollout token/8个KL位置完全沿用原方法。全部36层回答时保留正文和Memory；Memory位于L..L+63，suffix从L+64开始。前32个Prompt和四个Adapter的**初始值**与M32完全相同，额外32个Prompt用独立seed7634初始化；不加载任何训练好的Memory权重。

八个固定64题测试集：ARC、HotpotQA、2Wiki、MuSiQue、SciQ、MMLU-Pro、HellaSwag、GSM8K。QA cap64、GSM cap512，两种数字提取结果分别保留。报告相对M32的逐题变化和配对bootstrap区间。只有经过当前主机真实Native/C/M32推理parity后，才能复用原基线结果。

## 显存受限的执行方式

A共享GPU当前仍有外部进程。不终止任何外部进程，也不改训练定义：

1. **Sender-only GPU阶段**：仅加载冻结Llama与原始Recolor，在相同RAW接口、Token/Layer alignment上运行原始ExampleBuilder.recolor_outputs。保存原始FP16 pre-k_norm/pre-RoPE输出到本实验独立缓存，Qwen权重不占GPU。
2. **Qwen-only GPU阶段**：不加载Sender/Recolor权重；从CPU磁盘读取完全相同的输出，逐层施加冻结C残差，再执行一次Qwen k_norm和RoPE。原生正文KV仅作为教师/原生对照，存放CPU，需要教师前向时搬到GPU，Student反向期间释放该GPU副本。
3. 保留Activation Checkpointing，额外使用PyTorch save_on_cpu卸载自动微分保存的Tensor。冻结Qwen仍对Memory/Adapter可微。损失数学形式、样本顺序、梯度累积、输入精度不变。
4. 子进程启动前根据实际free显存设置CUDA allocator上限，并为外部任务留1536MiB。先预存约几十个极值/连续batch样本，再跑最长样本反向、连续四批、全部位置Mask和主机parity。通过后预存剩余全部样本并正式训练。
5. 若限额下OOM，CUDA进程退出释放显存，队列保留缓存/断点并等待更多空闲显存；不缩短样本、不减少Memory、不跳过批次。训练断点回滚的未保存更新日志单独归档；AMP溢出恢复RNG重试同批。

缓存占用在prepared/completed.json记录；A磁盘有足够空间，不删除旧实验。共享GPU和主存搬运会影响速度，不能用独占B的吞吐直接承诺耗时。

## 队列与日志

```sh
cd /hy-tmp/yezhe/异构模型/rnkv_keep_both_m64_a4_all36_3task_server_a_seed1234
/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py launch
tail -f logs/pipeline.log
```

检查runner_status.json区分预计算、训练、等待显存或失败。启动后独立后台运行，本地电脑可关闭。其他三组没有排队。本轮只做容量对照，一颗随机种子与既有已查看测试集只提供探索性证据。
