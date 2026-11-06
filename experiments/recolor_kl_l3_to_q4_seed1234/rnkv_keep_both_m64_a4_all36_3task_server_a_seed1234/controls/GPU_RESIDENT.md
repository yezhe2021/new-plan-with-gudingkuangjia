# 2026-11-05 GPU常驻执行切换

用户在A的GPU恢复空闲后要求关闭CPU卸载。原训练在完整第24步保存后暂停。完整模型、AdamW、GradScaler与CPU/CUDA RNG断点保留，恢复从第25步开始。

通过controls/gpu_resident_entry.py覆盖执行路径：关闭save_on_cpu，Native教师KV留GPU，当前样本的冻结Recolor输出一次搬到GPU并在GPU组装C缓存。继续使用已完成的磁盘KV缓存；不加载不再需要的Sender。Gradient Checkpointing保留。原科学代码与config.json没有重写，实际执行变更由use_gpu_resident.json和probe记录明确覆盖。

同一第24步断点、同一下一批次的对照：loss/task CE/full KL完全相同；一次更新后的参数最大绝对差1.3574e-7、相对L2差5.8905e-7，属于小量浮点差异。样例批次耗时15.85秒→10.16秒，只代表该批次，不承诺全程同倍率提速。按最长正文、正文+Gold与正文×答案覆盖选出的极值样本均通过GPU反向；分配峰值11.76GiB，预留峰值12.47GiB。

这是执行方式变更，M64、四个Rank16 Adapter、训练数据、损失、学习率、步数和已接受的24次更新均保留。恢复队列会校验补丁hash并按实测显存门槛启动；显存不足时不会静默重新启用CPU卸载。
