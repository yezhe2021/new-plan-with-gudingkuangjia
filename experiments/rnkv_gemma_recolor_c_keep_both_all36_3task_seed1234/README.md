# Gemma→Qwen · M32+A4 · 全36层 Keep-both · 三任务

用户2026-11-05确认在服务器B补跑全层版本，对照刚完成的Gemma Recolor+C19–30层实验。

唯一训练设置变化：回答时可读取Memory的层从19–30改为1–36。全部层一直保留翻译正文KV。Memory构建过程相同，从相同seed1234初始化的32个Memory Tokens和四个Rank16 Adapter重新训练，409600参数，Adapter位于第6/12/18/24层后。不会加载旧Memory训练权重继续微调。

直接冻结复用同一Gemma Recolor映射6fb66bc1…和C epoch4/step800权重99061a28…。不重建Recolor、不训练C；RAW接口、token/layer alignment、Qwen k_norm/RoPE约定保持相同。Gemma FP32、Qwen FP16。已有父实验40GiB缓存只读复用；未命中运行相同compute_base，不额外落盘，保留既有文件。A的Llama M64和旧Grouped Gemma的全层暂停任务均不受影响。

逐字节复制三任务数据：ARC/HotpotQA/GSM8K各1024训练、各64验证；4轮、1536步、有效batch8、AdamW lr1e-4/weight_decay1e-4/clip1，完整Gold CE+.5 Full-vocabulary KL；同一batch和混合前缀配额。验证只用于诊断，固定最终epoch4权重，不按测试结果选checkpoint。

同一Qwen、C、部分层训练权重在每任务首例上须复现旧结果；同权重下全层/部分层的Memory构建KV须完全相同。Tiny Qwen验证全36层mask、与旧部分层初始化完全相同、长Gold梯度和checkpoint梯度等价；真实GPU验证最长样本反向与连续批次后丢弃检查权重，重新从零开始正式训练。

测试原八任务各64题，直接比较C、19–30层单独训练模型、新全层单独训练模型、原生Qwen。QA cap64、GSM512，两种数字提取分别报告。保留原始题目与标注，争议样本不做结果导向排除。单seed与已查看测试集仅提供探索性证据。

后台队列：CPU准备/检查 → 等待GPU可用 → 旧结果parity → GPU smoke → Memory训练 → 八任务评测。NVML/CUDA故障时等待恢复，训练按已接受更新保存与恢复，不跳过样本。其他错误停止并写failure.json；不静默改结构或精度。

```sh
cd /hy-tmp/yezhe/异构模型/rnkv_gemma_recolor_c_keep_both_all36_3task_seed1234
/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py launch
tail -f logs/pipeline.log
```

根目录runner_status.json区分运行、等待GPU、暂停、失败。启动后不需要本机在线，不自动上传GitHub或长期人工盯进度。
