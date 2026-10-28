# 多数据集翻译器 Oracle 层组替换实验

冻结既有 Llama3.2-3B → Qwen3-4B 分组跨头 Stage-A 和 rank64 残差 Stage-B；不使用 Reader，不训练任何模块。

六个原有数据集 OBQA、ARC-Challenge、HellaSwag、MMLU-Pro、SciQ、GSM8K，各按 seed1234 随机抽取32道历史评测题。保留原 Prompt、token 网格、接收端 suffix、候选答案、归一化和评分。MMLU-Pro 是原有 custom split。GSM8K 保持4096 token自由生成上限；其余五项沿用原候选似然评分，不改成标签预测。

## 干预

36层分六组，每组6层。对原协议实际传输的全部 body 位置（native_prefix_length 到 body 结束，包括问题和选项）进行 K-only、V-only、KV 原生替换，或屏蔽该组对这些位置的直接注意力访问。缓存先由原接口完成组装，之后在相同位置、RoPE 后空间中替换。native anchor 和新处理的 suffix 不替换、不屏蔽。

这与 HotpotQA 的仅证据位置干预不同。屏蔽不是删除全部题目信息；原缓存生成的其他状态和新 suffix 仍可能间接携带信息。

加全翻译、全原生、全部传输位置屏蔽三条基线，共27配置×192题＝5184次样本配置评测。每题只捕获、翻译一次源 KV，各配置使用独立克隆的缓存，无配置间污染。

## 检查与结果

启动时六个任务分别检查：全翻译/全原生组装一致、MCQ历史评分复现、原生 full/cached 前向一致、K/V和层组干预隔离、anchor不变、屏蔽后被屏蔽缓存内容不影响输出，以及所有参数冻结。GSM8K 启动时只以16 token测试手写 mask-aware 生成与原生成器一致，并检查历史答案前缀；正式评测始终4096 token并逐题验证完整历史基线，不以16 token作为结果。

逐题、逐配置落盘，失败停止，支持从已完成配置续跑；不自动改变协议。输出每个任务的准确率、paired bootstrap区间、救回/损失题目、屏蔽对照、GSM8K截断率及宏平均。所有结果属于探索，不在同批题挑最优配置后当成确认结果。未运行留出确认、训练新Repair或上传GitHub。

## 运行

```bash
python run_pipeline.py launch
```

后台流程：prepare → smoke → 五个选择/候选任务 → GSM8K → summarize。先完成较快的数据集，GSM8K 中超长或重复输出仍遵守原4096上限。无需保持本机或SSH在线；服务器重启不会自动恢复。
