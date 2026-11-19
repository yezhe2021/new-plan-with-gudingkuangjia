# Does the frozen Receiver depend on transmitted KV content?

服务器 B；原固定OpenBookQA128、ARC-Challenge128。冻结同一Llama→grouped Stage-A→rank64 Stage-B→Qwen+旧KL-only Reader，不训练、不量化、不选择新checkpoint。测试KV内容依赖和选项标签绑定，不把“自身知识”与“读取题目信息”当成互斥来源。

## 输入与对照

Qwen只直接prefill所有题共有的native anchor（模板开头和第一个公共指令token），不prefill问题或选项；所有题及选项文本必须经被干预的KV进入。回答suffix也逐token相同。各题内KV长度、RoPE位置、层/头形状固定。原始模型预填充其他题目的缓存不会混入接收方。题目长度仍通过位置/掩码可见，屏蔽组测到的包括这种长度信息和模型/Reader的标签先验。

原始题共11个条件，按5类汇总：

- true：本题真实FP16翻译KV。
- masked：屏蔽全部传输key列，保留位置和相同anchor/suffix；不是把K/V数值置零。
- wrong ×3：同任务的其他题真实翻译KV，先按选项数和长度匹配，再用无自配对指派；选择不读取gold。K/V使用同一保序最近邻索引变到目标长度，在receiver RoPE之前替换。报告等长子集以及相同选项数子集。ARC唯一三选项题无同类donor，回退到同任务近长度四选项donor，并显式标注，不隐去。
- random_target_stats ×3：独立高斯K/V，匹配本题KV每层/头/channel沿token维度的均值、方差。它仍保留本题聚合统计，不能独自证明只靠模型先验。
- random_donor_stats ×3：匹配错题donor的统计量，去掉上一个条件保留的本题统计信息。每题每次重复与上一条件使用相同标准高斯样本，减少随机实现差异。

每题还做3个选项重排条件：全部选项内容按固定随机无不动点排列，gold标签随内容重映射；用新顺序重新捕获Llama KV并经过同一冻结前端，比较reordered_true、reordered_masked、reordered_old_kv（继续喂旧顺序KV）。报告正确率、原先答对题的保留率、预测是否跟随同一选项文本的新标签，以及旧KV是否保留旧标签倾向。所有随机安排在GPU评测前固定，不根据模型结果选择。

## 评分与核验

沿用原单token候选标签。只评分选项，不自由生成。一次forward在末尾附加固定dummy候选，在它之前的预测位置读取全部选项分数；GPU smoke核对与原逐候选评分一致，并确认换dummy不会影响前一位置logits。模型用原SDPA后端，避免与历史缓存的eager/SDPA差异混淆。

CPU测试覆盖字母/数字选项内容与标签重映射、donor非自配对与单例回退、K/V配对保序resize、噪声统计与可复现性、传输区屏蔽边界。GPU smoke验证masked true与masked noise的完整logits完全相同、dummy无前向泄漏、anchor长度、冻结无梯度。之后先评测全部256题的真实KV基线，两项任务正确率均达到50%才继续正式对照；基线分数原样复用，避免重复forward。原始接口和权重身份均固定在manifest。

指标：答对题数、候选归一化NLL、正确选项概率、相对true的预测改变比例；随机重复先题内平均，区间按题bootstrap。报告均匀随机猜测和多数标签参照，不能一律假设所有题都是四选一。

## 执行

```bash
python -m unittest -v test_controls
python run_experiment.py prepare
python run_pipeline.py --launch
```

独立进程依次执行重排题Sender捕获、冻结KV映射、GPU smoke、正式评测、自动汇总。失败停止，不自动重试。脚本仅清理自己新建的可再生临时张量，不动历史缓存/权重；正式轻量结果保存后自动追加服务器共享EXPERIMENT_RESULTS.md，不自动上传GitHub。

日志`logs/pipeline.log`；逐题`evaluation/<task>/<id>.json`；最终`summary.json`。

完成后可运行`python analyze_donor_labels.py`，补充统计候选标签词表完全一致的错题donor子集，保存到`diagnostics/donor_label_subset.json`。ARC历史题含1道数字标签题，除去标签词表不一致的donor配对可避免把标签格式差异误算为内容差异；子集条件不依赖答案或模型输出。另外比较错题KV的预测更接近donor还是真正接收题的原始预测。
