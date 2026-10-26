# HotpotQA 独立 128 题接收端选择实验

父目录 32 题通过预先设定的继续标准后启动。采用预先留出的原样 128 题，数据与父预实验按问题 ID 和证据文档标题隔离。Stage-A、Stage-B、Qwen 冻结，无 Reader。

生成和 KV 路径与父实验一致，使用全部接收、全部拒绝、原生证据、普通无证据问题控制，测试 clean / wrong / partial_wrong 三种等权场景。

选择规则仅在父实验 32 题上选择：问题 token 平均 log probability 增益或首个生成 token 熵下降，阈值候选为 0、预实验分位点及全收/全拒。以平均 F1 最大为主指标，平手比较 EM，再优先少接收；在测试前保存 selector.json 和哈希。这里使用开放答案生成的首 token 熵，区别于旧 SciQ 的选项熵。测试阶段不再调阈值，不选择测试表现最好的规则。

每题保存六条路径的完整输出、EM/F1/NLL、无标签特征和锁定规则的接收决策。汇总报告选定规则相对全收、全拒、相同接收率随机策略的收益；按问题做配对 bootstrap。先算接收和拒绝两个分支，不声称节省推理成本。

运行：`python run_experiment.py prepare`，然后 `python run_pipeline.py --launch`。结果自动写入 summary.json / RESULTS.md 和服务器共享实验汇总，结束时清理临时 KV。固定规则前不会运行留出集模型推理。
