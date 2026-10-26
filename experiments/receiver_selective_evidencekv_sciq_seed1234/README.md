# Receiver selection of external evidence KV

服务器B，冻结Llama3.2-3B、原grouped Stage-A + rank64残差Stage-B及Qwen3-4B。不安装或加载Q/O Reader，不训练模型参数。

Sender只读SciQ的support；Receiver通过外部KV接收support，随后原生读取问题、选项和共同回答前缀。Reference模板首token为共同native anchor，不含证据内容。两模型token按既有Full-Sync字符对齐。接收/拒绝各条件内保持证据槽位数量和绝对位置完全一致；拒绝时在每一层屏蔽全部证据key列，从问题处理开始屏蔽，避免问题隐状态已读入证据。额外的原生Qwen证据KV只用于正向参照，不参与gate。

采用已有SciQ官方test parquet的未用样本，排除历史SciQ64及已知模型适配训练/验证的精确重复问题，支持文本也去重。固定seed选择32题calibration、128题evaluation；这是官方test内部的独立校准/评测划分，不是官方validation/test基准。提供标注support，不评估检索。

每题四种KV：正确证据、同split其他题的证据、随机50%槽位替换成错题KV、匹配本题每层/头/channel均值方差的高斯噪声。错题KV在RoPE前用相同保序索引调整K/V长度；donor按长度配对且不读取答案。噪声保留目标聚合统计，只是扰动对照。

整段选择规则使用两个无需gold的信号：证据带来的问题token平均log-prob增益，以及候选答案熵的下降。各信号的阈值只在calibration上做小网格选择，先保存selector.json，再运行evaluation；测试逐题保存锁定selector的SHA256。固定零阈值、始终接收、始终拒绝同时报告。校准阶段允许最终选择退化为始终拒绝；此时不能宣称动态选择有收益。

每题选择需要计算接收和拒绝两个分支，是效果可行性实验，不宣称计算或通信节省。报告正确率、候选NLL、接收率、相对始终接收/拒绝的配对题级bootstrap区间、相同接收率的随机选择期望，以及读取gold的oracle上限（仅诊断，不是可部署方法）。四种扰动等权汇总仅代表该人为混合分布。

GPU预检：屏蔽后真实KV/噪声完整logits必须完全相同；原生证据cache与完整原生prefill分数一致；确认没有Q/O LoRA、参数全部冻结。先做8道calibration题预检，不根据表现筛选题目。

```bash
python -m unittest -v test_gate_rules
python run_experiment.py prepare
python run_pipeline.py --launch
```

流水线依次capture、translate、preflight、calibration、calibrate、test、summarize。失败停止，不自动重试。只清理本实验临时张量；结果自动追加服务器EXPERIMENT_RESULTS.md。输出summary.json、selector.json、evaluation/{calibration,test}/逐题文件和logs/pipeline.log。不自动上传GitHub。
