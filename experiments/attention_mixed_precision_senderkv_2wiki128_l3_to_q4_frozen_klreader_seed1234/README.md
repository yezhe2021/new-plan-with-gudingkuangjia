# Attention-aware Sender KV quantization: frozen KL-only Reader

服务器 B，Llama3.2-3B → grouped Stage-A + rank64 residual Stage-B → Qwen3-4B + 原先 KL-only Q/O-LoRA Reader。全部参数冻结，无训练。用户明确不使用新 task-CE+KL Reader，也不等待其评测结束；不停止其他实验。

固定旧 Reader 的 epoch4/step1024。数据为128道新的2Wiki受控证据题，与前轮24题按ID/问题排除；标注支持句+8个干扰句，非官方完整上下文基准。相同证据、问题、短答案前缀 `The answer is:`。使用已建立的原生 chat / Full-Sync 字符对齐接口、原始模型和RoPE约定。最长prompt768、标准答案16 tokens、greedy32；报告截断比例。

## 量化与对照

- FP16，全部INT4，最高注意力10% INT8+其余INT4，3个随机种子10% INT8+INT4，最低注意力10% INT8+INT4；共7个条件，对应A–E五类。
- 排序来自未量化Llama的最后一个强制前缀query，先按query heads/层聚合，再由字符偏移投影到证据文本单元及传输token。无gold前缀，不在量化后重排。前缀读取问题，因此不是未知未来问题的通信实验。
- 原Full-Sync映射使用的Sender位置先去重、按原位置排序。所有条件仅量化这些位置的pre-RoPE K/V。仅在证据重叠位置中选出 ceil(0.1*Nwire) 个保护位置；各层、KV头、K/V共用位置选择。
- 固定非对称min/range：K每channel沿64个传输位置分组，V每token沿64个channel分组。4/8bit使用相同存储的FP16 min/range，避免排序改变scale。借鉴KIVI的非对称轴，**不是未经修改的KIVI/KVTuner复现**；K的分组在按原序排列的传输位置上，不跨未传输位置重建连续窗。
- INT4真实nibble打包、INT8真实uint8，完整解码后再推理。数据包包含位宽掩码、原Sender位置、Receiver映射、FP16 min/range、维度和段长度；实际 len(packet) 计费，不以torch tensor元素数冒充通信字节。
- C/D/E使用相同INT8位置数、相同分组与元数据布局，逐题断言实际字节严格相等。名义数据位宽为4+4*m/N，ceil后约4.4bit；总有效位宽高于此数。全INT4的预算较低，作为参考，不称为等预算比较。
- 同一量化数据包解码的K/V用于两条路径：替换Llama原生缓存中对应位置；或按原对齐映射扩展，再输入冻结Stage-A+B和KL-only Reader。逐题核对数据包hash相同。Llama其余缓存位置、Qwen原生anchor、后续生成token均不量化。
- 保持原Full-Sync native body prefill路径；这是精度/通信payload实验，不宣称无prefill、实际网络传输加速或低显存推理。本轮仍在接收端完整反量化。

## 指标与核验

主要为各路径相对其FP16基线的标准答案 ΔNLL/token、ΔNLL/字符；逐题配对比较高注意力减随机均值/低注意力。另报greedy EM、截断率、第一预测位置KL、输入和映射输出K/V NMSE、实际wire bytes和effective bits/element。随机3次先题内平均，bootstrap以题为单位。Llama/Qwen绝对NLL/token不横向比较。

CPU测试覆盖INT4奇数长度打包、FP16精确往返、位置映射、等字节预算、局部保护不改变其他token误差、常数组及损坏数据包。GPU smoke核对原生FP16缓存还原、两条路径同数据包、有限输出和冻结无梯度；只做前1题，正式128题仍保留它，不按smoke表现调参或选题。

FP16异构基线可能受2Wiki任务迁移影响；若baseline低，正确率仅作诊断。不同量化策略都接近FP16也可能是INT4误差太小，不能直接否定注意力重要性。

## 运行

```bash
cd /hy-tmp/yezhe/异构模型/attention_mixed_precision_senderkv_2wiki128_l3_to_q4_frozen_klreader_seed1234
/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -m unittest -v test_quantization
/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python experiment.py prepare
/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python run_pipeline.py --launch
```

独立session顺序执行两模型smoke、Llama正式评测、Qwen正式评测及自动汇总；失败停止，不自动重试。启动不依赖其他实验完成，但加载前检查可用显存，尊重无关作业。仅这轮新建的可再生Sender临时张量在接收方逐题结果持久化后清理；历史缓存不动。

`logs/pipeline.log`查看进度；`evaluation/{llama,qwen}/<id>.json`含逐题结果；最终`summary.json`自动追加至服务器共享`EXPERIMENT_RESULTS.md`，不自动上传GitHub。

参考：https://github.com/jy-yuan/KIVI ；https://github.com/cmd2001/KVTuner 。
