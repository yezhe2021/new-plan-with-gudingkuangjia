# 细粒度KV误差与功能贡献：C，ARC32+HotpotQA32

只实现设计稿design.md的实验一、二，不训练、不做维度修复。沿用C和旧样本各前32题，全36层×8KVHead×全部正文Token×128维流式计算。

误差图谱保存post-RoPE K、inverse-RoPE但仍post-k_norm的K，以及V的逐维偏差/MAE/RMSE、Native尺度归一化统计和逐Token统计。归一化下限固定为max(0.05×Head RMS,1e-6)。不把原始数值尺度直接当成敏感度。

功能分析在原生Q、真实Translated Cache产生的Q两种参考下分别进行，位置与上下文相同。Gold上下文用全部标准答案探针；辅助上下文复用Phase1的C自生成Token和最多8个固定位置。每个KVHead严格匹配4个QueryHead。计算带符号q×deltaK贡献、逐维抵消、RoPE配对贡献、Attention Logit变化、Softmax JS和局部四种K/V组合的post-Wo差异。此处四组合仅是局部注意力算子计算，不执行维度/整模型修复。

绝对贡献的均值可因子分解为平均|Q|×平均|误差|，不能单独证明功能敏感性。全量贡献张量不分配；逐探针/Token分块，仅保留统计边际、关键Token的完整128维有符号数组与代表性热图。每个样本NPZ是派生统计，不是KV缓存。

RoPE后单个坐标有位置依赖，额外保存RoPE配对(d,d+64)的排名和带符号和，GPU检查配对点积在Key坐标系转换下不变。只将配对Top4/Top8作为分析读数，不修复8/16维。

统计按32道题cluster bootstrap，控制层与KV头差异，不把头或Token当独立样本。不声称已有独立验证；未来验证需排除所有曾查看的ARC128/Hotpot512及校准题。

```bash
python run_pipeline.py launch
tail -f logs/pipeline.log
```

后台prepare→smoke→ARC32→HotpotQA32→统计和绘图。错误停止并保存，Transformer等旧训练保持暂停。不自动监控或上传，不自动进入实验三。
