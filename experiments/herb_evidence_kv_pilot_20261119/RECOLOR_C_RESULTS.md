# HERB 原版 Recolor+C 对照

使用同一份 8 题样本、证据文本、Qwen token 序列及 192 token 贪心生成上限。载入历史归档的 Recolor 基础映射和 C 分支 rank-64 第 800 步权重，文件 SHA-256 与训练归档一致。Recolor+C 保留 RAW Llama 和原有末端 token 对齐，其 Qwen 端仅原生预填充公开聊天头。HeadMix 方案采用 Llama chat、Full-Sync 和固定首正文 token anchor。结果比较两条完整传输路径的行为。

| 条件 | 客户 ID 正确 | 根因正确 | 修复正确 | 三项都正确 | 词元 F1 |
|---|---:|---:|---:|---:|---:|
| Native KV（新 B） | 8/8 | 7/8 | 8/8 | 7/8 | 31.40% |
| HotpotQA HeadMix4 A+B（新 B） | 0/8 | 0/8 | 0/8 | 0/8 | 6.08% |
| **Recolor+C（新 B）** | **1/8** | **7/8** | **8/8** | **1/8** | **24.75%** |
| Recolor+C（A） | 1/8 | 7/8 | 8/8 | 1/8 | 24.66% |

两台服务器的 Native KV 对照都逐题复现新 B 的原始生成，8/8 题逐 token 一致。Recolor+C 在两台服务器上八条答案都自然终止，7/8 题逐 token 一致，另一题仅措辞和附加细节有差异，逐项判分一致。诊断探针和生成各用独立的缓存副本。

新 B 迁移后保留了可恢复训练的 C `last.pt`，缺少对应的基础 Recolor 映射和评测用 `epoch4_step800.pt`。本次从 A 补齐两个原版文件，校验后放在新 B 此实验的 `recolor_c_assets` 目录。

## 逐题核对

客户 ID 按字符串精确匹配，保留前导零。根因和修复沿用原试跑的事实标准。

| 样本 | 参考 ID | Recolor+C 的 ID（A、B 一致） | 根因 / 修复 |
|---|---|---|---|
| herb_01 | `CUST-0088` | `CUST-88` | ✓ / ✓ |
| herb_02 | `CUST-0088` | `CUST-8` | ✓ / ✓ |
| herb_03 | `CUST-0047` | `CUST-047` | ✓ / ✓ |
| herb_04 | `CUST-0025` | `CUST-2025` | ✓ / ✓ |
| herb_05 | `CUST-0058` | `CUST-0058` | ✓ / ✓ |
| herb_06 | `CUST-0059` | `CUST-05` | ✗ / ✓ |
| herb_07 | `CUST-0061` | `CUST-61` | ✓ / ✓ |
| herb_08 | `CUST-0032` | `CUST-32` | ✓ / ✓ |

`herb_06` 给出的根因只重述时区处理不当，修复正确提到聚合前归一化时间戳。其余七题同时保留了根因和修复。七条错误客户 ID 中，四条只丢失前导零，三条则插入或丢失了其他数字。

- [新 B 上的逐题回答](recolor_c_cap192_server_b/per_sample/)
- [新 B 上的机器汇总](recolor_c_cap192_server_b/summary.json)
- [A 上的逐题回答](recolor_c_cap192_server_a/per_sample/)
- [A 上的机器汇总](recolor_c_cap192_server_a/summary.json)
- [冻结的逐题输入](recolor_inputs.json)
- [原四组结果](RESULTS.md)
