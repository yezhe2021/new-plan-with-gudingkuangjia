# HERB 双 Sender 对照

2026-11-20 完成。8 道题继续使用原来的 Slack 和 PR 证据、Qwen 模板、问题、192 token 贪心生成与评分标准。每篇证据由被分配的 Sender 独立编码，两段翻译 KV 按 Slack → PR 的固定顺序组装到同一 Qwen 缓存。互换实验只互换 Sender 分工。两段 Qwen token 拼接后与原试跑输入完全相同。Qwen 的问题续写沿用原位置。

Full-Sync 用旧 V1 的 Llama/Gemma 全 token Stage-A/B 权重。DST 用各自的 DST Stage 1 与 Rank64 最终权重。Recolor+C 用各自的基础映射与第 800 步 C 权重。Full-Sync 与 DST 在 A，Recolor+C 在旧 B，均使用 V100 32GB。旧 B 首次加载 Gemma 时进程被系统终止，直接加载到 GPU 后重跑成功，模型精度和 checkpoint 保持一致。三套运行的 Qwen 原生联合及独立答案分别逐 token 一致，联合答案逐题复现原试跑。三套均通过原生缓存重建预检。

## 双 Sender 主结果

“分工”依次列 Slack 和 PR 的 Sender。客户 ID 严格要求完整字符且只有一个确定的答案。根因按造成故障的具体机制判分，修复按关键改动判分，允许同义表述。

| 方法 | 分工 | 客户 ID | 根因 | 修复 | 三项全对 | 词元 F1 |
|---|---|---:|---:|---:|---:|---:|
| Qwen 原生独立缓存 | Qwen / Qwen | 8/8 | 7/8 | 8/8 | 7/8 | 32.57% |
| Full-Sync Stage-A/B | Llama / Gemma | 0/8 | 0/8 | 0/8 | 0/8 | 0.00% |
| Full-Sync Stage-A/B | Gemma / Llama | 0/8 | 0/8 | 0/8 | 0/8 | 1.09% |
| DST + Rank64 | Llama / Gemma | 0/8 | 3/8 | 6/8 | 0/8 | 18.84% |
| DST + Rank64 | Gemma / Llama | 0/8 | 6/8 | 7/8 | 0/8 | 25.02% |
| Recolor+C | Llama / Gemma | 0/8 | 5/8 | 8/8 | 0/8 | 20.45% |
| Recolor+C | Gemma / Llama | 0/8 | 5/8 | 8/8 | 0/8 | 21.14% |

## 同模型独立编码对照

每个模型仍分别编码两段证据。Qwen 联合编码的词元 F1 为 31.40%，独立编码后拼接为 32.57%，三项全对都为 7/8。

| 方法 | Llama / Llama F1 | Gemma / Gemma F1 |
|---|---:|---:|
| Full-Sync Stage-A/B | 0.00% | 0.93% |
| DST + Rank64 | 26.13% | 19.55% |
| Recolor+C | 25.94% | 22.03% |

Full-Sync 的四个翻译组均未恢复事故事实，生成主要为空答或重复标签、词语。两种混合分工分别有 8/8 和 3/8 道题触顶。DST 和 Recolor+C 的翻译组全部自然终止。两个 DST 混合组、两个 Recolor+C 混合组的 F1 均低于各自的 Llama / Llama 独立编码组。

## 逐题复核

每格顺序为客户 ID、根因、修复，`1` 代表正确。Full-Sync 两种分工均为 `000`。

| 样本 | 原生独立 | DST Llama / Gemma | DST Gemma / Llama | Recolor+C Llama / Gemma | Recolor+C Gemma / Llama |
|---|---|---|---|---|---|
| herb_01 | 111 | 001 | 011 | 001 | 001 |
| herb_02 | 111 | 011 | 011 | 011 | 011 |
| herb_03 | 111 | 011 | 011 | 011 | 011 |
| herb_04 | 111 | 001 | 000 | 011 | 011 |
| herb_05 | 111 | 001 | 011 | 011 | 011 |
| herb_06 | 101 | 001 | 001 | 001 | 001 |
| herb_07 | 111 | 000 | 011 | 001 | 001 |
| herb_08 | 111 | 010 | 011 | 011 | 011 |

- `herb_01` 需要指出同步处理阻塞 UI 线程。DST 在 Llama 读 PR 时说出了同步机制，Gemma 读 PR 时只复述了失去响应。两种 Recolor+C 分工都答对异步修复，根因归因于请求量和阻塞，未交代同步处理机制。
- `herb_04` 的修复需要提到时间平滑。DST 的 Gemma / Llama 只说提高区分能力。两种 Recolor+C 分工都给出了时间平滑和灵敏度调整。
- `herb_05` 的 Recolor+C Llama / Gemma 同时给出了 `CUST-0058` 和 `CUST-058`，按严格规则扣分。只取首个报告的编号时，这组的 ID 与三项全对均为 1/8。
- `herb_07` 的 DST Gemma / Llama 同时指出过期统计与缺失索引，也给出两项修复。反向分工的修复只提更新统计。两种 Recolor+C 分工都给出了两项修复，但根因句均缺少至少一个必要原因。
- `herb_08` 的 DST Llama / Gemma 读出了时区处理错误，修复句只说正确解析和排序。Gemma / Llama 明确说要更新解析逻辑来正确处理时区。

这批人工编写题表明，两路异构 KV 可以保留技术语义，效果受翻译器和证据分工影响。客户编号仍是共同短板。原生对照表明在本批短证据上独立拼接本身可用。组间还包含各自的 Sender 捕获、对齐和训练差异，结果适用于这三条完整路径及这批题。

- [冻结样本](samples.json)
- [Full-Sync 逐题结果](multisender/fullsync_ab/per_sample/)
- [DST 逐题结果](multisender/dst_rank64/per_sample/)
- [Recolor+C 逐题结果](multisender/recolor_c/per_sample/)
