# HERB 证据 KV 试跑

8 道人工编写的故障调查题，证据来自 HERB 固定 revision `a00bca08f9118e482e6de9951fdcb654fbed5343` 发布的合成企业记录。每题两份来源按照 Slack → GitHub PR 固定顺序组成一段前缀。Slack 提供受影响客户和关联 PR，PR 提供根因及修复。`samples.json` 保留引用 ID、产品 Git blob、原始证据片段和对应的官方 citation 交集。

冻结 Llama-3.2-3B-Instruct、Qwen3-4B，以及 HotpotQA Same128 的 Grouped HeadMix4 Stage-A `best.pt` 和 rank-64 residual Stage-B `last.pt`。SHA-256 与 Same128 对照一致。服务端通过 `--models-root` 指定迁移后的模型目录。历史一条 Same128 样本的两端 tokenizer 输出及 Full-Sync 映射通过复查。模型与翻译器全程推理，Reader 关闭。

四组分别为 Q-only 紧凑问题提示、完整文本 prefill、原生 KV 续写、翻译 KV 续写。贪心采样、Qwen thinking 关闭。后两组按相同目标位置续写。Translated KV 的缀接使用固定原生模板 anchor 和译后证据 KV。参考答案只用于单独的前向评分和事后判分，逐题 KV NMSE 使用 post-k_norm、pre-RoPE 的 Qwen 目标。Native 预检逐题比较完整 prefill、原生续写和原生 KV 经过翻译缓存注入器三条路径在相同固定后续 token 上的 logits。

先用 96 token 跑通全部四组。观察到多组触顶后，冻结同一批数据、提示词和 checkpoint，用统一的 192 token 上限复跑全部四组。两次保留独立的逐样本文件、预检、日志及机器汇总。

在服务器 B 的 `attnkv` 环境下使用下列命令复跑 192 token 版本。该目录需与服务器上的 Same128 归档目录同级。

```bash
/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u evaluate.py \
  --models-root /hy-tmp/yezhe/data/all_models/models \
  --max-new-tokens 192 --output cap192
```
