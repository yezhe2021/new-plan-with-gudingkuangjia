# HERB 接收端全 ID 预填

2026-11-21 与 2026-11-22 在新服务器 B 的 RTX 4090 上跑两轮。题目、证据前缀、Qwen token 序列、192 token 贪心生成上限、冻结的 DST Stage 1（`284ab876…`）和 rank-64 残差（`aaae0362…`）都与 [DST 对照](DST_MEMORY_RESULTS.md) 一致。两次运行的原生 KV 输出 8/8 逐 token 相同。

## 起点

译后缓存在客户 ID 上丢的不只是四位数字。同一段 ` CUST-0088` 在 Qwen tokenizer 下是 7 个 token（`ĠC` `UST` `-` `0` `0` `8` `8`），第一轮只把其中 4 个数字 token 交给接收端本地重算，剩下的 ` CUST-` 留给模型自己生成，于是大小写和位数都由接收端重新决定。第二轮把整段 7 个 token 一起重算。

预填的构造方式仍是切段传输：发送端只发 ID 段之前和之后的 KV，中间的 7 个槽位在接收端用本地 tokenizer 和前向重算，装回缓存。`receiver_prefill.py` 为此把 `DigitHandoff` / `digit_handoff` 改名为 `TokenHandoff` / `make_handoff`，`evaluate_id_patch.py` 增加 `--receiver-prefill-scope {digits,full-id}`。两臂都会断言本地重算出的 token 序列解码后与发送端记录的原始文本逐字相同。

## 第一轮：只补四位数字

| 条件 | 客户 ID 精确 | 忽略大小写 |
|---|---:|---:|
| Native KV | 8/8 | 8/8 |
| Translated KV | 0/8 | 0/8 |
| 本地重算，前三码（3 槽） | 0/8 | 0/8 |
| 换入原生 KV，前三码（3 槽） | 3/8 | 3/8 |
| 本地重算，四位数字（4 槽） | 3/8 | 7/8 |
| 换入原生 KV，四位数字（4 槽） | 3/8 | 7/8 |

四位数字版本忽略大小写后 7/8，说明数字本身在 7 题上已经对齐；其中 4 题写成了 `Cust-0088`、`Cust-0047`、`Cust-0058`、`Cust-0059`，只有 3 题碰巧保留了 `CUST-` 全大写。`herb_07` 两臂都没给出 ID。三码版本在本地重算臂上除 `herb_07` 外的七题都丢掉末位数字，`herb_07` 连整段 ID 都没有；换成原生 KV 后有三题自己补回了末位。

## 第二轮：补整段 ID

7 个槽位连续覆盖 ` ĠC` `UST` `-` 和四位数字。以 `herb_01` 为例，槽位是 69–75，本地 token id 为 `356 8553 12 15 15 23 23`，与发送端记录的 ID 槽位完全相同；其余七题只是起始位置随表头长度变化。

| 条件 | 客户 ID 精确 | 忽略大小写 | 平均 P(句点) | 平均 P(末位后再进一位数字) |
|---|---:|---:|---:|---:|
| Native KV | 8/8 | 8/8 | 5.03e-07 | 0.00 |
| Translated KV | 0/8 | 0/8 | 7.57e-03 | 0.02 |
| **本地重算全 ID（7 槽）** | **8/8** | **8/8** | 5.39e-03 | 0.06 |
| 换入原生 KV 全 ID（7 槽） | 5/8 | 5/8 | 5.82e-04 | 0.50 |

本地重算全 ID 把 3/8 拉回 8/8，与原生持平。换入原生 KV 从 3/8 升到 5/8，剩下三题的失败方式变了：不再丢数字，而是在末位之后多写一位，`CUST-00888`、`CUST-00888`、`CUST-00322`。这三题的 P(再进一位数字) 都是 0.997 以上，P(句点) 不超过 1.6e-05，即缓存里的 ID 数字是对的，但接收端不知道已经写完了。

`herb_01` 的 KV 偏差可以对照：本地重算段相对译后缓存的 K/V 最大绝对差为 59.44 / 30.18，相对换入原生 KV 只有 7.42 / 12.41。

## 逐题核对

客户 ID 按字符串精确匹配，保留前导零。

| 样本 | 参考 ID | Native KV | Translated KV | 本地重算全 ID | 换入原生 KV 全 ID |
|---|---|---|---|---|---|
| herb_01 | `CUST-0088` | `CUST-0088` | `CUST-08` | `CUST-0088` | `CUST-00888` |
| herb_02 | `CUST-0088` | `CUST-0088` | `Cust-08` | `CUST-0088` | `CUST-00888` |
| herb_03 | `CUST-0047` | `CUST-0047` | `Cust-04` | `CUST-0047` | `CUST-0047` |
| herb_04 | `CUST-0025` | `CUST-0025` | `CUST-025` | `CUST-0025` | `CUST-0025` |
| herb_05 | `CUST-0058` | `CUST-0058` | `Cust-05` | `CUST-0058` | `CUST-0058` |
| herb_06 | `CUST-0059` | `CUST-0059` | `Cust-005` | `CUST-0059` | `CUST-0059` |
| herb_07 | `CUST-0061` | `CUST-0061` | `Cust-6` | `CUST-0061` | `CUST-0061` |
| herb_08 | `CUST-0032` | `CUST-0032` | `Cust-3` | `CUST-0032` | `CUST-00322` |

译后组 8 题都没保留前导零，4 题直接丢掉了前导零以外的数字或整段 ID。

## 适用范围

这一轮只判客户 ID。全 ID 预填恢复的是那 7 个槽位，只有 `herb_02` 的整段生成与原生逐字相同，其余 7 题在 ID 之后仍然分叉，根因和修复没有重新做人工判分，不能据此认为整条答案与原生等价。结论限于冻结的 DST + Rank64 权重、这一批 8 道 HERB 题和 192 token 上限。

在服务器 B 的 `attnkv` 环境下复跑第二轮（第一轮去掉 `--receiver-prefill-scope`，用默认 `digits`）：

```bash
/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u evaluate_id_patch.py dst_rank64 \
  --project /hy-tmp/yezhe/异构模型 \
  --models /hy-tmp/yezhe/data/all_models/models \
  --output receiver_full_id_prefill_20261122 \
  --receiver-prefill --receiver-prefill-scope full-id
```

- [第二轮机器汇总](receiver_full_id_prefill_20261122/summary.json)
- [第二轮逐题生成与槽位审计](receiver_full_id_prefill_20261122/per_sample/)
- [第二轮运行日志](receiver_full_id_prefill_20261122/run.log)
- [第一轮机器汇总](receiver_digit_prefill_20261121/summary.json)
- [第一轮逐题生成与槽位审计](receiver_digit_prefill_20261121/per_sample/)
- [ID 段 KV 补丁（七槽恢复与长度匹配对照）](id_kv_patch/)
- [DST + Rank64 基线](DST_MEMORY_RESULTS.md)
