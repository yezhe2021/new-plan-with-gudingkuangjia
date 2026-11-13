# HELMET 人工复核：逐条重判 2400 个答案

对 `evaluation/helmet/{1024,2048,4096,8192}` 下全部 200×4×3 个条件重新判对错，
不采信 `summary.json` 里的 `helmet_subem` / `em` / `f1`。原始逐条判定见同目录
`manual_judgments.csv`（arm, length, id, question, gold, aliases, prediction,
manual, reason, official_subem, em, f1）。

## 结论

| 臂 | 1k | 2k | 4k | 8k | 全部 | 官方 subEM | 官方 EM | 官方 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 73.50 | 67.50 | 63.00 | 61.00 | **66.25** | 63.50 | 21.12 | 38.15 |
| recolor_c | 52.50 | 51.50 | 49.00 | 39.50 | **48.12** | 48.50 | 9.25 | 21.54 |
| dst_rank64 | 60.00 | 53.50 | 45.50 | 37.00 | **49.00** | 40.12 | 33.38 | 46.27 |

对 native 的配对差（800 条配对 bootstrap，4000 次）：recolor −18.12 [−21.88, −14.37]，
DST −17.25 [−21.50, −13.12]；保持率 0.726 / 0.740。
DST 对 recolor **+0.88 [−3.00, +4.75]，跨零，两者打平**。
8k 相对 1k 的下降：native −12.5pp，recolor −13.0pp，DST −23.0pp。

官方 subEM 与人工判定相差 −0.38pp（recolor）、+2.75pp（native）、**+8.88pp（DST）**。
按 800 条计，官方分别漏判 32 / 49 / 72 条、误判 35 / 27 / 1 条。

**关键变化**：官方口径下 DST 比 recolor 低 8.4pp，人工口径下两者只差 0.9pp，
且排序反转。差异不来自模型能力，来自评分器对答案长度的偏好（见下）。

## 判定规则

1. 剥掉 `Answer:` 前缀和 markdown，取模型实际给出的答案；
2. 是/否题只看首个 token 的极性；
3. 实体题：gold（或其别名）以词边界出现在**首句**即算对；只在中段解释里出现、
   首句给的是另一个答案的，算错；带 "the document does not mention" 之类回避的算错；
4. 人名/地名允许省略中间名、昵称、州名、父名后缀（"Lee Hazlewood" 对
   "Barton Lee Hazlewood"、"Crested Butte" 对 "Crested Butte, Colorado"）；
   日期允许月日换序；明显拼写错误（"Donhue"/"Donohue"）算对；
5. 只给年份而 gold 是完整日期、或只给三项中的一项，算错。

全部 216 条与官方不一致的样本都逐条读过并单独裁决；规则未覆盖的比较题
（X 或 Y、谁更早/更多）另做了逐条结论核对。

## 官方评分器为什么会偏

`experiments/recolor_kl_l3_to_q4_seed1234/metrics.py` 的 `normalize()` 是
`''.join(c for c in text.lower() if c not in string.punctuation)`——删标点但**不补空格**：

- `19th-century` → `19thcentury`，而 `19th century.` → `19th century`，永远匹配不上；
- `James Gay-Rees` → `jamesgayrees`，`James Gay Rees` → `james gay rees`，同样失败。

`helmet_subem` 又要求整条 alias 作为子串出现，gold 是全名/全称时就漏判短答案：

- `Barton Lee Hazlewood` ← `Lee Hazlewood`
- `the U.S. state of Kansas` ← `Kansas`
- `University of Texas at Austin` ← `University of Texas`
- `rhythm guitar` ← `guitar`

反过来，子串匹配会把"提到"当成"答对"：

- gold `no`，答 `Yes. ... world No. 8 in doubles`——`no 8` 里命中 ` no `；
- gold `Manchester Orchestra`，答 `DC Talk has the most members. ... Manchester Orchestra is a quartet`；
- gold `Gardnerville`，答 `Minden, Nevada has a greater population ... while Gardnerville had 5,656`；
- recolor 有 9 条 `To determine who is younger between ... we need to compare their birth dates.`
  在 64 token 处截断，根本没给答案，仍被记对。

这三类错误的净效果与答案长度强相关：DST 平均 3–4 个 token 精确作答，被漏判 72 条；
recolor 写整段推理，被误判 35 条。所以 subEM/F1 上的"DST 保持率崩到 0.63"里，
约 9pp 是度量假象，不是能力差距。F1 那一列（DST 46.27 > native 38.15）同样是长度效应，
不能单独引用。

## 判断口径的少数派

- `rhythm guitar` 判 `guitar` 算对（12 条）：问的是乐器，节奏吉他是吉他的一种；
- 问"which person"但 gold 是角色列表（`musician, editor and composer.`，12 条）：
  点名 John Carpenter 的算对，点名 William Blake Herron 的算错；
- 只给年份的 7 条（`1945` 对 `28 June 1945` 等）算错；
- 三项并列只给一项的 1 条（`New York City` 对 `New York City, Long Island, and Westchester`）算错；
- 个别 gold 与问题不符（问年份、gold 是雕像名）仍按 gold 判，未做修正。
