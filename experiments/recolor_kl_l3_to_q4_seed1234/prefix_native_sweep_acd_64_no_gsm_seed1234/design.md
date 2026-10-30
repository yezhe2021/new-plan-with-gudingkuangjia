# 前缀原生移植扫描（Prefix Native Sweep）实验设计

状态：设计稿，未实现。
对象：`recolor_kl_l3_to_q4_seed1234` 的三个臂 A / C / D。
前置实验：`layergroup_oracle_repair_acd_64_no_gsm_seed1234`（已 completed，manifest `f824025e…`）。

---

## 0. 要回答的问题（写死，不许中途改）

> **翻译误差在深度上是"集中"还是"铺开"的？**

这个问题决定下一步投哪里：

- 集中 → 分层修复有天花板接近原生，值得继续投残差；
- 铺开 → 每层修一点永远补不满，分层修复这条路本身到不了 90%，应该改接口。

这不是一个提升分数的实验，是一个**带预先注册判据的决策实验**。如果判据跑完是"模糊"，就老实报模糊，不许事后挑解释。

---

## 1. 实验定义

对每个臂，把 Qwen 的 36 层 cache 按**前缀**切开：

```
prefix00 = 层 1-36 全翻译            ← 等于 umasked 基线（复用）
prefix06 = 层 1-6  原生 | 7-36 翻译
prefix12 = 层 1-12 原生 | 13-36 翻译
prefix18 = 层 1-18 原生 | 19-36 翻译
prefix24 = 层 1-24 原生 | 25-36 翻译
prefix30 = 层 1-30 原生 | 31-36 翻译
prefix36 = 层 1-36 全原生            ← 必须精确等于原生基线
```

- 干预类型固定为 **`kv`**（K 和 V 一起换）。理由是这一轮问的是"如果这几层是完全正确的"，K-only 是另一个问题（路由修好了但内容还是翻译的）。
- 层号沿用现有代码约定：`layers=list(range(k))` 是 0-based，即 `k=6` 表示物理层 1-6。**与现有 `conditions()` 的 `range(g*6,(g+1)*6)` 语义一致，不要另起一套。**
- 6 个新 condition × 3 个臂 × 64 题 × 3 个数据集 = **3456 个新配置**，复用 unmasked 后约 **30 分钟**。

### 为什么是 64 题而不是 128

必须和前置实验**用同一批 64 题、同一顺序**。这样 `prefix24` 和 `g4_kv` 可以**逐题配对**比较，这比两个独立估计对比强得多，也是本实验的一个副产品（见 §6）。

---

## 2. 目录与依赖

```
experiments/recolor_kl_l3_to_q4_seed1234/
  prefix_native_sweep_acd_64_no_gsm_seed1234/
    config.json
    experiment.py
    run_pipeline.py
    README.md
```

依赖三个上游，全部按 manifest 身份断言：

| 用途 | 路径 | 身份 |
|---|---|---|
| `core.py` / `train.py` / `evaluate.py` / `prepared/test.jsonl` | 父目录 `recolor_kl_l3_to_q4_seed1234` | `f6b176c9…` |
| 复用 `unmasked` 与 `native` 结果 | `layergroup_oracle_repair_acd_64_no_gsm_seed1234` | `f824025e…` |
| A/C/D 权重 `runs/*/epoch4_step800.pt` | 父目录 | 见父 manifest |

复用项（逐题、逐臂断言 `manifest_sha256` / `id` / `method` / `condition['layers']` 一致后才读）：

- `evaluation/{task}/per_config/native/{id}.json` → 原生基线
- `evaluation/{task}/per_config/{A,C,D}/unmasked/{id}.json` → 翻译基线

**不要重算这两个**，否则会出现和前置实验不可比的 FP16 抖动。

---

## 3. 代码改动（相对 `layergroup_oracle_repair_acd_64_no_gsm_seed1234/experiment.py`）

整个实验的核心改动只有 `conditions()` 一个函数。其余靠复制。

### 3.1 `conditions()`

```python
def conditions():
    return [dict(name='unmasked', mode='none', layers=[])] + [
        dict(name=f'prefix{k:02d}', mode='kv', layers=list(range(k)))
        for k in CFG['prefix_cuts']]
```

`replace()`、`evaluate()`、per-config 断点续跑、`first_logits` 保护性检查**全部原样可用** —— 现有 smoke 里的

```python
selected = l in con['layers'] and (con['mode']=='kv' or ...)
```

对前缀条件天然成立，不需要改。

### 3.2 `evaluate()` 里的进度常数

```python
# 原：task_configurations=4864
# 新：64 × (3 方法 × 7 条件 + 1 native) = 1408
task_configurations=1408,
```

### 3.3 `make_summary()`

删掉 `substitution_minus_same_group_mask`（没有 mask 条件了），保留 `metrics` / `comparisons` / `recovery_ratio`，**新增组边际增益**：

```python
def marginals(metrics, method, key):
    """相邻切点之间新增的 6 层原生，各自贡献多少。"""
    prev_k, prev = 0, metrics[method]['unmasked'][key]
    out = []
    for k in CFG['prefix_cuts']:
        cur = metrics[method][f'prefix{k:02d}'][key]
        out.append(dict(k_from=prev_k, k_to=k, delta=cur - prev))
        prev_k, prev = k, cur
    return out
```

`recovery_ratio` 公式不变（`(acc_k − acc_unmasked) / (acc_native − acc_unmasked)`）。

### 3.4 `summarize()` 输出的表

```
| Method | k | ARC64 | HellaSwag64 | HotpotQA64 EM | HotpotQA64 F1 | recovery(ARC) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 36 | ...native... |
| A | 0 | ... | (unmasked)
| A | 6 | ...
...
```

外加一张边际表：

```
| Method | 新增层 | ARC Δ | Hotpot EM Δ |
|---|---|---:|---:|
| A | 01-06 | +0.4 | -3.1 |
| A | 07-12 | -0.4 | +1.6 |
...
```

### 3.5 RUN 命令（沿用现有 pipeline）

```bash
cd /hy-tmp/yezhe/异构模型/recolor_kl_l3_to_q4_seed1234/prefix_native_sweep_acd_64_no_gsm_seed1234
python run_pipeline.py launch
```

`run_pipeline.py` 直接复制前置实验的，`actions` 顺序不变：
`prepare → smoke → evaluate arc → evaluate hellaswag → evaluate hotpotqa → summarize`。

---

## 4. `config.json`

```json
{
  "name": "prefix_native_sweep_acd_64_no_gsm_seed1234",
  "seed": 1234,
  "parent": "/hy-tmp/yezhe/异构模型/recolor_kl_l3_to_q4_seed1234",
  "parent_manifest_sha256": "f6b176c9915d2e968ce560fb2f981f6e2cefdd9405a9e2beeac9afa0a739b4c6",
  "oracle_experiment": "layergroup_oracle_repair_acd_64_no_gsm_seed1234",
  "oracle_manifest_sha256": "f824025ec02cc94fb39c3b804a3201eaffd115188b2b52c1bb4cc81d092c3ee2",
  "methods": ["A", "C", "D"],
  "test_counts": {"arc": 64, "hellaswag": 64, "hotpotqa": 64},
  "prefix_cuts": [6, 12, 18, 24, 30, 36],
  "mode": "kv",
  "training": false,
  "gsm8k": false,
  "reader": false,
  "automatic_upload": false,
  "continuous_monitoring": false,
  "decision_rule": {
    "localized": "存在组 g，其边际增益 >= 2 × 各组边际中位数，且 >= 8pp",
    "distributed": "所有组边际都在 [-3, +8]pp 内，且 准确率 vs k 的线性拟合 R^2 >= 0.9",
    "ambiguous": "以上都不满足；如实报告，不宣称任何一侧",
    "note": "本判据在运行前写定，不允许根据结果调整阈值或事后挑选解释。"
  },
  "caveats": [
    "Same 192 questions, same immutable parent order as layergroup_oracle_repair_acd_64_no_gsm_seed1234; reused unmasked and native results are identity-checked, never recomputed.",
    "Intervention is kv-only: each condition replaces body K and V with native Qwen values for layers 1..k, after cache assembly, in post-k_norm/post-RoPE coordinates on the matching Qwen token/position grid. Public header, positions and suffix are untouched.",
    "prefix36 is not an independent measurement: it must equal the native baseline exactly and serves as a full-scale endpoint check.",
    "Native KV is offline oracle information. This measures the potential value of perfect knowledge at a given depth, not a deployable repair.",
    "Because the sweep is cumulative, group marginals are conditional on everything below already being native. They are NOT the same quantity as the isolated single-group effects in the parent experiment; do not compare the two as if they were.",
    "Non-monotonicity is possible and reportable. No per-dataset best cut is promoted as a method.",
    "No GSM8K in this sweep: generation cost and a different protocol. Add only after the curve is read.",
    "Greedy decoding: 1 question = 1.56pp at n=64. Curve shape is the primary readout; individual point CIs are wide by design.",
    "Pipeline fails on error, preserves partial results per configuration, no automatic upload or monitoring, does not resume other paused experiments."
  ]
}
```

---

## 5. Smoke 检查（在现有基础上改两处）

保留现有全部检查，因为 `conditions()` 换掉后它们仍然逐条覆盖新条件：

1. unmasked 与历史预测一致（`M.parity`）
2. 自替换 logits 逐位不变
3. **全原生端点 == 原生基线**（现有 171-173 行；这条现在也是 `prefix36` 的定义）
4. 36 层 cache 的每一层、每一个 tensor：`[0:start]` header 未被改动，`[start:end]` 命中/未命中与 `con['layers']` 严格一致
5. native cached vs direct prefill 的预测一致

**新增两条断言：**

```python
# prefix36 必须与原生基线逐分数相等（不只是预测相等）
assert value_prefix36['scores'] == native_value['scores']   # hotpotqa 比对 generation

# prefix00 必须与 unmasked 逐分数相等
```

这两条是整条曲线两端的锚。锚不成立，曲线无意义。

**新增一条只记录、不断言的诊断：**

```python
# 单调性不强制，但必须记录
monotone = all(b >= a - 1e-9 for a, b in zip(accs, accs[1:]))
```

曲线不单调是可报告的结果（说明"更多原生"在某段反而有害），不是 bug。

---

## 6. 副产品：一次对"错配假说"的直接检验

前置实验里有两组数字很反常：

| 配置 | ARC64 |
|---|---:|
| A 的 `g4_kv`（只换 19-24） | 75.00% |
| A 的 `g4_v`（只换 19-24 的 V） | **23.44%** |
| A 的 `g5_mask` | 18.75% |
| D 的 `g4_v` | **20.31%** |
| C 的 `g4_v` | 62.50% |

同一个干预，A/D 掉 42-44 个点，C 只掉 3 个点。如果"注入的 KV 是别处算出来的"这个假说成立，那**在渐进扫描里同样的层不该崩**——因为 `prefix24` 时下层已经是原生、上下文自洽。

所以：

```python
# 前置实验逐题结果
old = read(oracle / 'evaluation/arc/per_config/A/g4_v/{id}.json')['result']
# 本实验逐题结果
new = read(here / 'evaluation/arc/per_config/A/prefix24/{id}.json')['result']
```

用 `C.paired()` 做逐题配对。**如果 `prefix24` 显著高于 `g4_v`，错配假说成立，前置实验那 54 个替换配置的数字全部要被理解为低估，而 18 个 mask 配置的解释仍然干净。** 这一条本身就值这 30 分钟。

---

## 7. 判据与后续动作（运行前定死）

| 曲线形状 | 判据 | 结论 | 下一动 |
|---|---|---|---|
| 某段突然抬升 | localized | 误差集中在一条窄带 | 只在该带训残差；同时重查该段 Recolor depth stencil 的对应关系 |
| 近似线性爬升 | distributed | 误差均匀铺在每层 | 分层修复封顶，转接口层面（传隐状态 / 部分层重算） |
| 早期就抬升（1-6） | localized（底层） | 底层对齐本身有问题 | 先查 sender token/位置对齐，别训残差 |
| 都不满足 | ambiguous | 分辨率不够 | 在抬升段用 3 层步长补 3-4 个点，其余不动 |

**不做的事：** 不根据曲线挑一个"最好的 k"当成方法报出去。这条曲线是诊断，不是方法。

---

## 8. 预计开销

| 项 | 数量 |
|---|---|
| 新配置 | 6 prefix × 3 臂 × 64 题 × 3 任务 = 3456 |
| 复用 | unmasked 576 + native 192 |
| 参考速度（前置实验 14400 配置 / 87 分钟 ≈ 165 配置/分钟） | **≈ 21 分钟纯推理 + 模型加载 ≈ 30 分钟** |
| 磁盘 | per-config JSON 约 4.2k 个，每个 ~2KB，另加 per-sample，**< 100MB** |

---

## 9. 交付物清单

- [ ] `config.json` / `experiment.py` / `run_pipeline.py` / `README.md`
- [ ] `smoke/completed.json`（含两条新锚断言）
- [ ] `evaluation/{arc,hellaswag,hotpotqa}/summary.json`（含 `marginals`）
- [ ] `summary.json`
- [ ] `RESULTS.md`：曲线表 + 边际表 + 判据结论一行
- [ ] 提交 commit message 里写明**判据结论**（localized / distributed / ambiguous），不要只写"add results"
