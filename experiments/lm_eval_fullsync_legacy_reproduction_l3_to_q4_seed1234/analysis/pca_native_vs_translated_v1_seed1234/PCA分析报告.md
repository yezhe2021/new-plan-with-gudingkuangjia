# lm-eval 翻译后 KV 与 Qwen 原生 KV 的 PCA 分析

已在服务器 B 完成。使用 lm-eval 重新训练的 Llama→Qwen translator，MCQ 与 GSM8K 两套权重分别分析。
OpenBookQA、ARC-Challenge、GSM8K 各 32 个固定官方 prompt：8 个拟合原生 PCA 坐标，24 个做对照。共 96 个 prompt。
每个 prompt 均匀随机抽取至多 32 个被翻译的位置做几何分析；读取诊断使用完整前缀，不截断文本。
K 取 key normalization 后、RoPE 前；所有组共用同一原生 PCA 中心和坐标，不单独归一化。下面几何指标均为 36 层 × 8 KV head 的等权平均。

## 输出一致性与几何重建

首 token 输出 KL 来自本次实际 KV 注入；它衡量下一 token 分布与原生路径的差异，并不是任务准确率。

| 组 | Stage-A 输出 KL | Stage-A+B 输出 KL | 原生重建 KL 数值基线 |
|---|---:|---:|---:|
| mcq | 1.7684 | 0.7829 | 0.0000316 |
| gsm8k | 3.7410 | 0.4427 | 0.0000601 |

| 组 | KV | Stage-A NMSE | Stage-A+B NMSE | Stage-A 余弦 | Stage-A+B 余弦 |
|---|---|---:|---:|---:|---:|
| mcq | K | 0.1270 | 0.1340 | 0.9343 | 0.9310 |
| mcq | V | 0.7642 | 0.7695 | 0.7304 | 0.7259 |
| gsm8k | K | 0.1308 | 0.1401 | 0.9313 | 0.9266 |
| gsm8k | V | 0.4746 | 0.4830 | 0.7805 | 0.7747 |

这批固定样本中，Stage-B 后输出分布更接近原生，KV 几何重建误差却略有增加。表示空间接近程度与接收方输出一致性需要分别衡量。该观察不包含融合实验，也不证明接收方适配必然有效。

## 方差结构

有效秩用中心化协方差特征值的熵计算，是方差集中程度的指标；有效秩降低不等于已经证明语义信息丢失。

| 组 | KV | 原生有效秩 | Stage-A 有效秩 | Stage-A+B 有效秩 | Stage-A+B 方差/原生 |
|---|---|---:|---:|---:|---:|
| mcq | K | 38.18 | 17.73 | 17.70 | 0.612 |
| mcq | V | 72.34 | 35.98 | 35.92 | 1.089 |
| gsm8k | K | 35.82 | 18.97 | 18.89 | 0.637 |
| gsm8k | V | 73.13 | 45.13 | 45.02 | 0.721 |

## 看图方式

蓝色为 Qwen 原生，橙色为 Stage-A，红色为 Stage-A+B。箭头连接少量固定抽取的对应 token，从原生指向 Stage-A+B。
PCA 图固定展示 KV head 0 和第 1、12、24、36 层，不依据结果挑选层或 head。标题标注原生校准样本 PC1+PC2 的解释方差比例。每幅子图的坐标在三组之间相同；不同层的坐标不能直接比较。
二维重叠不能代表高维逐 token 对齐，尤其 V 的前两个主成分平均只解释 MCQ 的约 19%、GSM8K 的约 16% 方差。

## mcq 图表

校准 16 个 prompt、255 个采样 token；分析 48 个 prompt、761 个采样 token。

### Key PCA

![mcq Key PCA](figures/mcq_pca_K.png)

### Value PCA

![mcq Value PCA](figures/mcq_pca_V.png)

### 逐层高维指标

![mcq 逐层高维指标](figures/mcq_layer_metrics.png)

### 固定原生 query 的读取偏差

![mcq 固定原生 query 的读取偏差](figures/mcq_reading.png)

### 逐层逐 head 重建误差

![mcq 逐层逐 head 重建误差](figures/mcq_head_heatmaps.png)

### 方差谱

![mcq 方差谱](figures/mcq_spectra.png)

固定 query 诊断在完整前缀上施加原始位置编码并展开 GQA，保持原生 query 和末尾 self KV 不变。它隔离缓存差异；完整注入后的高层 query 可能变化，两类指标不能混同。

原有官方 lm-eval 日志在这些分析样本上的评分（复用结果，本次没有重新做完整任务评测）：
```json
{
  "openbookqa": {
    "native": {
      "acc": 0.2916666666666667,
      "acc_norm": 0.4166666666666667
    },
    "stage_a": {
      "acc": 0.25,
      "acc_norm": 0.375
    },
    "stage_b": {
      "acc": 0.375,
      "acc_norm": 0.375
    }
  },
  "arc_challenge": {
    "native": {
      "acc": 0.5416666666666666,
      "acc_norm": 0.5833333333333334
    },
    "stage_a": {
      "acc": 0.25,
      "acc_norm": 0.3333333333333333
    },
    "stage_b": {
      "acc": 0.5,
      "acc_norm": 0.5416666666666666
    }
  }
}
```

## gsm8k 图表

校准 8 个 prompt、256 个采样 token；分析 24 个 prompt、768 个采样 token。

### Key PCA

![gsm8k Key PCA](figures/gsm8k_pca_K.png)

### Value PCA

![gsm8k Value PCA](figures/gsm8k_pca_V.png)

### 逐层高维指标

![gsm8k 逐层高维指标](figures/gsm8k_layer_metrics.png)

### 固定原生 query 的读取偏差

![gsm8k 固定原生 query 的读取偏差](figures/gsm8k_reading.png)

### 逐层逐 head 重建误差

![gsm8k 逐层逐 head 重建误差](figures/gsm8k_head_heatmaps.png)

### 方差谱

![gsm8k 方差谱](figures/gsm8k_spectra.png)

固定 query 诊断在完整前缀上施加原始位置编码并展开 GQA，保持原生 query 和末尾 self KV 不变。它隔离缓存差异；完整注入后的高层 query 可能变化，两类指标不能混同。

原有官方 lm-eval 日志在这些分析样本上的评分（复用结果，本次没有重新做完整任务评测）：
```json
{
  "gsm8k": {
    "native": {
      "exact_match": 0.9166666666666666,
      "flexible_exact_match": 0.875
    },
    "stage_a": {
      "exact_match": 0.0,
      "flexible_exact_match": 0.041666666666666664
    },
    "stage_b": {
      "exact_match": 0.5833333333333334,
      "flexible_exact_match": 0.5833333333333334
    }
  }
}
```

## 复现与原始结果

- [分析代码](<analyze.py>)
- [原始机器汇总](<summary.json>)
- [完整英文机器报告](<REPORT.md>)
- 每层每个 head 的结果和二维坐标在 metrics；每个样本的输入身份及诊断在 per_sample。
- 两套权重的元数据检查、原生缓存重建和固定 query 对照均通过；权重以原有验证损失选取，没有为 PCA 调参。
- GSM8K 的长 prompt 包含 few-shot 示范，当前采样没有把示范与最终问题分开统计。
- 原生 90% 方差子空间的校准集较小，原生分析样本本身也有子空间外能量；必须与原生基线比较。
- 未保存任何原始 KV 张量。完整图形同时提供 PNG 和 PDF。共享 EXPERIMENT_RESULTS.md 已由实验程序自动更新。
