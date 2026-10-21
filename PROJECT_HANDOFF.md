# 项目交接（2026-10-21）

## 当前研究对象

异构模型 KV 复用：Llama3.2-3B-Instruct Sender → Qwen3-4B Receiver。
Stage-A 做跨模型 KV 映射，Stage-B 在 Receiver KV 空间做功能残差修正；
最近又加入冻结 Writer 上的 Q/O-LoRA Reader，研究接收方读取翻译 KV 的能力。
历史中还有 Gemma、反向传输、QASPER、拼接、PCA、lm-eval 等实验，
请按实验目录判断协议，不能把历史不同设置的准确率直接混合比较。

当前执行服务器为 **B**。新电脑接入看 [NEW_COMPUTER_SETUP.md](NEW_COMPUTER_SETUP.md)。
本文件是导航与状态交接，不替代或手工改写机器生成的实验结果汇总。

## 最近完成的主线

1. [MCQ HeteroFold-chat Full-Sync Writer](experiments/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/)：
   当前 Stage-A 来源与旧 MCQ Stage-B 对照。它是我们的 Writer，不是论文 HeteroFold 映射。
2. [UFD Mixed-prefix Stage-B](experiments/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234/)：
   冻结现有 MCQ Stage-A，新训 rank64 残差；OBQA/ARC/GSM8K/MMLU-Pro 各512条，
   总2048，4 epoch、有效 batch8、1024次更新。
   统一 full-vocabulary forward KL(Native || Student)，无 Gold CE、choice KL、attention/KV loss。
   首 epoch Native prefix，后3个 epoch 每批精确一半在线 Student prefix；teacher 读取相同 prefix。
   训练 rollout64、每例至多8个监督位置，与正式 GSM8K 生成上限4096不是一回事。
3. [Full36 Q/O-LoRA Reader](experiments/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/)：
   冻结上述 Stage-A+B 和 Qwen 原参数，仅训36层 q_proj/o_proj 的 rank16、alpha16 LoRA，
   7,667,712参数；同数据/损失/1024更新。Native cache/teacher 禁用 Reader，
   translated cache 的后续读取启用 Reader；不改已经传输的 KV。
4. Reader 目录内补齐 [Llama Native](experiments/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/llama_native/)
   和 [HeteroFold baseline](experiments/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/heterofold_baseline/)。
   两个 follow-up 均已完成，脚本、配置、smoke、日志、逐样本及自动汇总随本次上传保留。
   HeteroFold 新跑 OBQA/MMLU-Pro/SciQ，ARC/HellaSwag/GSM8K 复用已核对同样本同协议的结果；没有重训映射。

截至本次交接，上述主线及两个补充评测都已完成，不需要为接手而重新启动。
这不表示所有历史实验都完成或当前所有服务器进程都空闲。

## 从哪些文件读结果

当前六条件、六数据集的完整机器汇总：

- [summary_with_llama_native_and_heterofold.json](experiments/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/summary_with_llama_native_and_heterofold.json)
- 条件：Qwen Native、旧 MCQ Stage-B、UFD Stage-B、UFD+Reader、Llama Native、HeteroFold。
- 数据量：OBQA128、ARC-Challenge128、HellaSwag64、GSM8K32、custom MMLU-Pro128、SciQ64。
- 包含各条件正确数/正确率，以及每对条件共同答对、各自独有答对和共同答错数量。

逐样本记录（包括相应生成/分数/正确性）：

- [Writer/Reader](experiments/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/evaluation/per_sample.jsonl)
- [Llama Native](experiments/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/llama_native/evaluation/per_sample.jsonl)
- [HeteroFold](experiments/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/heterofold_baseline/evaluation/per_sample.jsonl)

每个实验还有原始 `summary.json`、`config.json`、`manifest.json`、`logs/pipeline.log`，
原主实验汇总没有被新 baseline 覆盖。服务器共享自动汇总：
`/hy-tmp/yezhe/异构模型/EXPERIMENT_RESULTS.md`。
仓库根目录同名文件是较早的归档目录生成物，不保证包含之后全部新增 follow-up；
读取最新结果请优先使用上述具体实验机器文件，不要手工重写共享汇总。

## 关键运行时依赖与权重位置

以下均相对于服务器 B 的 `/hy-tmp/yezhe/异构模型/`，不在 GitHub 中：

| 用途 | 相对路径 |
| --- | --- |
| 当前 Stage-A | `mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/runs/stage_a/best.pt` |
| 旧 MCQ Stage-B | `mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/runs/stage_b/best.pt`（实际加载身份以 UFD manifest/config 为准） |
| UFD primary | `ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234/runs/stage_b/last.pt`，固定 epoch4/step1024 |
| Reader primary | `ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/runs/reader/epoch4_step1024.pt` |
| 论文 HeteroFold map | `heterofold_official_l3_to_q4_seed20260825/maps/llama3_to_qwen4/heterofold.pt` |

加载前应核对身份，已记录的 SHA256：

```text
Stage-A  db1f295c502f27305a9d2ac410c47a1bdef4db7e0255f817483867b19d78cf0d
UFD      25942a14d77a2a36173cccf6653d0e86cef01bdfdbd082836395faa895ba995e
Reader   a757ce6e4559168ee227ec53e380f16df9f6f6cd2c82b02c556802fdcf6f9599
HF map   dc2ea778b2b6ea4d6c74c678548946f59fb1e5970f935be3ecc5e4fb58ac718e
```

代码依赖包括 MCQ 父目录、原 HeteroFold 包、历史
`lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234` 配置，以及 UFD/Reader 的父子关系。
具体文件哈希、数据身份和依赖以 manifest、config 和代码中的 audit 为准；
不要只复制一个脚本后猜测缺失模块。
训练续跑还依赖服务器上的 aligned Sender states / prefix 等缓存；
缓存不随 GitHub 归档提供，缺失时需经授权重建，不能忽略身份检查。

## 科研解释边界

- HeteroFold **任务级推理/评分配置** 是当前默认；这不等于所有实验已经与官方 lm-eval 完全一致。
- GSM8K 使用 boxed 数学指令、各模型自己的 chat template、Qwen thinking=False、greedy4096及固定数字解析器。
- HellaSwag 是长度归一化续写 likelihood，不是生成选项字母；其他 MCQ 按各实验固定候选评分。
- 本轮 MMLU-Pro 是从 official test 构造的 **CUSTOM** question-grouped 划分，含512训练与64验证，
  测试128；不能报告为整份从未参与训练的官方 MMLU-Pro test，也不是官方5-shot CoT。
- Writer Full-Sync 使用既定 native anchor/tail、对齐位置及 post-k_norm/pre-RoPE K 约定；
  论文 HeteroFold arm 使用其 RAW Sender/capture/fold/cache 约定，不要把两套接口混用。
- HeteroFold 与我们训练预算/接口不同，是现有方法对照，不是仅改变一个结构的等预算消融。
- primary checkpoint 固定最后 step1024；best validation checkpoint 仅诊断，不能按 test 选模型。
- 对结果的进一步解读、改结构或追加训练属于后续请求，本交接不会自动启动。

## 工作习惯与上传

先核对样本/Prompt/权重/评分，保持 raw outputs；缺 baseline 时优先复用身份已验证结果。
启动用户请求的实验前检查 GPU/CUDA，成功后给日志命令，默认不持续监督。
上传时一个实验一个目录，补充评测嵌在该目录，直接带上脚本和轻量结果，
不反复迁移大权重、不额外手写长分析报告。提交成功后仍需核实远端 commit。
