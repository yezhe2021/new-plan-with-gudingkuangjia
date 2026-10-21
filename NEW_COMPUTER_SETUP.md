# 新电脑接手：从这里开始

更新时间：2026-10-21。本项目采用「本地 GitHub 代码与结果 + 远程 GPU 服务器」的方式工作。
换电脑不需要重新训练，也不需要在新电脑安装 CUDA 或下载全部模型。

## 1. 获取仓库

安装 Git 和 OpenSSH 客户端，然后在你选定的工作目录执行：

```bash
git clone https://github.com/yezhe2021/new-plan-with-gudingkuangjia.git
cd new-plan-with-gudingkuangjia
git rev-parse HEAD
```

普通完整 clone 即可，不需要复用旧电脑的 `.hf_repro_upload` 或 sparse-checkout。
如果已有 checkout，先检查 `git status`；没有本地改动时可使用 `git pull --ff-only`。
不要为了更新而强制清除本地改动。查看文档和已有结果无需 Python/GPU。

在新对话中打开此仓库作为工作目录，把下面这句话发给接手助手即可：

> 请先完整阅读根目录 AGENTS.md、NEW_COMPUTER_SETUP.md、PROJECT_HANDOFF.md、
> EXPERIMENT_PROTOCOL_DEFAULTS.md 和 EXPERIMENT_RESULTS_POLICY.md。
> 默认使用服务器 B。先核对现有结果，不要自动重训；之后按我本次请求继续。

旧会话记忆、本地 skills、自动化任务和认证信息不会因为 clone 自动迁移；
本仓库的交接文档记录了继续研究所需的项目约定。

## 2. 单独配置 SSH 认证

将 [ssh_config.example](tools/ssh_config.example) 中需要的 Host 配置合并到
新电脑的 SSH config，**不要覆盖已有配置**：

- Windows：用户目录下 `.ssh/config`。
- Linux/macOS：`~/.ssh/config`。

示例不含密码或私钥。新电脑仍需配置你有权限使用的认证方式，例如创建新密钥，
由你或服务器管理员把公钥加入服务器。不要把密码、私钥或 Token 提交到 GitHub。
首次连接时，向可信来源核对主机指纹；不要关闭主机密钥校验。

```bash
ssh -o ConnectTimeout=20 server-b
```

B 当前地址为 `root@i-2.gpushare.com:36738`，A 为 `root@172.23.219.50:10044`。
端口/IP 若发生变化，只更新本机 SSH 配置并先确认目标服务器身份。
旧电脑使用过的 `127.0.0.1:7892` Git 代理是本机配置，不应直接搬到新电脑。

## 3. 只读检查服务器 B

下面的命令不会启动训练、占用 GPU 显存或修改远端实验：

```bash
ssh server-b "hostname; nvidia-smi; df -h /hy-tmp"
ssh server-b "/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python --version"
ssh server-b "ls -ld /hy-tmp/yezhe/异构模型 /hy-tmp/yezhe/all_models/models /hy-tmp/yezhe/all_models/datasets"
```

当前主要环境（2026-10-21 只读核实）：

| 项目 | 服务器 B |
| --- | --- |
| Python | `/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python`，3.10.20 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 5.9.0 |
| lm-evaluation-harness | 0.4.12（历史官方任务实验；不是所有新实验的评分器） |
| NumPy / safetensors / accelerate | 2.2.6 / 0.7.0 / 1.13.0 |
| 实验目录 | `/hy-tmp/yezhe/异构模型` |
| 数据集 | `/hy-tmp/yezhe/all_models/datasets` |
| Llama | `/hy-tmp/yezhe/all_models/models/LLM-Research/Llama-3.2-3B-Instruct` |
| Qwen | `/hy-tmp/yezhe/all_models/models/Qwen/Qwen3-4B` |

这是已存在环境的记录，不是建议在服务器上无条件升级/重装依赖。
更多模型与任务的精确路径、版本和输入哈希以各实验 `config.json` / `manifest.json` 为准。

服务器 A 仅作历史/备用：实验 `/home/yezhe/异构模型`，Python
`/home/yezhe/data/miniconda3/envs/attnkv/bin/python`。不要默认把 B 上的任务搬到 A。

SSH 偶发超时、拒绝或 banner exchange 失败不代表训练退出。
先做有限次数连接重试，恢复后查看日志/进程，不要重复启动流水线。

## 4. 看当前已完成结果

先读 [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)，其中给出了最新自动汇总、逐样本输出、
脚本及 checkpoint 路径。已归档的结果可以直接在 GitHub 或本地阅读，不必连接服务器。

服务器 B 上最新 Reader 日志查看命令：

```bash
ssh server-b "tail -n 60 /hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/logs/pipeline.log"
```

这轮已完成；看历史日志不要误认为还在运行。其他任务的实时状态需另查，
本交接不保证整台服务器空闲。

## 5. GitHub 能迁移什么，不能迁移什么

| 内容 | 位置/处理方式 |
| --- | --- |
| 实验代码、配置、轻量日志、机器汇总、逐样本预测 | 本仓库，直接 clone |
| 模型、Writer/Reader 权重、优化器状态、KV 缓存 | 仍在服务器，不在 GitHub |
| 数据集原始文件及运行时准备的完整训练数据 | 服务器；仓库保留实验划分/身份记录，具体看 manifest |
| SSH/GitHub/Hugging Face 认证 | 新电脑单独设置，绝不写入仓库 |

**新电脑继续使用同一服务器 B**：clone + SSH 认证后即可读取原环境继续工作。
**改用全新 GPU 服务器**：仅 clone 不够，还需另外迁移权重、模型、数据及代码依赖，
核对哈希和路径并做 smoke/parity 检查。这不是本次已完成的服务器备份。

部分脚本使用服务器绝对路径及历史实验代码依赖。不要在 Windows 本地直接运行它们，
也不要把路径机械替换成新电脑路径。部署任何修改前先核实远端是否有未归档改动，
避免用旧归档覆盖在用的代码。
