# 上传实验到 GitHub（操作手册）

适用对象：把服务器 B 上已完成的 `experiments/<实验目录>` 归档进本仓库。

**不含**模型权重、优化器状态、KV 缓存、checkpoint、凭据。归档只负责"代码能读懂、结果能复算"。

相关约定见根目录 [AGENTS.md](../AGENTS.md) 的上传段落和 [EXPERIMENT_RESULTS_POLICY.md](../EXPERIMENT_RESULTS_POLICY.md)。

---

## 0. 前提：确认代理

本机到 `github.com` 的**直连不稳定**（整段收不到字节，不是单纯慢）。git 操作一律走本机代理。端口因机器而异，先确认：

```bash
lsof -iTCP -sTCP:LISTEN -P -n | grep -E "127\.0\.0\.1:(789[0-9]|108[0-7]|8080)"
export GIT_PROXY=http://127.0.0.1:7890
```

`git push` 的 HTTPS 认证来自已登录的 `gh`（`gh auth status` 显示 `Logged in to github.com account yezhe2021`）。
代理**只写在命令参数里**，不要写进 `~/.gitconfig` —— 换网络环境后会失效并污染全局配置。

---

## 1. 一次性：准备 partial clone

仓库工作区约 **640 MB、4.5 万文件**。不要全量 clone：

```bash
git -c http.proxy=$GIT_PROXY -c http.version=HTTP/1.1 \
    clone --depth 1 --filter=blob:none --sparse \
    https://github.com/yezhe2021/new-plan-with-gudingkuangjia.git upload_repo
```

`--filter=blob:none` 只拉提交和目录树（约 12 MB），文件内容按需取。
`--sparse` 让工作区只落地需要的路径。

---

## 2. 从服务器取文件

在服务器**源头就排除**缓存产物，不要指望事后清理：

```bash
ssh server-b 'cd "/hy-tmp/yezhe/异构模型/<父实验>" && \
  tar czf - --exclude="__pycache__" --exclude="*.pyc" --exclude="*.lock" \
  <新实验目录>' > exp.tgz
```

先验证包没被截断，再往下走：

```bash
tar tzf exp.tgz | wc -l                      # 条目数
mkdir -p /tmp/x && tar xzf exp.tgz -C /tmp/x
find /tmp/x -type f | wc -l
```

对照服务器上的实际文件数。tar 在 SSH 掉线时会静默产出不完整的包。

---

## 3. 放进 checkout

```bash
cd upload_repo
git -c http.proxy=$GIT_PROXY fetch --depth 1 origin main
git checkout -B main origin/main

git sparse-checkout set --no-cone \
  '/experiments/<父实验>/<新实验>/' \
  '/tools/'

mkdir -p experiments/<父实验>/<新实验>
cp -R /tmp/x/<新实验>/. experiments/<父实验>/<新实验>/
```

`--no-cone` 是必要的：cone 模式会把父目录下的所有文件也拉下来，
`recolor_kl_l3_to_q4_seed1234` 这种几十 MB 的父目录会白白多下载一遍。

---

## 4. 上传前检查（必做）

```bash
T=experiments/<父实验>/<新实验>

find $T \( -name '*.pt' -o -name '*.bin' -o -name '*.safetensors' \
        -o -name '*.npy' -o -name '*.npz' -o -name '*.ckpt' \) -print
find $T \( -name '*.lock' -o -name '__pycache__' -o -name '*.pyc' \) -print
```

两条命令的输出**都必须为空**。有输出就删掉再继续。

---

## 5. 暂存与防呆自检

```bash
git add $T

# 必须为空：本次提交只能是纯新增
git diff --cached --name-status | grep -v '^A'

git diff --cached --stat | tail -1
```

第二条是**最关键的一道闸**。sparse checkout 下如果路径或索引操作出错，
可能把仓库里其它文件当成删除（`D`）或修改（`M`）暂存进去。
一旦出现非 `A` 的行，立刻 `git reset` 重来，不要提交。

---

## 6. 提交

commit message 优先用实验自己生成的（写在 `summary.json` 里）：

```bash
python3 -c "import json; print(json.load(open('$T/summary.json'))['suggested_commit_message'])"
```

格式：**一行标题**（含实验名和判据结论）+ 空行 + **几行身体**（关键数字、排除了什么、为什么）。
判据结论（localized / distributed / ambiguous）必须出现在提交信息里，
否则半年后 `git log` 只能看到"加了结果"，看不到结论。

```bash
git -c user.name="yezhe2021" \
    -c user.email="yezhe2021@users.noreply.github.com" \
    commit -F msg.txt
```

---

## 7. 推送

```bash
git -c http.proxy=$GIT_PROXY -c http.version=HTTP/1.1 push origin main
```

期望输出：`<旧SHA>..<新SHA>  main -> main`

推送失败时**重试同一个 commit**：不要 `--amend`、不要新建提交。
按 AGENTS.md 的约定，瞬时故障重试同一个提交，不产生重复提交。

---

## 8. 验证远端

**不要用递归 tree API** —— 4.5 万文件的响应会直接超时。两点即可：

```bash
# (1) 远端分支指向的提交
curl -sS "https://api.github.com/repos/yezhe2021/new-plan-with-gudingkuangjia/git/refs/heads/main" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['object']['sha'])"

git rev-parse HEAD     # 两者必须相同
```

```bash
# (2) 关键文件的 blob SHA 逐字节比对
git rev-parse HEAD:$T/RESULTS.md

curl -sS "https://api.github.com/repos/yezhe2021/new-plan-with-gudingkuangjia/contents/$T/RESULTS.md" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['sha'], d['size'])"
```

SHA 相等即内容一字不差。挑 3–5 个关键文件（`RESULTS.md`、`config.json`、
`experiment.py`、`summary.json`、一个 `per_config` 样本）就够了。

---

## 9. 已知问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `Failed to connect to github.com port 443 after 75003 ms` | 直连被阻断；git 默认连接超时 75 秒，每次失败白等 | 加 `-c http.proxy=$GIT_PROXY` |
| `Error in the HTTP2 framing layer` | 代理链路下 HTTP/2 不稳 | 加 `-c http.version=HTTP/1.1` |
| `git ls-files` / `git status` 卡几分钟 | partial clone 索引有 4.5 万条，会触发按需取 blob | 避免全量命令；只操作目标路径 |
| API 请求挂 30–300 秒 | GitHub API 偶发不稳 | 请求加 3–5 次重试；一次只查一个文件 |
| 工作区里看不到 `experiments/` 其它目录 | `--no-cone` sparse 把路径外文件移出工作区 | 正常，索引里仍在，不影响提交 |
| tar 包明显小于预期 | SSH 中途断开，tar 静默截断 | 用第 2 步的条目数核对 |

---

## 10. 时间预期

| 步骤 | 耗时 |
|---|---|
| 首次 partial clone | 1–3 分钟 |
| 之后每次：fetch + 拷文件 + add + push（20 MB / 4000 文件量级） | **2–5 分钟** |
| 验证 | < 1 分钟 |

真正慢的从来不是推送（pack 通常只有几 MB），而是 clone、4.5 万条索引操作、
以及验证请求的重试。把这三件避开，一次上传就是几分钟的事。

---

## 附：一次完整的最小命令序列

```bash
export GIT_PROXY=http://127.0.0.1:7890
P='experiments/recolor_kl_l3_to_q4_seed1234'
N='prefix_native_sweep_acd_64_no_gsm_seed1234'
T="$P/$N"

ssh server-b "cd '/hy-tmp/yezhe/异构模型/$(dirname $P)' && \
  tar czf - --exclude='__pycache__' --exclude='*.pyc' --exclude='*.lock' $N" > exp.tgz

cd upload_repo
git -c http.proxy=$GIT_PROXY fetch --depth 1 origin main
git checkout -B main origin/main
git sparse-checkout set --no-cone "/$T/" '/tools/'
mkdir -p "$T" && tar xzf ../exp.tgz -C /tmp && cp -R /tmp/$N/. "$T"/

git add "$T"
git diff --cached --name-status | grep -v '^A'   # 必须为空
git -c user.name=yezhe2021 -c user.email=yezhe2021@users.noreply.github.com commit -F msg.txt
git -c http.proxy=$GIT_PROXY -c http.version=HTTP/1.1 push origin main

git rev-parse HEAD
curl -sS "https://api.github.com/repos/yezhe2021/new-plan-with-gudingkuangjia/git/refs/heads/main" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['object']['sha'])"
```
