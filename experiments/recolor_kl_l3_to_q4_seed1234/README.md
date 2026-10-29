# Recolor-KL：固定 HeteroFold Stage-1 的功能蒸馏

五组：A Recolor-only；B官方目标线性R16；C官方目标线性R64；D Full-vocabulary KL线性R64；E Full KL GELU-MLP R64。固定同一份原始Recolor，完全保留 RAW Sender、token alignment、layer stencil与原生Receiver header的官方传输接口。禁止引入Grouped Stage-A、Full-Sync、Reader或旧Stage-B权重。

原校准集1600train/400validation保持原split，其中两种语料各800train/200validation。Recolor metadata确认仅在1600train上拟合，400验证未进入Stage-1拟合。所有新残差右因子从零开始；C/D的结构、Rank、初始因子相同，D/E仅加GELU。仅残差可训练，两个LLM与Recolor全部冻结。

统一4轮、batch8、800步、lr3e-4；其余优化器设置统一沿用官方默认wd1e-4/clip1。主结果固定第800步，验证最优只作诊断，避免不同损失的选点规则混入主结果。B是同预算重训的原始方法对照；原有历史折叠HeteroFold单独标记，不冒充同预算新B。

C/D不只改变loss：官方监督原生问题Query，本方案监督续写位置且后3轮混合原生/学生在线前缀。Full指完整词表，最多64续写token中选8个位置，不是全部位置。

所有组的残差都在pre-k_norm/pre-RoPE空间执行，再使用同一个官方finalize_layer做一次k_norm和RoPE。训练与部署都保留显式残差，不折叠，避免曾经发生的算子顺序/舍入混杂。官方attention公式不改；统一finalize带来的微小FP16算子顺序差异需要记录并过parity检查。模型FP16、Recolor/残差FP32，Recolor输出按官方ExampleBuilder转FP16。

先做A/C/D正确性门槛，再顺序训练C/D并完成400验证集交叉诊断，之后B/E，最后统一测试ARC128/HellaSwag64/GSM8K32/HotpotQA512。Hotpot使用现有固定distractor-dev512题，全上下文，严格排除与全部校准题目的ID和规范化问题重叠。测试仅一次固定最终模型，不用结果调参。

学生测试只预填充公开chat header；正文由RAW Llama经Recolor和本组残差得到，不能用原生正文KV。Native只作为训练/验证教师或独立原生基线。记录相同前缀完整词表KL、Attention KL、输出NMSE、自由生成EM/F1/数值准确率和截断率。过去评测过的测试题不能叫“从未看过”，但本轮不用于优化或选超参。

一次性后台流程，单卡串行，失败停止、保留断点，不持续监控或自动上传。之前的拼接实验仍暂停。

实现依据：
- https://github.com/daniel-eai/Prefill-Free-Multi-Agent-LLMs/blob/main/heterofold/calibration.py
- https://github.com/daniel-eai/Prefill-Free-Multi-Agent-LLMs/blob/main/heterofold/transfer.py

reference/保存当前服务器官方实现与原校准数据的只读快照；主实验不修改原仓库。
