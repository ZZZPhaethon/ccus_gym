# CCUS-Gym

CCUS-Gym 是一个基于 PettingZoo 的**异构多智能体**研究平台，用于研究在运行中断、CO2 质量约束和经济权衡条件下，CCUS（碳捕集-运输-封存）网络的协同调度问题。

![Overview](assets/project_overview.png)

项目目前包含三个层次：

- **研究型环境 `CCUSEnv`**：物理仿真、中断机制、奖励函数、CO2 纯度/组分质量建模
- **MAPPO 基线**：用于 transport 和 storage 智能体的最小可跑 role-shared MAPPO
- **混合 LLM+MAPPO 框架**：LLM（如 Qwen3）控制战略层 emitter 决策，MAPPO 控制运营层 transport/storage 决策

## 核心设计思路

不同角色的决策性质差异显著，因此采用**异构智能体架构**：

| 角色 | 决策类型 | 控制方式 |
|------|----------|----------|
| `emitter` | 战略层：捕集多少、走哪条路线、提纯强度 | **LLM**（Qwen3） |
| `transport` | 运营层：调度阈值、目的地偏好、质量阈值 | **MAPPO** |
| `storage` | 运营层：注入比例、质量目标偏置 | **MAPPO** |

LLM 每 12 步（每年）决策一次并缓存动作，解决了推理延迟与环境步频不匹配的问题。MAPPO 每步更新，保留在线学习能力。

## 初步实验结果

`minimal` 网络、`T` 场景（运输中断）、seed 42，三组对比：

| 指标 | 纯 MAPPO（前50轮）| 纯 MAPPO（2000轮）| **Hybrid Qwen3-8B（50轮）** |
|------|------------------|-------------------|------------------------------|
| 平均评分 | -16.60 | -18.51 | **+11.62** |
| 最高评分 | -13.47 | -8.22 | **+14.06** |
| 平均封存量 | 10.27 Mt | 8.88 Mt | **23.86 Mt (+132%)** |
| 平均排放量 | 25.49 Mt | 27.01 Mt | **12.09 Mt (−53%)** |
| 质量违规 | 2.7 次 | 0.8 次 | **0.3 次** |
| 压力违规 | 0 | 0 | **0** |

核心结论：**纯 MAPPO 跑满 2000 轮的最好成绩（-8.22）仍远低于 Hybrid 50 轮的平均水平（+11.62）**。
LLM 的开箱即用领域知识（识别中断、比较运输成本、判断纯度是否达标）完全弥补了训练量的差距。

## 架构

```text
决策层 (sim/env.py)
    -> 解析动作（支持 LLM 和 MAPPO 混合输入）
    -> 构建观测
    -> 计算奖励

物理层 (core/physical.py)
    -> 提名结算
    -> 模拟管道 / 船运 / 铁路流动
    -> 更新封存压力
    -> 处理质量惩罚与溢出

LLM 层 (llm/)
    -> 将物理状态转为自然语言描述
    -> 调用 LLM 推理（支持本地加载 / OpenAI 兼容 API）
    -> 解析 JSON 输出为连续动作向量
    -> 缓存动作，每 call_interval 步刷新一次

训练层 (rl/)
    -> hybrid_runner.py：异构 episode 采集 + 仅更新 MAPPO 角色
    -> mappo.py：role-shared MAPPO 训练器
```

## 文件结构

```text
ccus_gym/
├── core/
│   ├── network.py
│   ├── physical.py
│   ├── quality.py
│   ├── storage_proxy.py
│   └── tools.py
├── sim/
│   ├── env.py
│   ├── disruptions.py
│   ├── configs.py
│   └── case_loader.py
├── llm/                        ← LLM 决策模块（新增）
│   ├── __init__.py
│   ├── emitter_policy.py       ← HTTP API 模式（OpenAI 兼容）
│   └── local_policy.py         ← 本地模型模式（HuggingFace transformers）
├── rl/
│   ├── training.py
│   ├── mappo.py
│   └── hybrid_runner.py        ← 混合训练循环（新增）
├── baselines/
│   └── rule_based.py
├── cli/
│   ├── train_mappo.py          ← 纯 MAPPO 训练
│   ├── train_hybrid.py         ← LLM+MAPPO 混合训练（新增）
│   ├── eval_mappo.py
│   ├── batch_mappo.py
│   └── eval_rule_based.py
├── viz/
│   └── episode_animation.py
scripts/
├── download_qwen3.sh           ← 下载 Qwen3 模型权重（新增）
└── run_hybrid_slurm.sh         ← SLURM GPU 任务提交脚本（新增）
```

## 安装

```bash
pip install -r requirements.txt
```

主要依赖：`numpy`, `gymnasium`, `pettingzoo`, `pyyaml`, `torch`, `matplotlib`, `tensorboard`

混合训练额外需要：
```bash
pip install transformers>=4.51.0 accelerate>=1.0.0
```

## 快速开始：只使用环境

```python
from ccus_gym import CCUSEnv, make_config

config = make_config(base="minimal", scenario_family="T", severity=0.3, seed=1)
env = CCUSEnv(config)
obs, infos = env.reset()

for _ in range(env.episode_length):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    if all(terminations.values()):
        break

print(env.get_episode_stats())
```

## 快速开始：纯 MAPPO 训练（baseline）

```bash
python -m ccus_gym.cli.train_mappo \
  --base minimal --scenario T --severity 0.3 \
  --episodes 200 --device cpu \
  --history-csv runs/mappo_T/history.csv \
  --plot runs/mappo_T/curves.png \
  --best-save runs/mappo_T/best.pt
```

## 快速开始：混合 LLM+MAPPO 训练

### 第一步：下载模型（登录节点执行，一次性）

```bash
# Qwen3-8B 约 16GB，推荐用于正式实验
HF_TOKEN=你的token bash scripts/download_qwen3.sh Qwen3-8B

# Qwen3-1.7B 约 3.8GB，用于快速测试
HF_TOKEN=你的token bash scripts/download_qwen3.sh Qwen3-1.7B
```

HuggingFace token 申请地址：[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)（Read 权限即可）

### 第二步：提交 SLURM 任务（GPU 节点）

```bash
sbatch scripts/run_hybrid_slurm.sh
```

查看进度：
```bash
tail -f logs/hybrid_<jobid>.out
```

### 第三步：直接调用（已有 GPU 环境）

```bash
python -m ccus_gym.cli.train_hybrid \
  --llm-backend local \
  --llm-model /scratch_tide/yc5224/models/Qwen3-8B \
  --llm-call-interval 12 \
  --base minimal --scenario T --severity 0.5 \
  --episodes 50 --device cuda \
  --history-csv runs/hybrid_T/history.csv \
  --plot runs/hybrid_T/curves.png \
  --llm-log runs/hybrid_T/llm_reasoning.json \
  --best-save runs/hybrid_T/best.pt
```

也支持 OpenAI 兼容 API（DashScope、vLLM 等）：

```bash
python -m ccus_gym.cli.train_hybrid \
  --llm-backend api \
  --llm-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --llm-model qwen-plus \
  --llm-api-key 你的key \
  --base minimal --scenario TS --episodes 50
```

### LLM 推理日志

`--llm-log` 会保存每次 LLM 决策的完整推理文本，例如：

```json
{
  "emitter_0": [
    {
      "timestep": 0,
      "reasoning": "Pipeline is available and cost-effective ($8/t vs ship $18/t). Ship is disrupted. Buffer at 47%, send aggressively via pipeline.",
      "action": [0.9, 0.1, 0.85, 0.8, 0.2]
    }
  ]
}
```

这是能源领域论文中"可解释性"的核心证据。

## 实验对比设计

以下是推荐的完整对比实验组：

| 组别 | 命令入口 | 说明 |
|------|----------|------|
| 规则基线 | `eval_rule_based.py` | 经济规则，无学习 |
| 纯 MAPPO | `train_mappo.py` | 全 RL，标准 baseline |
| Hybrid LLM+MAPPO | `train_hybrid.py` | LLM emitter + MAPPO transport/storage |

建议在 T / TS / TSG 三种中断场景下各跑多个 seed，横向对比封存量、成本和质量违规指标。

## 环境能力

### 智能体

管道（pipeline）是被动基础设施，**不是** RL 智能体。可训练角色：

| 角色 | 典型决策 |
|------|----------|
| `emitter` | 路由分配、发送比例、捕集比例、提纯强度 |
| `transport_ship` / `transport_rail` | 调度阈值、目的地偏好、质量阈值 |
| `storage_0`, `storage_1`, ... | 注入比例、质量目标偏置 |

### CO2 质量建模

支持三种捕集方法：`post_combustion`、`pre_combustion`、`oxy_fuel`，各有默认纯度、杂质组成和成本参数。运行中可通过 `purification_effort` 动态提升纯度，storage 侧按纯度/杂质阈值进行接收约束。

### 中断系统

支持 7 种场景族（`T` / `S` / `G` / `TS` / `TG` / `SG` / `TSG`），来自三类中断域：

- **运输中断 (T)**：管道故障、船运天气、铁路冲突
- **供给中断 (S)**：设备故障、产量波动、计划维护
- **地质中断 (G)**：井故障、监管停注

### 奖励设计

```text
r_i = w_global × R_system + w_local × r_i_local
```

全局项：封存量、排放量、超压违规、能耗、质量违规
局部项：各角色专属的成本、利用率、压力裕度等信号

## TensorBoard

```bash
tensorboard --logdir runs/hybrid_T/tb
```

记录指标包括：封存量、排放量、运输成本、质量违规、各角色 policy/value loss 和 entropy。

## 多随机种子批量实验

```bash
python -m ccus_gym.cli.batch_mappo \
  --base minimal --scenario T --severity 0.5 \
  --episodes 200 --seeds 11,12,13,14,15 \
  --device cpu --output-dir runs/batch_mappo_T
```

## 备注

- `storage_proxy.py` 中的代理模型功能保留，如使用需安装 `scikit-learn`
- SLURM 脚本默认申请 `tide` 分区（H200 GPU），QOS 为 `long`
- 混合训练输出格式与纯 MAPPO 完全一致（相同 CSV 列、相同 checkpoint 格式），可直接对比
