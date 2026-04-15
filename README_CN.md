# CCUS-Gym

CCUS-Gym 是一个基于 PettingZoo 的多智能体强化学习环境，用于研究在
运行中断、CO2 质量约束和经济权衡条件下，CCUS（碳捕集-运输-封存）网络的协同调度问题。

目前这个项目同时包含两部分：

- 一个研究型环境 `CCUSEnv`：包含物理仿真、中断机制、奖励函数，以及面向 CO2
  纯度/组分的质量建模
- 一个最小可跑的 MAPPO 基线：包含训练、评估、checkpoint、TensorBoard、
  CSV/JSONL 日志、训练曲线和多随机种子批量实验

## 这个项目现在能解决什么问题

当前环境定位是 **固定网络结构下的运营优化**，不是做“管网怎么设计”的问题。

一个案例主要定义：

- 排放源（emitters）
- 可用运输方式（transport modes）
- 封存点（storage sites）
- 固定的 emitter -> transport -> storage 可行路线

在这个固定网络上，再让各类智能体学习如何调度。

### 环境层能力

- 基于 PettingZoo 的多智能体 `ParallelEnv`
- 三类智能体：`emitter`、`transport`、`storage`
- 连续动作空间
- 集中训练 / 分散执行（CTDE）风格奖励
- 运输、供给、地质三类随机中断
- 面向 CO2 纯度和组分的质量建模
- 不同捕集方法对应的默认纯度/杂质范围
- storage 侧基于纯度/杂质阈值的接收约束
- 经济上下文：碳税、电价、捕集补贴、封存信用、off-spec 惩罚
- 极端经济情景：如电价冲击、政策收紧

### 训练层能力

- 最小可跑的 role-shared MAPPO 基线
- 每种角色各一套 actor-critic
- 使用 Beta 分布处理 `[0, 1]` 连续动作
- centralized critic 使用固定维度的全局状态向量
- 支持确定性评估
- 支持按指定指标自动选择 best checkpoint
- 支持 JSONL / CSV / TensorBoard / PNG 日志
- 支持多随机种子批量实验

## 架构

环境内部采用“决策层”和“物理层”分离的结构：

```text
决策层 (env.py)
    -> 解析动作
    -> 构建观测
    -> 计算奖励

物理层 (physical.py)
    -> 提名结算
    -> 模拟管道 / 船运 / 铁路流动
    -> 更新封存压力
    -> 处理质量惩罚与溢出
```

训练器与环境并列存在：

```text
mappo.py
    -> role-shared MAPPO 训练器
    -> checkpoint 保存/加载
    -> 确定性评估
    -> CSV / JSONL / TensorBoard / 曲线图工具
```

## 文件说明

| 文件 | 作用 |
|------|------|
| `env.py` | 主环境 `CCUSEnv` |
| `physical.py` | 物理层仿真与月度结算逻辑 |
| `network.py` | 网络结构数据类和连通关系容器 |
| `quality.py` | CO2 纯度/组分默认值、混合计算、storage 质量检查 |
| `disruptions.py` | 随机中断生成器 |
| `configs.py` | 内置网络、场景和经济参数配置 |
| `case_loader.py` | YAML 案例加载器 |
| `training.py` | 训练配置和训练元数据工具 |
| `mappo.py` | 最小 MAPPO 实现及实验工具 |
| `train_mappo.py` | 训练命令行脚本 |
| `eval_mappo.py` | checkpoint 评估命令行脚本 |
| `batch_mappo.py` | 多随机种子批量实验脚本 |
| `storage_proxy.py` | 可选的 storage 压力 ML 代理模型 |
| `tools.py` | 物理层只读查询工具 |

## 智能体设计

管道（pipeline）是被动基础设施，**不是** RL 智能体。当前可训练的角色有：

| 角色 | 智能体 | 典型决策 |
|------|--------|----------|
| `emitter` | `emitter_0`, `emitter_1`, ... | 路由分配、发送比例、捕集比例、提纯强度 |
| `transport` | `transport_ship`, `transport_rail` | 调度阈值、目的地偏好、质量阈值、可选报价 |
| `storage` | `storage_0`, `storage_1`, ... | 注入比例、质量目标偏置 |

环境允许不同角色拥有不同观测和动作维度，而当前 MAPPO 基线则按“角色共享参数”的方式训练。

## CO2 质量与捕集方法

`quality.py` 中引入了一个轻量但明确的 CO2 质量建模层。

目前支持的默认捕集方法包括：

- `post_combustion`
- `pre_combustion`
- `oxy_fuel`

对于每个 emitter，配置或案例文件里可以指定：

- `capture_method`
- `base_purity`
- `composition`
- `capture_cost_per_t`
- `capture_energy_mwh_per_t`
- 提纯成本/能耗放大系数

运行过程中，环境可以：

- 通过 `purification_effort` 提高流股纯度
- 对多个来源流股在 storage 入口做混合
- 对不满足 storage 质量约束的流体施加惩罚或限制

因此现在这个环境不再只是“按体积调度 CO2”，而是更接近一个
**面向组分/纯度约束的 CCUS 协同调度研究平台**。

## 中断系统

当前支持 7 种中断场景族：

- `T`
- `S`
- `G`
- `TS`
- `TG`
- `SG`
- `TSG`

分别来自三类中断域：

- 运输中断：管道故障、船运天气、铁路冲突
- 供给中断：设备故障、产量波动、计划维护
- 地质中断：井故障、监管停注

除了物理中断之外，配置中还可以加入 `extreme_scenarios`，
在特定时间窗口内调整经济环境，例如：

- 电价上升
- 碳税提高

## 奖励设计

奖励采用分角色的 CTDE 结构：

```text
r_i = w_global * R_system + w_local * r_i_local
```

当前实现里已经包含：

- 全局共享项：封存量、排放量、超压违规、能耗、质量违规
- emitter 局部项：发送量、排放量、缓冲状态、运输成本、捕集成本、捕集能耗、纯度激励
- transport 局部项：交付量、利用率、拒运量、收入、质量相关行为
- storage 局部项：注入量、压力裕度、质量惩罚、注入义务、溢出归责

## 安装

在当前目录下执行：

```bash
python -m pip install -r requirements.txt
```

主要依赖包括：

- `numpy`
- `gymnasium`
- `pettingzoo`
- `pyyaml`
- `torch`
- `matplotlib`
- `tensorboard`
- `scikit-learn`（可选，但如果使用 storage proxy 模型就需要）

## 快速开始：只使用环境

下面的导入示例默认你是从 **包含 `ccus_gym/` 包目录的父目录** 运行 Python。

```python
from ccus_gym import CCUSEnv, make_config

config = make_config(
    base="minimal",
    scenario_family="T",
    severity=0.3,
    seed=1,
)

env = CCUSEnv(config)
obs, infos = env.reset()

for _ in range(env.episode_length):
    actions = {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }
    obs, rewards, terminations, truncations, infos = env.step(actions)
    if all(terminations.values()):
        break

print(env.get_episode_stats())
```

也可以从 YAML 案例创建环境：

```python
from ccus_gym import CCUSEnv

env = CCUSEnv.from_case("path/to/your_case.yaml", seed=123)
```

## 快速开始：最小 MAPPO 训练

先跑一个最小训练示例：

```bash
python train_mappo.py \
  --base minimal \
  --scenario T \
  --severity 0.3 \
  --episodes 10 \
  --device cpu
```

如果你希望同时产出实验日志、曲线和 checkpoint：

```bash
python train_mappo.py \
  --base minimal \
  --scenario T \
  --severity 0.3 \
  --episodes 20 \
  --device cpu \
  --history runs/demo/history.jsonl \
  --history-csv runs/demo/history.csv \
  --tensorboard-dir runs/demo/tb \
  --plot runs/demo/training.png \
  --best-save runs/demo/best.pt \
  --latest-save runs/demo/latest.pt \
  --best-metric score
```

当前支持的 best-checkpoint 指标有：

- `score`
- `total_stored`
- `total_vented`

默认的 `score` 是在 `mappo.py` 里定义的综合指标，大体上是：

- 奖励更多封存
- 惩罚更多排放
- 惩罚质量违规和压力违规

## 模型评估

可以对已保存的 checkpoint 做确定性评估：

```bash
python eval_mappo.py \
  --checkpoint runs/demo/best.pt \
  --base minimal \
  --scenario T \
  --severity 0.3 \
  --episodes 5 \
  --device cpu \
  --output runs/demo/eval.json
```

## TensorBoard

如果训练时使用了 `--tensorboard-dir runs/demo/tb`，可以用下面命令查看日志：

```bash
tensorboard --logdir runs/demo/tb
```

当前会记录的典型指标包括：

- 总封存量 / 总排放量 / 总捕集量
- 运输成本 / 捕集成本 / 能耗
- 压力违规 / 质量违规
- 综合训练分数 `score`
- 各角色的 policy loss / value loss / entropy

## 多随机种子批量实验

可以直接运行多随机种子实验并自动汇总：

```bash
python batch_mappo.py \
  --base minimal \
  --scenario T \
  --severity 0.3 \
  --episodes 20 \
  --eval-episodes 5 \
  --seeds 11,12,13 \
  --device cpu \
  --best-metric score \
  --output-dir runs/batch_t03
```

脚本会自动生成：

- 每个 seed 一个子目录
- `history.jsonl`
- `history.csv`
- `training.png`
- `tb/`
- `best.pt`
- `latest.pt`
- `eval.json`
- 顶层 `aggregate.csv`
- 顶层 `summary.json`

## 可编程接口

目前常用导出包括：

- `CCUSEnv`
- `make_config`
- `load_case`
- `DEFAULT_MAPPO_CONFIG`
- `train_mappo`
- `evaluate_policies`
- `save_checkpoint`
- `load_checkpoint`
- `plot_training_history`
- `write_tensorboard_history`
- `build_role_groups`
- `describe_training_setup`

## 当前范围

这个仓库现在已经是一个 **可用于研究和实验的原型系统**，但还不是生产级 MARL 训练平台。

需要注意的是：

- 环境层比训练器更丰富
- 当前 MAPPO 实现是有意保持最小可跑
- rollout 采样是单环境、同步式
- 还没有分布式训练或更复杂的经验管理机制

不过，它已经可以支持一条完整、可复现实验链路：

```text
environment -> training -> logging -> best checkpoint -> evaluation -> multi-seed summary
```

## 备注

- `storage_proxy.py` 中的可选代理模型功能仍然保留。
- 如果使用 proxy-based storage，就需要安装 `scikit-learn`。
- 英文版 README 与当前实现保持同步，中文 README 也已更新到同一能力范围。
