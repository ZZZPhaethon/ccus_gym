# CCUS-Gym: multi-agent reinforcement learning environment for CCUS

![Teaser image](assets/project_overview.png)

CCUS-Gym is a PettingZoo-based multi-agent reinforcement learning environment for
coordinated CO2 capture, transport, and storage under operational disruptions,
quality constraints, and economic trade-offs.

The project now includes both:

- a research environment (`CCUSEnv`) with physics, disruptions, rewards, and
  composition-aware CO2 quality modelling
- a minimal end-to-end MAPPO baseline with training, evaluation, checkpointing,
  TensorBoard logging, CSV/JSONL history export, curve plotting, and multi-seed
  batch runs

## What This Project Can Model

The current environment is designed for **fixed-network operational optimisation**
rather than pipeline network design. A case defines:

- emitters
- available transport modes
- storage sites
- fixed emitter-to-transport-to-storage route options

Each agent then learns how to operate within that fixed network.

### Environment features

- Multi-agent PettingZoo `ParallelEnv`
- Three agent roles: `emitter`, `transport`, `storage`
- Continuous action spaces
- Centralized-training / decentralized-execution reward structure
- Stochastic disruptions across transport, supply, and geological domains
- Composition-aware CO2 stream quality
- Capture-method-specific purity and impurity defaults
- Storage acceptance constraints based on purity / impurity limits
- Economic context including carbon tax, electricity price, capture subsidy,
  storage credit, and off-spec penalties
- Extreme economic scenario hooks such as electricity-price spikes and policy tightening

### Training features

- Minimal role-shared MAPPO baseline
- One actor-critic pair per role
- Beta-distribution policy for `[0, 1]` continuous actions
- Centralized critic over a fixed-dimension global state vector
- Deterministic checkpoint evaluation
- Best-checkpoint selection by configurable metric
- JSONL / CSV / TensorBoard / PNG logging
- Multi-seed batch experiment runner

## Architecture

The environment keeps decision logic separate from physics:

```text
Decision Layer (sim/env.py)
    -> parses actions
    -> builds observations
    -> computes rewards

Physical Layer (core/physical.py)
    -> settles nominations
    -> simulates pipeline / ship / rail movement
    -> applies storage pressure dynamics
    -> tracks quality penalties and overflow
```

The minimal training stack sits beside the environment:

```text
rl/mappo.py
    -> role-shared MAPPO trainer
    -> checkpoint save/load
    -> deterministic evaluation
    -> CSV / JSONL / TensorBoard / plot helpers
```

## Repository Guide

The codebase is now grouped by function:

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
├── rl/
│   ├── training.py
│   └── mappo.py
├── cli/
│   ├── train_mappo.py
│   ├── eval_mappo.py
│   └── batch_mappo.py
├── __init__.py
├── README.md
├── README_CN.md
└── requirements.txt
```

## Agent Design

Pipeline is passive infrastructure and is **not** an RL agent. The trainable
roles are:

| Role | Agents | Typical decisions |
|------|--------|-------------------|
| `emitter` | `emitter_0`, `emitter_1`, ... | route allocation, send fraction, capture fraction, purification effort |
| `transport` | `transport_ship`, `transport_rail` | dispatch threshold, destination bias, quality threshold, optional price |
| `storage` | `storage_0`, `storage_1`, ... | injection fraction, quality target bias |

The environment uses heterogeneous observations and actions, but the MAPPO
baseline shares parameters by role.

## CO2 Quality and Capture Methods

`quality.py` introduces lightweight but explicit quality modelling.

Supported default capture-method families:

- `post_combustion`
- `pre_combustion`
- `oxy_fuel`

For each emitter, the config/case can specify:

- `capture_method`
- `base_purity`
- `composition`
- `capture_cost_per_t`
- `capture_energy_mwh_per_t`
- purification cost/energy multipliers

At runtime the environment can:

- raise purity through `purification_effort`
- blend multiple incoming streams at storage
- penalize or restrict flows that violate storage quality limits

This makes the environment closer to a research testbed for
**composition-aware CCUS coordination**, not just volume routing.

## Disruptions

Seven disruption families are supported:

- `T`
- `S`
- `G`
- `TS`
- `TG`
- `SG`
- `TSG`

They combine:

- transport disruptions: pipeline failure, ship weather, rail conflict
- supply disruptions: equipment failure, production swing, maintenance
- geological disruptions: well failure, regulatory stop

In addition to physical disruptions, configs can define `extreme_scenarios`
that modify economic conditions such as electricity price and carbon tax over
selected time windows.

## Rewards

Rewards follow a role-factored CTDE pattern:

```text
r_i = w_global * R_system + w_local * r_i_local
```

The current reward implementation includes:

- shared terms for stored CO2, vented CO2, pressure violations, energy use,
  and quality violations
- emitter-local terms for sent volume, venting, buffer state, transport cost,
  capture cost, capture energy, and purity incentive
- transport-local terms for delivered volume, utilization, rejected volume,
  revenue, and quality-sensitive behavior
- storage-local terms for injected volume, pressure margin, quality penalties,
  injection obligations, and overflow attribution

## Installation

From this directory:

```bash
python -m pip install -r requirements.txt
```

Main dependencies:

- `numpy`
- `gymnasium`
- `pettingzoo`
- `pyyaml`
- `torch`
- `matplotlib`
- `tensorboard`
- `scikit-learn` (optional but needed for proxy-model storage pressure)

## Quick Start: Environment Only

The import examples below assume you run Python from the **parent directory**
that contains the `ccus_gym/` package folder.

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

You can also build from a YAML case:

```python
from ccus_gym import CCUSEnv

env = CCUSEnv.from_case("path/to/your_case.yaml", seed=123)
```

## Quick Start: Minimal MAPPO Training

Run training from the parent directory that contains the `ccus_gym/` package:

```bash
python -m ccus_gym.cli.train_mappo \
  --base minimal \
  --scenario T \
  --severity 0.3 \
  --episodes 10 \
  --device cpu
```

Train with experiment outputs:

```bash
python -m ccus_gym.cli.train_mappo \
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

Supported best-checkpoint metrics:

- `score`
- `total_stored`
- `total_vented`

The default `score` is a composite metric defined in `rl/mappo.py` that rewards
stored CO2 and penalizes venting plus violations.

## Evaluation

Evaluate a saved checkpoint deterministically:

```bash
python -m ccus_gym.cli.eval_mappo \
  --checkpoint runs/demo/best.pt \
  --base minimal \
  --scenario T \
  --severity 0.3 \
  --episodes 5 \
  --device cpu \
  --output runs/demo/eval.json
```

## TensorBoard

If you trained with `--tensorboard-dir runs/demo/tb`, you can inspect logs with:

```bash
tensorboard --logdir runs/demo/tb
```

Typical logged series include:

- total stored / vented / captured
- transport cost / capture cost / energy use
- pressure and quality violations
- composite training score
- policy/value losses and entropy for each role

## Multi-Seed Batch Experiments

Run multiple seeds and aggregate results:

```bash
python -m ccus_gym.cli.batch_mappo \
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

This creates:

- one subdirectory per seed
- `history.jsonl`
- `history.csv`
- `training.png`
- `tb/`
- `best.pt`
- `latest.pt`
- `eval.json`
- top-level `aggregate.csv`
- top-level `summary.json`

## Programmatic API

Useful exports from `ccus_gym` include:

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

## Current Scope

This repository now gives you a **research-capable prototype**, not a
production-grade MARL stack. In particular:

- the environment is much richer than the trainer
- the MAPPO implementation is intentionally minimal
- rollout collection is single-environment and synchronous
- no distributed training or replay infrastructure is included

That said, the project already supports a full reproducible loop:

```text
environment -> training -> logging -> best checkpoint -> evaluation -> multi-seed summary
```

## Notes

- The optional storage proxy model is still supported through `storage_proxy.py`.
- If you use proxy-based storage sites, `scikit-learn` must be installed.
- The English README reflects the current implemented feature set more closely
  than the older project summary.
