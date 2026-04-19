# CCUS-Gym: Heterogeneous Multi-Agent Framework for CCUS Network Optimisation

CCUS-Gym is a PettingZoo-based research platform for coordinated CO2 capture,
transport, and storage under operational disruptions, quality constraints, and
economic trade-offs.

The project now contains three layers:

- **Research environment `CCUSEnv`**: physics simulation, disruption mechanisms,
  reward functions, and composition-aware CO2 quality modelling
- **MAPPO baseline**: minimal role-shared MAPPO for transport and storage agents
- **Hybrid LLM+MAPPO framework**: an LLM (e.g. Qwen3) controls strategic emitter
  decisions while MAPPO handles operational transport/storage decisions

## Core Design Rationale

Agent roles differ fundamentally in decision character, motivating a
**heterogeneous agent architecture**:

| Role | Decision type | Controller |
|------|--------------|------------|
| `emitter` | Strategic: how much to capture, which route, purification level | **LLM** (Qwen3) |
| `transport` | Operational: dispatch thresholds, destination preference | **MAPPO** |
| `storage` | Operational: injection fraction, quality target | **MAPPO** |

The LLM acts every 12 steps (once per simulated year) and caches its action in
between, decoupling inference latency from the environment step frequency.
MAPPO continues to update every step, preserving online learning.

## Preliminary Results

Minimal network, scenario `T` (transport disruptions), Qwen3-1.7B, first 10 episodes:

| Metric | Pure MAPPO (ep 0–9) | Hybrid LLM+MAPPO (ep 0–9) |
|--------|---------------------|---------------------------|
| Mean score | -18.09 | **-2.77** |
| CO2 stored | 11.16 Mt | **18.19 Mt (+63%)** |
| CO2 vented | 24.61 Mt | **17.71 Mt (−28%)** |
| Quality violations | 9.3 | **6.5** |

The LLM's out-of-the-box domain knowledge — identifying disruptions, comparing
transport costs, checking purity thresholds — delivers immediate benefit without
any training.

## Architecture

```text
Decision Layer  (sim/env.py)
    -> parses actions from both LLM and MAPPO agents
    -> builds observations
    -> computes rewards

Physical Layer  (core/physical.py)
    -> settles nominations
    -> simulates pipeline / ship / rail movement
    -> applies storage pressure dynamics
    -> tracks quality penalties and overflow

LLM Layer  (llm/)
    -> converts physical state to natural-language description
    -> calls LLM (local model or OpenAI-compatible API)
    -> parses JSON output into continuous action vector
    -> caches action for call_interval steps before refreshing

Training Layer  (rl/)
    -> hybrid_runner.py  — heterogeneous episode collection; only MAPPO roles updated
    -> mappo.py          — role-shared MAPPO trainer, checkpoint, logging helpers
```

## Repository Structure

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
├── llm/                         ← LLM decision module (new)
│   ├── __init__.py
│   ├── emitter_policy.py        ← HTTP API mode (OpenAI-compatible)
│   └── local_policy.py          ← Local model mode (HuggingFace transformers)
├── rl/
│   ├── training.py
│   ├── mappo.py
│   └── hybrid_runner.py         ← Hybrid training loop (new)
├── baselines/
│   └── rule_based.py
├── cli/
│   ├── train_mappo.py           ← Pure MAPPO training
│   ├── train_hybrid.py          ← LLM+MAPPO hybrid training (new)
│   ├── eval_mappo.py
│   ├── batch_mappo.py
│   └── eval_rule_based.py
├── viz/
│   └── episode_animation.py
scripts/
├── download_qwen3.sh            ← Download Qwen3 weights (new)
└── run_hybrid_slurm.sh          ← SLURM GPU job submission (new)
```

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `gymnasium`, `pettingzoo`, `pyyaml`, `torch`,
`matplotlib`, `tensorboard`

Additional dependencies for hybrid training:

```bash
pip install transformers>=4.51.0 accelerate>=1.0.0
```

## Quick Start: Environment Only

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

## Quick Start: Pure MAPPO (Baseline)

```bash
python -m ccus_gym.cli.train_mappo \
  --base minimal --scenario T --severity 0.3 \
  --episodes 200 --device cpu \
  --history-csv runs/mappo_T/history.csv \
  --plot runs/mappo_T/curves.png \
  --best-save runs/mappo_T/best.pt
```

## Quick Start: Hybrid LLM+MAPPO Training

### Step 1 — Download model weights (login node, once only)

```bash
# Qwen3-8B (~16 GB), recommended for experiments
HF_TOKEN=your_token bash scripts/download_qwen3.sh Qwen3-8B

# Qwen3-1.7B (~3.8 GB), for quick tests
HF_TOKEN=your_token bash scripts/download_qwen3.sh Qwen3-1.7B
```

Get a free Read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### Step 2 — Submit SLURM job (GPU node)

```bash
sbatch scripts/run_hybrid_slurm.sh
tail -f logs/hybrid_<jobid>.out
```

### Step 3 — Or run directly (if GPU is already available)

```bash
python -m ccus_gym.cli.train_hybrid \
  --llm-backend local \
  --llm-model /path/to/models/Qwen3-8B \
  --llm-call-interval 12 \
  --base minimal --scenario T --severity 0.5 \
  --episodes 50 --device cuda \
  --history-csv runs/hybrid_T/history.csv \
  --plot runs/hybrid_T/curves.png \
  --llm-log runs/hybrid_T/llm_reasoning.json \
  --best-save runs/hybrid_T/best.pt
```

Remote API (DashScope, vLLM, Ollama) is also supported:

```bash
python -m ccus_gym.cli.train_hybrid \
  --llm-backend api \
  --llm-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --llm-model qwen-plus --llm-api-key YOUR_KEY \
  --base minimal --scenario TS --episodes 50
```

### LLM reasoning log

`--llm-log` saves the full chain-of-thought for every LLM call, e.g.:

```json
{
  "emitter_0": [
    {
      "timestep": 0,
      "reasoning": "Pipeline available and cheaper ($8/t vs ship $18/t). Ship is disrupted. Buffer at 47% — send aggressively via pipeline.",
      "action": [0.9, 0.1, 0.85, 0.8, 0.2]
    }
  ]
}
```

This interpretability record is a key evidence source for domain-journal publications.

## Recommended Experiment Design

| Group | Entry point | Description |
|-------|-------------|-------------|
| Rule-based baseline | `eval_rule_based.py` | Economic heuristics, no learning |
| Pure MAPPO | `train_mappo.py` | Full RL, standard baseline |
| Hybrid LLM+MAPPO | `train_hybrid.py` | LLM emitter + MAPPO transport/storage |

Run each group across scenarios T / TS / TSG with multiple seeds, then compare
CO2 stored, operating cost, and quality violations.

## Environment Features

### Agents

Pipeline is passive infrastructure and is **not** an RL agent.

| Role | Typical decisions |
|------|-------------------|
| `emitter` | Route allocation, send fraction, capture fraction, purification effort |
| `transport_ship` / `transport_rail` | Dispatch threshold, destination bias, quality threshold |
| `storage_0`, `storage_1`, … | Injection fraction, quality target bias |

### CO2 Quality Modelling

Three capture-method families are supported: `post_combustion`, `pre_combustion`,
`oxy_fuel` — each with default purity, impurity composition, and cost parameters.
Purity can be raised at runtime through `purification_effort`. Storage sites
enforce acceptance constraints based on purity and impurity thresholds.

### Disruption System

Seven scenario families (`T` / `S` / `G` / `TS` / `TG` / `SG` / `TSG`) drawn
from three disruption domains:

- **Transport (T)**: pipeline failure, ship weather, rail conflict
- **Supply (S)**: equipment failure, production swing, planned maintenance
- **Geological (G)**: well failure, regulatory injection halt

### Rewards

```text
r_i = w_global × R_system + w_local × r_i_local
```

Shared terms: stored CO2, vented CO2, pressure violations, energy use, quality violations.
Role-local terms: per-agent cost, utilisation, pressure margin, overflow attribution.

## TensorBoard

```bash
tensorboard --logdir runs/hybrid_T/tb
```

Logged series: stored/vented/captured CO2, transport cost, quality violations,
training score, and per-role policy loss / value loss / entropy.

## Multi-Seed Batch Experiments

```bash
python -m ccus_gym.cli.batch_mappo \
  --base minimal --scenario T --severity 0.5 \
  --episodes 200 --seeds 11,12,13,14,15 \
  --device cpu --output-dir runs/batch_mappo_T
```

## Notes

- The optional storage proxy model in `storage_proxy.py` requires `scikit-learn`.
- The SLURM script targets the `tide` partition (H200 GPUs) with `--qos=long`.
- Hybrid training output format is identical to pure MAPPO (same CSV columns,
  same checkpoint format), enabling direct comparison.