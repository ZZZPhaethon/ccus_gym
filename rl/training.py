"""Training-oriented helpers for CCUS-Gym.

The project does not ship a full MARL trainer, but these helpers expose the
environment metadata typically needed by a MAPPO-style training pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ccus_gym.sim.env import CCUSEnv


DEFAULT_MAPPO_CONFIG: Dict[str, Any] = {
    "algorithm": "MAPPO",
    "shared_policy_by_role": True,
    "hidden_dim": 128,
    "rollout_length": 120,
    "ppo_epochs": 10,
    "mini_batches": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "learning_rate": 3e-4,
    "best_metric": "score",
    "score_vented_weight": 1.0,
    "score_quality_weight": 0.5,
    "score_pressure_weight": 2.0,
}


def build_role_groups(env: CCUSEnv) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {
        "emitter": [],
        "transport": [],
        "storage": [],
    }
    for agent in env.possible_agents:
        if agent.startswith("emitter_"):
            groups["emitter"].append(agent)
        elif agent.startswith("transport_"):
            groups["transport"].append(agent)
        elif agent.startswith("storage_"):
            groups["storage"].append(agent)
    return groups


def describe_training_setup(env: CCUSEnv) -> Dict[str, Any]:
    groups = build_role_groups(env)
    obs_dims = {
        agent: int(np.prod(env.observation_space(agent).shape))
        for agent in env.possible_agents
    }
    act_dims = {
        agent: int(np.prod(env.action_space(agent).shape))
        for agent in env.possible_agents
    }
    role_specs = {}
    for role, agents in groups.items():
        if not agents:
            continue
        role_specs[role] = {
            "agents": agents,
            "observation_dim": obs_dims[agents[0]],
            "action_dim": act_dims[agents[0]],
            "parameter_sharing": True,
        }
    return {
        "default_algorithm": DEFAULT_MAPPO_CONFIG["algorithm"],
        "mappo_defaults": dict(DEFAULT_MAPPO_CONFIG),
        "role_groups": groups,
        "role_specs": role_specs,
        "global_state_dim": int(env.global_state_vector().shape[0]),
    }


def make_env_and_training_spec(config: Dict[str, Any]) -> Dict[str, Any]:
    env = CCUSEnv(config)
    spec = describe_training_setup(env)
    return {
        "env": env,
        "training_spec": spec,
    }
