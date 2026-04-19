"""Hybrid episode runner: LLM controls emitters, MAPPO controls transport + storage.

The emitter LLM policy is called every `call_interval` timesteps (default: 12 months)
and caches its action in between. MAPPO policies for transport and storage are updated
normally after each episode via PPO.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np

from ccus_gym.sim.env import CCUSEnv
from ccus_gym.llm.emitter_policy import LLMEmitterPolicy
from ccus_gym.rl.mappo import (
    AgentTrajectory,
    RoleMAPPOPolicy,
    Transition,
    _build_batch,
    _role_from_agent,
    _set_seed,
    build_role_policies,
    save_checkpoint,
    score_episode,
    selection_metric_value,
)
from ccus_gym.rl.training import DEFAULT_MAPPO_CONFIG, build_role_groups


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _get_transport_cost(env: CCUSEnv, mode_name: str) -> float:
    pl = env.physical_layer
    if mode_name == "pipeline" and pl.pipeline is not None:
        return float(pl.pipeline.base_cost)
    if mode_name == "ship" and pl.ships is not None:
        return float(pl.ships.base_cost)
    if mode_name == "rail" and pl.rail is not None:
        return float(pl.rail.cost_per_t_km * pl.rail.distance_km)
    return 0.0


def build_emitter_context(env: CCUSEnv, agent: str) -> Dict[str, Any]:
    """Build a human-readable context dict for the LLM emitter policy."""
    eid = env._agent_comp_id[agent]
    pl = env.physical_layer

    # Emitter state
    emitter_state = pl.emitters[eid].get_state()

    # Routes and connected transports
    routes = pl.get_routes_for_emitter(eid)
    transport_ids = pl.get_connected_transport_ids(eid)
    storage_ids = pl.get_connected_storage_ids(eid)

    # De-duplicate transport modes while preserving route order
    seen: set = set()
    route_modes: List[str] = []
    for tid, _ in routes:
        mode = pl.get_mode_name(tid)
        if mode not in seen:
            seen.add(mode)
            route_modes.append(mode)

    # Transport states (one per unique mode that appears in routes)
    transport_states: List[Dict[str, Any]] = []
    for mode in route_modes:
        transport_states.append(pl.get_transport_state(mode))

    # Approximate cost per mode
    transport_costs: Dict[str, float] = {
        mode: _get_transport_cost(env, mode) for mode in route_modes
    }

    # Storage states
    storage_states: Dict[int, Dict[str, Any]] = {
        sid: pl.storage_sites[sid].get_state() for sid in storage_ids
    }

    return {
        "agent_id": agent,
        "emitter_id": eid,
        "n_routes": len(routes),
        "route_modes": route_modes,
        "emitter_state": emitter_state,
        "transport_states": transport_states,
        "transport_costs": transport_costs,
        "storage_states": storage_states,
        "economic": env.get_economic_context(),
        "timestep": env.timestep,
        "episode_length": env.episode_length,
    }


# ---------------------------------------------------------------------------
# Hybrid episode collection
# ---------------------------------------------------------------------------

def collect_hybrid_episode(
    env: CCUSEnv,
    llm_policies: Dict[str, LLMEmitterPolicy],
    mappo_policies: Dict[str, RoleMAPPOPolicy],
    *,
    seed: Optional[int] = None,
    deterministic: bool = False,
) -> tuple[Dict[str, Dict[str, AgentTrajectory]], Dict[str, Any]]:
    """Run one full episode with LLM emitters + MAPPO transport/storage.

    Only transport and storage trajectories are returned for MAPPO updates.
    Emitter trajectories are collected but not used for gradient updates.
    """
    obs, _ = env.reset(seed=seed)

    role_trajectories: Dict[str, Dict[str, AgentTrajectory]] = {
        role: {agent: AgentTrajectory(transitions=[]) for agent in agents}
        for role, agents in build_role_groups(env).items()
    }

    done = False
    while not done:
        state = env.global_state_vector()
        actions: Dict[str, np.ndarray] = {}
        cached: Dict[str, Transition] = {}

        for agent in env.agents:
            role = _role_from_agent(agent)

            if role == "emitter" and agent in llm_policies:
                ctx = build_emitter_context(env, agent)
                action = llm_policies[agent].act(ctx)
                actions[agent] = action
                cached[agent] = Transition(
                    obs=np.asarray(obs[agent], dtype=np.float32),
                    state=np.asarray(state, dtype=np.float32),
                    action=np.asarray(action, dtype=np.float32),
                    log_prob=0.0,
                    reward=0.0,
                    done=False,
                    value=0.0,
                )
            else:
                action, log_prob, value = mappo_policies[role].act(
                    obs[agent], state, deterministic=deterministic
                )
                actions[agent] = action
                cached[agent] = Transition(
                    obs=np.asarray(obs[agent], dtype=np.float32),
                    state=np.asarray(state, dtype=np.float32),
                    action=np.asarray(action, dtype=np.float32),
                    log_prob=float(log_prob),
                    reward=0.0,
                    done=False,
                    value=float(value),
                )

        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        done = all(terminations.values()) or all(truncations.values())

        for agent in env.agents:
            transition = cached[agent]
            transition.reward = float(rewards[agent])
            transition.done = bool(terminations[agent] or truncations[agent])
            role_trajectories[_role_from_agent(agent)][agent].transitions.append(transition)

        obs = next_obs

    return role_trajectories, env.get_episode_stats()


# ---------------------------------------------------------------------------
# Hybrid training loop
# ---------------------------------------------------------------------------

def build_llm_emitter_policies(
    env: CCUSEnv,
    *,
    base_url: str = "http://localhost:11434/v1",
    model: str = "qwen3",
    api_key: str = "none",
    call_interval: int = 12,
    temperature: float = 0.3,
    timeout: int = 60,
) -> Dict[str, LLMEmitterPolicy]:
    """Create one LLMEmitterPolicy per emitter agent."""
    groups = build_role_groups(env)
    policies: Dict[str, LLMEmitterPolicy] = {}
    for agent in groups.get("emitter", []):
        eid = env._agent_comp_id[agent]
        n_routes = len(env.physical_layer.get_routes_for_emitter(eid))
        act_dim = int(np.prod(env.action_space(agent).shape))
        policies[agent] = LLMEmitterPolicy(
            agent_id=agent,
            n_routes=n_routes,
            action_dim=act_dim,
            base_url=base_url,
            model=model,
            api_key=api_key,
            call_interval=call_interval,
            temperature=temperature,
            timeout=timeout,
        )
    return policies


def train_hybrid(
    env: CCUSEnv,
    llm_policies: Dict[str, LLMEmitterPolicy],
    train_config: Dict[str, Any] | None = None,
    *,
    episodes: int = 10,
    seed: int = 42,
    device: str = "cpu",
    mappo_policies: Dict[str, RoleMAPPOPolicy] | None = None,
    best_checkpoint_path: str = "",
    latest_checkpoint_path: str = "",
) -> Dict[str, Any]:
    """Train MAPPO policies for transport + storage while LLM controls emitters.

    Args:
        env: CCUSEnv instance.
        llm_policies: LLMEmitterPolicy per emitter agent.
        train_config: Optional overrides for DEFAULT_MAPPO_CONFIG.
        episodes: Number of training episodes.
        seed: Random seed.
        device: Torch device string.
        mappo_policies: Pre-built policies (e.g. from checkpoint). If None,
            new policies are created for transport and storage only.
        best_checkpoint_path: Save best MAPPO checkpoint here.
        latest_checkpoint_path: Save latest MAPPO checkpoint here.

    Returns:
        Dict with keys: policies, history, train_config, best_metric, best_episode.
    """
    cfg = dict(DEFAULT_MAPPO_CONFIG)
    if train_config:
        cfg.update(train_config)
    _set_seed(seed)

    # Build MAPPO policies for transport + storage only
    if mappo_policies is None:
        all_policies = build_role_policies(env, cfg, device=device)
        mappo_policies = {
            role: p for role, p in all_policies.items() if role != "emitter"
        }

    gamma = float(cfg.get("gamma", DEFAULT_MAPPO_CONFIG["gamma"]))
    gae_lambda = float(cfg.get("gae_lambda", DEFAULT_MAPPO_CONFIG["gae_lambda"]))
    best_metric_name = str(cfg.get("best_metric", "score"))
    best_metric_value = float("-inf")
    best_episode_record: Dict[str, Any] | None = None

    history: List[Dict[str, Any]] = []

    for episode_idx in range(episodes):
        # Reset LLM caches at episode start
        for llm_p in llm_policies.values():
            llm_p.reset()

        trajectories, ep_stats = collect_hybrid_episode(
            env, llm_policies, mappo_policies, seed=seed + episode_idx
        )

        # Update only transport + storage policies
        role_updates: Dict[str, Dict[str, float]] = {}
        for role, role_trajs in trajectories.items():
            if role == "emitter" or not role_trajs or role not in mappo_policies:
                continue
            batch = _build_batch(role_trajs, gamma=gamma, gae_lambda=gae_lambda, device=device)
            stats = mappo_policies[role].update(batch, cfg)
            role_updates[role] = {
                "policy_loss": stats.policy_loss,
                "value_loss": stats.value_loss,
                "entropy": stats.entropy,
            }

        # Build episode record (same schema as train_mappo for easy comparison)
        episode_record: Dict[str, Any] = {
            "episode": episode_idx,
            "total_stored": float(ep_stats["total_stored"]),
            "total_vented": float(ep_stats["total_vented"]),
            "total_captured": float(ep_stats.get("total_captured", 0.0)),
            "transport_cost": float(ep_stats.get("transport_cost", 0.0)),
            "capture_cost": float(ep_stats.get("capture_cost", 0.0)),
            "energy_use": float(ep_stats.get("energy_use", 0.0)),
            "pressure_violations": int(ep_stats["pressure_violations"]),
            "quality_violations": int(ep_stats.get("quality_violations", 0)),
            "role_updates": role_updates,
            # Log first LLM call reasoning per episode for interpretability
            "llm_reasoning": {
                agent: (p.call_log[-1]["reasoning"] if p.call_log else "")
                for agent, p in llm_policies.items()
            },
        }
        episode_record["score"] = score_episode(episode_record, cfg)
        history.append(episode_record)

        metric_value = selection_metric_value(episode_record, best_metric_name, cfg)
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_episode_record = dict(episode_record)
            if best_checkpoint_path:
                save_checkpoint(
                    best_checkpoint_path,
                    mappo_policies,
                    metadata={
                        "seed": seed,
                        "best_metric": best_metric_name,
                        "best_metric_value": best_metric_value,
                        "best_episode": episode_idx,
                        "mode": "hybrid_llm_emitter",
                    },
                )
        if latest_checkpoint_path:
            save_checkpoint(
                latest_checkpoint_path,
                mappo_policies,
                metadata={
                    "seed": seed,
                    "latest_episode": episode_idx,
                    "mode": "hybrid_llm_emitter",
                },
            )

    return {
        "policies": mappo_policies,
        "llm_policies": llm_policies,
        "history": history,
        "train_config": cfg,
        "best_metric": {
            "name": best_metric_name,
            "value": best_metric_value,
            "episode": best_episode_record["episode"] if best_episode_record else None,
        },
        "best_episode": best_episode_record,
    }
