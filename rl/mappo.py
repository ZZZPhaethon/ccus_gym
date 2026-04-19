"""Minimal MAPPO trainer for CCUS-Gym.

This module implements a compact role-shared MAPPO baseline:
- one actor-critic pair per role (emitter / transport / storage)
- beta-distribution policy for Box([0, 1]) action spaces
- centralized critic over the flattened global state vector

It is intentionally lightweight so the project has an end-to-end
trainable deep RL loop without depending on an external MARL library.
"""

from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, Independent
from torch.nn import functional as F

from ccus_gym.sim.env import CCUSEnv
from ccus_gym.rl.training import DEFAULT_MAPPO_CONFIG, build_role_groups


def _role_from_agent(agent: str) -> str:
    if agent.startswith("emitter_"):
        return "emitter"
    if agent.startswith("transport_"):
        return "transport"
    if agent.startswith("storage_"):
        return "storage"
    raise ValueError(f"Unknown agent name: {agent}")


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def score_episode(
    episode_record: Dict[str, Any],
    config: Dict[str, Any] | None = None,
) -> float:
    cfg = dict(DEFAULT_MAPPO_CONFIG)
    if config:
        cfg.update(config)
    stored = float(episode_record.get("total_stored", 0.0))
    vented = float(episode_record.get("total_vented", 0.0))
    quality_violations = float(episode_record.get("quality_violations", 0.0))
    pressure_violations = float(episode_record.get("pressure_violations", 0.0))
    return (
        stored
        - float(cfg.get("score_vented_weight", 1.0)) * vented
        - float(cfg.get("score_quality_weight", 0.5)) * quality_violations
        - float(cfg.get("score_pressure_weight", 2.0)) * pressure_violations
    )


def selection_metric_value(
    episode_record: Dict[str, Any],
    metric_name: str,
    config: Dict[str, Any] | None = None,
) -> float:
    if metric_name == "score":
        return score_episode(episode_record, config)
    raw = float(episode_record.get(metric_name, 0.0))
    if metric_name in {"total_vented", "pressure_violations", "quality_violations"}:
        return -raw
    return raw


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BetaActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_dim, hidden_dim)
        self.alpha_head = nn.Linear(hidden_dim, action_dim)
        self.beta_head = nn.Linear(hidden_dim, action_dim)

    def dist(self, obs: torch.Tensor) -> Independent:
        h = self.backbone(obs)
        alpha = F.softplus(self.alpha_head(h)) + 1.0
        beta = F.softplus(self.beta_head(h)) + 1.0
        return Independent(Beta(alpha, beta), 1)


class ValueCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = MLP(state_dim, hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


@dataclass
class Transition:
    obs: np.ndarray
    state: np.ndarray
    action: np.ndarray
    log_prob: float
    reward: float
    done: bool
    value: float


@dataclass
class AgentTrajectory:
    transitions: List[Transition]


@dataclass
class TrainBatch:
    obs: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    old_log_prob: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


@dataclass
class RoleTrainStats:
    policy_loss: float
    value_loss: float
    entropy: float


class ReturnNormalizer:
    """Per-role running mean/variance normalizer for value function targets.

    Critic is trained on normalized returns; raw values are recovered via
    denormalize() so that GAE operates in the original reward scale.
    Uses exponential moving average to track statistics online.
    """

    def __init__(self, momentum: float = 0.99, eps: float = 1e-8) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.momentum = momentum
        self.eps = eps
        self._initialized = False

    def update(self, returns: np.ndarray) -> None:
        batch_mean = float(np.mean(returns))
        batch_var = float(np.var(returns))
        if not self._initialized:
            self.mean = batch_mean
            self.var = max(batch_var, self.eps)
            self._initialized = True
        else:
            m = self.momentum
            self.mean = m * self.mean + (1.0 - m) * batch_mean
            self.var = max(m * self.var + (1.0 - m) * batch_var, self.eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var ** 0.5 + self.eps)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.var ** 0.5 + self.eps) + self.mean

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "initialized": self._initialized}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.mean = float(d["mean"])
        self.var = float(d["var"])
        self._initialized = bool(d["initialized"])


class RoleMAPPOPolicy:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        config: Dict[str, Any],
        *,
        device: str,
    ) -> None:
        hidden_dim = int(config.get("hidden_dim", 128))
        self.device = torch.device(device)
        self.actor = BetaActor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic = ValueCritic(state_dim, hidden_dim).to(self.device)
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(
            params,
            lr=float(config.get("learning_rate", DEFAULT_MAPPO_CONFIG["learning_rate"])),
        )
        self.value_normalizer = ReturnNormalizer()

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor.dist(obs_t)
        if deterministic:
            base_dist = dist.base_dist
            action_t = base_dist.concentration1 / (
                base_dist.concentration1 + base_dist.concentration0
            )
        else:
            action_t = dist.sample()
        action_t = action_t.clamp(1e-4, 1.0 - 1e-4)
        log_prob = dist.log_prob(action_t)
        value = self.critic(state_t)
        # Denormalize so GAE is computed in the original reward scale
        if self.value_normalizer._initialized:
            value = self.value_normalizer.denormalize(value)
        return (
            action_t.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def update(self, batch: TrainBatch, config: Dict[str, Any]) -> RoleTrainStats:
        clip_ratio = float(config.get("clip_ratio", DEFAULT_MAPPO_CONFIG["clip_ratio"]))
        ppo_epochs = int(config.get("ppo_epochs", DEFAULT_MAPPO_CONFIG["ppo_epochs"]))
        mini_batches = int(config.get("mini_batches", DEFAULT_MAPPO_CONFIG["mini_batches"]))
        entropy_coef = float(config.get("entropy_coef", DEFAULT_MAPPO_CONFIG["entropy_coef"]))
        value_coef = float(config.get("value_coef", DEFAULT_MAPPO_CONFIG["value_coef"]))

        # Update running stats and pre-compute normalized returns for this batch
        returns_np = batch.returns.cpu().numpy()
        self.value_normalizer.update(returns_np)
        returns_normalized = self.value_normalizer.normalize(batch.returns)

        n = batch.obs.shape[0]
        mb_size = max(1, n // max(1, mini_batches))

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []

        for _ in range(ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, mb_size):
                idx = perm[start:start + mb_size]
                obs = batch.obs[idx]
                state = batch.state[idx]
                action = batch.action[idx].clamp(1e-4, 1.0 - 1e-4)
                old_log_prob = batch.old_log_prob[idx]
                ret_norm = returns_normalized[idx]
                advantages = batch.advantages[idx]

                dist = self.actor.dist(obs)
                new_log_prob = dist.log_prob(action)
                entropy = dist.entropy().mean()
                value = self.critic(state)

                ratio = torch.exp(new_log_prob - old_log_prob)
                surrogate_1 = ratio * advantages
                surrogate_2 = torch.clamp(
                    ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
                ) * advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                # Value loss in normalized space — scale-invariant across roles
                value_loss = torch.mean((ret_norm - value) ** 2)
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    max_norm=1.0,
                )
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

        return RoleTrainStats(
            policy_loss=float(np.mean(policy_losses)) if policy_losses else 0.0,
            value_loss=float(np.mean(value_losses)) if value_losses else 0.0,
            entropy=float(np.mean(entropies)) if entropies else 0.0,
        )


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_adv = delta + gamma * gae_lambda * non_terminal * last_adv
        advantages[t] = last_adv
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def _build_batch(
    trajectories: Dict[str, AgentTrajectory],
    *,
    gamma: float,
    gae_lambda: float,
    device: str,
) -> TrainBatch:
    obs_list: List[np.ndarray] = []
    state_list: List[np.ndarray] = []
    action_list: List[np.ndarray] = []
    logp_list: List[float] = []
    return_list: List[np.ndarray] = []
    adv_list: List[np.ndarray] = []

    for traj in trajectories.values():
        rewards = np.asarray([t.reward for t in traj.transitions], dtype=np.float32)
        values = np.asarray([t.value for t in traj.transitions], dtype=np.float32)
        dones = np.asarray([t.done for t in traj.transitions], dtype=np.float32)
        advantages, returns = _compute_gae(
            rewards,
            values,
            dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        for i, transition in enumerate(traj.transitions):
            obs_list.append(transition.obs)
            state_list.append(transition.state)
            action_list.append(transition.action)
            logp_list.append(transition.log_prob)
            return_list.append(returns[i])
            adv_list.append(advantages[i])

    advantages_arr = np.asarray(adv_list, dtype=np.float32)
    advantages_arr = (advantages_arr - advantages_arr.mean()) / (
        advantages_arr.std() + 1e-8
    )

    return TrainBatch(
        obs=torch.as_tensor(np.asarray(obs_list), dtype=torch.float32, device=device),
        state=torch.as_tensor(np.asarray(state_list), dtype=torch.float32, device=device),
        action=torch.as_tensor(np.asarray(action_list), dtype=torch.float32, device=device),
        old_log_prob=torch.as_tensor(np.asarray(logp_list), dtype=torch.float32, device=device),
        returns=torch.as_tensor(np.asarray(return_list), dtype=torch.float32, device=device),
        advantages=torch.as_tensor(advantages_arr, dtype=torch.float32, device=device),
    )


def _collect_episode(
    env: CCUSEnv,
    policies: Dict[str, RoleMAPPOPolicy],
    *,
    seed: int | None = None,
    deterministic: bool = False,
) -> Tuple[Dict[str, Dict[str, AgentTrajectory]], Dict[str, Any]]:
    obs, _ = env.reset(seed=seed)
    role_trajectories: Dict[str, Dict[str, AgentTrajectory]] = {
        "emitter": {},
        "transport": {},
        "storage": {},
    }

    for role, agents in build_role_groups(env).items():
        role_trajectories[role] = {
            agent: AgentTrajectory(transitions=[]) for agent in agents
        }

    done = False
    while not done:
        state = env.global_state_vector()
        actions: Dict[str, np.ndarray] = {}
        cached: Dict[str, Transition] = {}
        for agent in env.agents:
            role = _role_from_agent(agent)
            action, log_prob, value = policies[role].act(
                obs[agent],
                state,
                deterministic=deterministic,
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
            role = _role_from_agent(agent)
            transition = cached[agent]
            transition.reward = float(rewards[agent])
            transition.done = bool(terminations[agent] or truncations[agent])
            role_trajectories[role][agent].transitions.append(transition)
        obs = next_obs

    return role_trajectories, env.get_episode_stats()


def build_role_policies(
    env: CCUSEnv,
    config: Dict[str, Any],
    *,
    device: str = "cpu",
) -> Dict[str, RoleMAPPOPolicy]:
    role_groups = build_role_groups(env)
    state_dim = int(env.global_state_vector().shape[0])
    policies: Dict[str, RoleMAPPOPolicy] = {}
    for role, agents in role_groups.items():
        if not agents:
            continue
        obs_dim = int(np.prod(env.observation_space(agents[0]).shape))
        act_dim = int(np.prod(env.action_space(agents[0]).shape))
        policies[role] = RoleMAPPOPolicy(
            obs_dim=obs_dim,
            action_dim=act_dim,
            state_dim=state_dim,
            config=config,
            device=device,
        )
    return policies


def train_mappo(
    env: CCUSEnv,
    train_config: Dict[str, Any] | None = None,
    *,
    episodes: int = 10,
    seed: int = 42,
    device: str = "cpu",
    policies: Dict[str, RoleMAPPOPolicy] | None = None,
    best_checkpoint_path: str = "",
    latest_checkpoint_path: str = "",
) -> Dict[str, Any]:
    cfg = dict(DEFAULT_MAPPO_CONFIG)
    if train_config:
        cfg.update(train_config)
    _set_seed(seed)
    if policies is None:
        policies = build_role_policies(env, cfg, device=device)

    history: List[Dict[str, Any]] = []
    best_metric_name = str(cfg.get("best_metric", "score"))
    best_metric_value = float("-inf")
    best_episode_record: Dict[str, Any] | None = None
    best_policies_state: Dict[str, Any] | None = None
    gamma = float(cfg.get("gamma", DEFAULT_MAPPO_CONFIG["gamma"]))
    gae_lambda = float(cfg.get("gae_lambda", DEFAULT_MAPPO_CONFIG["gae_lambda"]))

    for episode_idx in range(episodes):
        trajectories, ep_stats = _collect_episode(env, policies, seed=seed + episode_idx)
        role_updates: Dict[str, Dict[str, float]] = {}
        for role, role_trajs in trajectories.items():
            if not role_trajs:
                continue
            batch = _build_batch(
                role_trajs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                device=device,
            )
            stats = policies[role].update(batch, cfg)
            role_updates[role] = {
                "policy_loss": stats.policy_loss,
                "value_loss": stats.value_loss,
                "entropy": stats.entropy,
            }

        episode_record = {
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
        }
        episode_record["score"] = score_episode(episode_record, cfg)
        history.append(episode_record)

        metric_value = selection_metric_value(episode_record, best_metric_name, cfg)
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_episode_record = dict(episode_record)
            best_policies_state = {
                role: {
                    "actor": deepcopy(policy.actor.state_dict()),
                    "critic": deepcopy(policy.critic.state_dict()),
                    "optimizer": deepcopy(policy.optimizer.state_dict()),
                }
                for role, policy in policies.items()
            }
            if best_checkpoint_path:
                save_checkpoint(
                    best_checkpoint_path,
                    policies,
                    metadata={
                        "seed": seed,
                        "best_metric": best_metric_name,
                        "best_metric_value": best_metric_value,
                        "best_episode": episode_idx,
                    },
                )
        if latest_checkpoint_path:
            save_checkpoint(
                latest_checkpoint_path,
                policies,
                metadata={
                    "seed": seed,
                    "latest_episode": episode_idx,
                    "best_metric": best_metric_name,
                    "best_metric_value": best_metric_value,
                },
            )

    return {
        "policies": policies,
        "history": history,
        "train_config": cfg,
        "best_metric": {
            "name": best_metric_name,
            "value": best_metric_value,
            "episode": best_episode_record["episode"] if best_episode_record else None,
        },
        "best_episode": best_episode_record,
        "best_policies_state": best_policies_state,
    }


def evaluate_policies(
    env: CCUSEnv,
    policies: Dict[str, RoleMAPPOPolicy],
    *,
    episodes: int = 3,
    seed: int = 42,
    deterministic: bool = True,
) -> Dict[str, Any]:
    episode_stats: List[Dict[str, Any]] = []
    for episode_idx in range(episodes):
        _, ep_stats = _collect_episode(
            env,
            policies,
            seed=seed + episode_idx,
            deterministic=deterministic,
        )
        episode_stats.append(
            {
                "episode": episode_idx,
                "total_stored": float(ep_stats["total_stored"]),
                "total_vented": float(ep_stats["total_vented"]),
                "total_captured": float(ep_stats["total_captured"]),
                "pressure_violations": int(ep_stats["pressure_violations"]),
                "quality_violations": int(ep_stats.get("quality_violations", 0)),
                "transport_cost": float(ep_stats.get("transport_cost", 0.0)),
                "capture_cost": float(ep_stats.get("capture_cost", 0.0)),
            }
        )

    def _mean(key: str) -> float:
        return float(np.mean([ep[key] for ep in episode_stats])) if episode_stats else 0.0

    return {
        "episodes": episode_stats,
        "summary": {
            "episodes": episodes,
            "mean_total_stored": _mean("total_stored"),
            "mean_total_vented": _mean("total_vented"),
            "mean_total_captured": _mean("total_captured"),
            "mean_pressure_violations": _mean("pressure_violations"),
            "mean_quality_violations": _mean("quality_violations"),
            "mean_transport_cost": _mean("transport_cost"),
            "mean_capture_cost": _mean("capture_cost"),
        },
    }


def save_checkpoint(
    path: str,
    policies: Dict[str, RoleMAPPOPolicy],
    metadata: Dict[str, Any] | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "metadata": metadata or {},
        "roles": {},
    }
    for role, policy in policies.items():
        payload["roles"][role] = {
            "actor": policy.actor.state_dict(),
            "critic": policy.critic.state_dict(),
            "optimizer": policy.optimizer.state_dict(),
            "value_normalizer": policy.value_normalizer.state_dict(),
        }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    env: CCUSEnv,
    config: Dict[str, Any] | None = None,
    *,
    device: str = "cpu",
    load_optimizer: bool = True,
) -> Tuple[Dict[str, RoleMAPPOPolicy], Dict[str, Any]]:
    cfg = dict(DEFAULT_MAPPO_CONFIG)
    if config:
        cfg.update(config)
    policies = build_role_policies(env, cfg, device=device)
    payload = torch.load(path, map_location=device)
    for role, state in payload.get("roles", {}).items():
        if role not in policies:
            continue
        policies[role].actor.load_state_dict(state["actor"])
        policies[role].critic.load_state_dict(state["critic"])
        if load_optimizer and "optimizer" in state:
            policies[role].optimizer.load_state_dict(state["optimizer"])
        if "value_normalizer" in state:
            policies[role].value_normalizer.load_state_dict(state["value_normalizer"])
    return policies, payload.get("metadata", {})


def save_history_jsonl(path: str, history: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def save_history_csv(path: str, history: List[Dict[str, Any]]) -> None:
    if not history:
        raise ValueError("History is empty; nothing to save.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "episode",
        "total_stored",
        "total_vented",
        "total_captured",
        "transport_cost",
        "capture_cost",
        "energy_use",
        "pressure_violations",
        "quality_violations",
        "score",
        "emitter_policy_loss",
        "emitter_value_loss",
        "emitter_entropy",
        "transport_policy_loss",
        "transport_value_loss",
        "transport_entropy",
        "storage_policy_loss",
        "storage_value_loss",
        "storage_entropy",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            role_updates = row.get("role_updates", {})
            writer.writerow(
                {
                    "episode": row.get("episode"),
                    "total_stored": row.get("total_stored"),
                    "total_vented": row.get("total_vented"),
                    "total_captured": row.get("total_captured"),
                    "transport_cost": row.get("transport_cost"),
                    "capture_cost": row.get("capture_cost"),
                    "energy_use": row.get("energy_use"),
                    "pressure_violations": row.get("pressure_violations"),
                    "quality_violations": row.get("quality_violations"),
                    "score": row.get("score"),
                    "emitter_policy_loss": role_updates.get("emitter", {}).get("policy_loss"),
                    "emitter_value_loss": role_updates.get("emitter", {}).get("value_loss"),
                    "emitter_entropy": role_updates.get("emitter", {}).get("entropy"),
                    "transport_policy_loss": role_updates.get("transport", {}).get("policy_loss"),
                    "transport_value_loss": role_updates.get("transport", {}).get("value_loss"),
                    "transport_entropy": role_updates.get("transport", {}).get("entropy"),
                    "storage_policy_loss": role_updates.get("storage", {}).get("policy_loss"),
                    "storage_value_loss": role_updates.get("storage", {}).get("value_loss"),
                    "storage_entropy": role_updates.get("storage", {}).get("entropy"),
                }
            )


def write_tensorboard_history(log_dir: str, history: List[Dict[str, Any]]) -> None:
    if not history:
        raise ValueError("History is empty; nothing to log.")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard support requires the 'tensorboard' package."
        ) from exc

    writer = SummaryWriter(log_dir=log_dir)
    try:
        for row in history:
            step = int(row.get("episode", 0))
            writer.add_scalar("train/total_stored", float(row.get("total_stored", 0.0)), step)
            writer.add_scalar("train/total_vented", float(row.get("total_vented", 0.0)), step)
            writer.add_scalar("train/total_captured", float(row.get("total_captured", 0.0)), step)
            writer.add_scalar("train/transport_cost", float(row.get("transport_cost", 0.0)), step)
            writer.add_scalar("train/capture_cost", float(row.get("capture_cost", 0.0)), step)
            writer.add_scalar("train/energy_use", float(row.get("energy_use", 0.0)), step)
            writer.add_scalar("train/pressure_violations", float(row.get("pressure_violations", 0.0)), step)
            writer.add_scalar("train/quality_violations", float(row.get("quality_violations", 0.0)), step)
            writer.add_scalar("train/score", float(row.get("score", 0.0)), step)

            role_updates = row.get("role_updates", {})
            for role, metrics in role_updates.items():
                writer.add_scalar(
                    f"loss/{role}_policy",
                    float(metrics.get("policy_loss", 0.0)),
                    step,
                )
                writer.add_scalar(
                    f"loss/{role}_value",
                    float(metrics.get("value_loss", 0.0)),
                    step,
                )
                writer.add_scalar(
                    f"policy/{role}_entropy",
                    float(metrics.get("entropy", 0.0)),
                    step,
                )
    finally:
        writer.flush()
        writer.close()


def plot_training_history(path: str, history: List[Dict[str, Any]]) -> None:
    if not history:
        raise ValueError("History is empty; nothing to plot.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    episodes = [row["episode"] for row in history]
    total_stored = [row.get("total_stored", 0.0) for row in history]
    total_vented = [row.get("total_vented", 0.0) for row in history]
    score = [row.get("score", 0.0) for row in history]
    pressure_violations = [row.get("pressure_violations", 0) for row in history]
    quality_violations = [row.get("quality_violations", 0) for row in history]

    emitter_policy = [
        row.get("role_updates", {}).get("emitter", {}).get("policy_loss", np.nan)
        for row in history
    ]
    transport_policy = [
        row.get("role_updates", {}).get("transport", {}).get("policy_loss", np.nan)
        for row in history
    ]
    storage_policy = [
        row.get("role_updates", {}).get("storage", {}).get("policy_loss", np.nan)
        for row in history
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(episodes, total_stored, label="stored")
    ax.plot(episodes, total_vented, label="vented")
    ax.set_title("Storage vs Venting")
    ax.set_xlabel("Episode")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(episodes, score, color="black", label="score")
    ax.set_title("Training Score")
    ax.set_xlabel("Episode")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(episodes, pressure_violations, label="pressure")
    ax.plot(episodes, quality_violations, label="quality")
    ax.set_title("Violations")
    ax.set_xlabel("Episode")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(episodes, emitter_policy, label="emitter")
    ax.plot(episodes, transport_policy, label="transport")
    ax.plot(episodes, storage_policy, label="storage")
    ax.set_title("Policy Loss by Role")
    ax.set_xlabel("Episode")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
