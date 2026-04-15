"""Economic rule-based baseline for CCUS-Gym."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from ccus_gym.core.quality import compute_effective_stream, storage_quality_penalty
from ccus_gym.sim.env import CCUSEnv


DEFAULT_RULE_BASED_CONFIG: Dict[str, Any] = {
    "profit_margin_scale": 120.0,
    "buffer_emergency_threshold": 0.75,
    "storage_soft_limit": 0.80,
    "storage_hard_limit": 0.92,
    "base_ship_quality": 0.91,
    "base_rail_quality": 0.89,
    "offspec_weight": 0.35,
    "pressure_weight": 0.45,
    "transport_util_weight": 0.25,
}


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _transport_object(env: CCUSEnv, mode_name: str) -> Any:
    if mode_name == "pipeline":
        return env.physical_layer.pipeline
    if mode_name == "ship":
        return env.physical_layer.ships
    if mode_name == "rail":
        return env.physical_layer.rail
    return None


def _transport_base_cost(env: CCUSEnv, mode_name: str) -> float:
    obj = _transport_object(env, mode_name)
    return float(getattr(obj, "base_cost", 0.0)) if obj is not None else 0.0


def _connected_storage_ids_for_mode(env: CCUSEnv, mode_name: str) -> List[int]:
    mode_idx = env.physical_layer._mode_index.get(mode_name, -1)  # type: ignore[attr-defined]
    return list(env.physical_layer._transport_to_storage.get(mode_idx, []))  # type: ignore[attr-defined]


def _storage_target_to_raw(site: Any, target_purity: float) -> float:
    raw = (float(target_purity) - float(site.min_purity)) / 0.08 + 0.5
    return _clip01(raw)


def _dispatch_target_to_raw(pricing_active: bool, target: float) -> float:
    low = 0.3 if pricing_active else 0.2
    return _clip01((float(target) - low) / 0.7)


@dataclass
class EconomicRuleBasedController:
    """Economics-aware heuristic controller."""

    config: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        cfg = dict(DEFAULT_RULE_BASED_CONFIG)
        if self.config:
            cfg.update(self.config)
        self.config = cfg

    def act_all(
        self,
        env: CCUSEnv,
        observations: Dict[str, np.ndarray] | None = None,
        state: np.ndarray | None = None,
    ) -> Dict[str, np.ndarray]:
        economic = env.get_economic_context()
        physical_state = env.physical_layer.get_state()
        pricing_active = bool(getattr(env, "_pricing_active", False))

        storage_actions, storage_targets = self._build_storage_actions(env, physical_state, economic)
        transport_actions, transport_thresholds = self._build_transport_actions(
            env,
            physical_state,
            economic,
            storage_targets,
            pricing_active=pricing_active,
        )
        emitter_actions = self._build_emitter_actions(
            env,
            physical_state,
            economic,
            storage_targets,
            transport_thresholds,
            pricing_active=pricing_active,
        )

        actions: Dict[str, np.ndarray] = {}
        actions.update(emitter_actions)
        actions.update(transport_actions)
        actions.update(storage_actions)
        return actions

    def _build_storage_actions(
        self,
        env: CCUSEnv,
        physical_state: Dict[str, Any],
        economic: Dict[str, float],
    ) -> Tuple[Dict[str, np.ndarray], Dict[int, float]]:
        actions: Dict[str, np.ndarray] = {}
        quality_targets: Dict[int, float] = {}
        offspec_penalty = float(economic.get("offspec_penalty", 0.0))

        for sid, site in env.physical_layer.storage_sites.items():
            sstate = physical_state.get("storage_sites", {}).get(sid, {})
            pressure_frac = float(sstate.get("pressure_frac", 0.0))
            fill_frac = float(sstate.get("fill_frac", 0.0))
            is_disrupted = float(sstate.get("is_disrupted", 0.0)) > 0.5

            if is_disrupted or pressure_frac >= float(self.config["storage_hard_limit"]):
                inject_frac = 0.0
            elif pressure_frac >= 0.88:
                inject_frac = 0.18
            elif pressure_frac >= float(self.config["storage_soft_limit"]):
                inject_frac = 0.42
            elif pressure_frac >= 0.65:
                inject_frac = 0.72
            else:
                inject_frac = 0.92 - 0.15 * fill_frac

            target_purity = min(
                0.995,
                max(float(site.min_purity), float(site.min_purity) + 0.01 + 0.02 * min(1.0, offspec_penalty / 12.0)),
            )
            quality_targets[int(sid)] = target_purity
            raw_quality = _storage_target_to_raw(site, target_purity)
            actions[f"storage_{sid}"] = np.asarray(
                [_clip01(inject_frac), raw_quality],
                dtype=np.float32,
            )

        return actions, quality_targets

    def _build_transport_actions(
        self,
        env: CCUSEnv,
        physical_state: Dict[str, Any],
        economic: Dict[str, float],
        storage_targets: Dict[int, float],
        *,
        pricing_active: bool,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
        actions: Dict[str, np.ndarray] = {}
        thresholds: Dict[str, Dict[str, float]] = {}
        offspec_penalty = float(economic.get("offspec_penalty", 0.0))

        for mode_name in env._transport_modes:  # type: ignore[attr-defined]
            if mode_name in getattr(env, "_passive_transport_modes", set()):
                continue

            tstate = physical_state.get("transports", {}).get(mode_name, {})
            available = float(tstate.get("available_capacity", 0.0))
            capacity = max(float(tstate.get("capacity", 1e-9)), 1e-9)
            utilization = float(tstate.get("utilization", 0.0))
            avail_frac = available / capacity
            connected_storage = _connected_storage_ids_for_mode(env, mode_name)
            pressure_fracs = [
                float(physical_state.get("storage_sites", {}).get(sid, {}).get("pressure_frac", 0.0))
                for sid in connected_storage
            ]
            avg_pressure = float(np.mean(pressure_fracs)) if pressure_fracs else 0.0

            target_threshold = np.clip(
                0.90
                - float(self.config["pressure_weight"]) * max(0.0, avg_pressure - 0.55)
                - float(self.config["transport_util_weight"]) * utilization
                + 0.15 * avail_frac,
                0.25 if pricing_active else 0.2,
                0.95,
            )

            if mode_name == "ship":
                quality_threshold = np.clip(
                    float(self.config["base_ship_quality"])
                    + float(self.config["offspec_weight"]) * min(0.08, offspec_penalty / 200.0)
                    - 0.03 * avail_frac,
                    0.86,
                    0.97,
                )
            else:
                quality_threshold = np.clip(
                    float(self.config["base_rail_quality"])
                    + 0.03 * min(1.0, offspec_penalty / 20.0)
                    - 0.02 * avail_frac,
                    0.85,
                    0.95,
                )

            if len(connected_storage) > 1:
                storage_scores = []
                for sid in connected_storage[:2]:
                    target_purity = storage_targets.get(int(sid), 0.9)
                    pressure = float(
                        physical_state.get("storage_sites", {}).get(sid, {}).get("pressure_frac", 0.0)
                    )
                    storage_scores.append((sid, (1.0 - pressure) + 0.15 * target_purity))
                destination_pref = 1.0 if storage_scores[1][1] > storage_scores[0][1] else 0.0
            else:
                destination_pref = 0.5

            thresholds[mode_name] = {
                "threshold": float(target_threshold),
                "quality_threshold": float(quality_threshold),
                "destination_pref": float(destination_pref),
            }

            dispatch_raw = _dispatch_target_to_raw(pricing_active, target_threshold)
            quality_raw = _clip01((quality_threshold - 0.85) / 0.12)

            if pricing_active:
                carbon_tax = max(float(economic.get("carbon_tax", 0.0)), 1.0)
                posted_price = np.clip(
                    (_transport_base_cost(env, mode_name) * (1.05 + utilization)) / (2.0 * carbon_tax),
                    0.1,
                    0.95,
                )
                if mode_name == "ship":
                    actions[f"transport_{mode_name}"] = np.asarray(
                        [dispatch_raw, posted_price, destination_pref, quality_raw, 0.5],
                        dtype=np.float32,
                    )
                else:
                    actions[f"transport_{mode_name}"] = np.asarray(
                        [dispatch_raw, posted_price, quality_raw, 0.75],
                        dtype=np.float32,
                    )
            else:
                actions[f"transport_{mode_name}"] = np.asarray(
                    [dispatch_raw, destination_pref, quality_raw],
                    dtype=np.float32,
                )

        return actions, thresholds

    def _build_emitter_actions(
        self,
        env: CCUSEnv,
        physical_state: Dict[str, Any],
        economic: Dict[str, float],
        storage_targets: Dict[int, float],
        transport_thresholds: Dict[str, Dict[str, float]],
        *,
        pricing_active: bool,
    ) -> Dict[str, np.ndarray]:
        actions: Dict[str, np.ndarray] = {}
        carbon_tax = float(economic.get("carbon_tax", 0.0))
        electricity_price = float(economic.get("electricity_price", 0.0))
        capture_subsidy = float(economic.get("capture_subsidy", 0.0))
        storage_credit = float(economic.get("storage_credit", 0.0))
        offspec_penalty = float(economic.get("offspec_penalty", 0.0))
        congestion_threshold = float(getattr(env, "congestion_threshold", 0.8))
        k_congestion = float(getattr(env, "k_congestion", 5.0))

        for eid, emitter in env.physical_layer.emitters.items():
            e_state = physical_state.get("emitters", {}).get(eid, {})
            buffer_frac = float(e_state.get("buffer_frac", 0.0))
            routes = env.physical_layer.get_routes_for_emitter(eid)
            route_scores: List[float] = []
            route_efforts: List[float] = []

            for transport_idx, sid in routes:
                mode_name = env._transport_modes[transport_idx]  # type: ignore[attr-defined]
                tstate = physical_state.get("transports", {}).get(mode_name, {})
                sstate = physical_state.get("storage_sites", {}).get(sid, {})

                available = float(tstate.get("available_capacity", 0.0))
                capacity = max(float(tstate.get("capacity", 1e-9)), 1e-9)
                avail_frac = available / capacity
                pressure_frac = float(sstate.get("pressure_frac", 0.0))
                disrupted = float(tstate.get("is_disrupted", 0.0)) > 0.5 or float(sstate.get("is_disrupted", 0.0)) > 0.5

                desired_purity = max(
                    transport_thresholds.get(mode_name, {}).get("quality_threshold", 0.85),
                    storage_targets.get(int(sid), getattr(env.physical_layer.storage_sites[sid], "min_purity", 0.9)),
                )
                purification_effort = _clip01((desired_purity - float(emitter.base_purity)) / 0.03)
                _, composition = compute_effective_stream(
                    emitter.capture_method,
                    purification_effort,
                    base_purity=emitter.base_purity,
                    base_composition=emitter.composition,
                )
                quality_penalty, _ = storage_quality_penalty(
                    composition,
                    min_purity=desired_purity,
                    max_impurities=env.physical_layer.storage_sites[sid].max_impurities,
                )

                capture_cost = emitter.capture_cost_per_t * (1.0 + emitter.purification_cost_factor * purification_effort)
                energy_cost = electricity_price * emitter.capture_energy_mwh_per_t * (
                    1.0 + emitter.purification_energy_factor * purification_effort
                )
                transport_cost = _transport_base_cost(env, mode_name)
                congestion_cost = 0.0
                if mode_name == "pipeline":
                    pipe_util = float(tstate.get("utilization", 0.0))
                    if pipe_util > congestion_threshold:
                        congestion_cost = transport_cost * (pipe_util - congestion_threshold) * k_congestion

                benefit = carbon_tax + capture_subsidy + storage_credit
                margin = benefit - capture_cost - energy_cost - transport_cost - congestion_cost
                viability = max(0.0, 1.0 - pressure_frac) * np.clip(avail_frac + 0.15, 0.0, 1.0)
                if disrupted:
                    viability *= 0.1
                route_score = margin * viability - offspec_penalty * quality_penalty - 8.0 * max(0.0, pressure_frac - 0.85)
                route_scores.append(float(route_score))
                route_efforts.append(float(purification_effort))

            best_score = max(route_scores) if route_scores else -np.inf
            positive_scores = [max(0.0, s) for s in route_scores]

            if best_score > 0.0:
                capture_target = 0.92
                send_raw = np.clip(0.55 + best_score / float(self.config["profit_margin_scale"]) + 0.25 * buffer_frac, 0.35, 1.0)
            elif buffer_frac >= float(self.config["buffer_emergency_threshold"]):
                capture_target = 0.45
                send_raw = np.clip(0.35 + 0.8 * (buffer_frac - float(self.config["buffer_emergency_threshold"])), 0.25, 0.75)
            else:
                capture_target = 0.2 if carbon_tax <= 0.0 else 0.35
                send_raw = 0.0

            if positive_scores and sum(positive_scores) > 1e-9:
                scaled = np.asarray(positive_scores, dtype=np.float64)
                scaled = scaled / np.max(scaled)
                route_logits = scaled.tolist()
                purification = float(np.average(route_efforts, weights=np.asarray(positive_scores)))
            elif route_scores:
                best_idx = int(np.argmax(route_scores))
                route_logits = [0.0 for _ in route_scores]
                route_logits[best_idx] = 1.0
                purification = route_efforts[best_idx]
            else:
                route_logits = []
                purification = 0.0

            raw_capture = _clip01((capture_target - 0.1) / 0.9)
            action_values = route_logits + [
                _clip01(send_raw),
                raw_capture,
                _clip01(purification),
            ]

            if pricing_active:
                bid_signal = 0.0 if best_score <= 0.0 else _clip01(best_score / max(carbon_tax, 50.0))
                action_values.append(bid_signal)

            action_dim = int(np.prod(env.action_space(f"emitter_{eid}").shape))
            padded = np.zeros(action_dim, dtype=np.float32)
            padded[: min(action_dim, len(action_values))] = np.asarray(
                action_values[:action_dim],
                dtype=np.float32,
            )
            actions[f"emitter_{eid}"] = padded

        return actions


def evaluate_rule_based(
    env: CCUSEnv,
    controller: EconomicRuleBasedController | None = None,
    *,
    episodes: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    ctrl = controller or EconomicRuleBasedController()
    episode_stats: List[Dict[str, Any]] = []
    for episode_idx in range(episodes):
        observations, _ = env.reset(seed=seed + episode_idx)
        done = False
        while not done:
            state = env.global_state_vector()
            actions = ctrl.act_all(env, observations, state)
            observations, _, terminations, truncations, _ = env.step(actions)
            done = all(terminations.values()) or all(truncations.values())
        ep_stats = env.get_episode_stats()
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
