"""PettingZoo ParallelEnv for multi-disruption CCUS network coordination.

Decision Layer: Thin wrapper around the PhysicalLayer.
Agents submit decision intents; physical layer settles and executes them.

Agents:
    - emitter_0, emitter_1, ...: One per CO2 source
    - transport_pipeline, transport_ship, [transport_rail]: One per transport mode
    - storage_0, storage_1, ...: One per storage site

Observations:
    Each agent sees its own state plus noisy information about connected
    components (noise controlled by the alpha mechanism parameter).

Actions (decision intents, NOT direct physical manipulation):
    - Emitters: routing fractions + volume nomination (remapped to [0.2, 1.0])
    - Transport: acceptance threshold (remapped to [0.3, 1.0])
    - Storage: injection rate limit fraction

Reward:
    Role-factored CTDE rewards:
    r_i = w_global * R_system + w_local * r_i_local
    where R_system is shared and r_i_local is role-specific.
"""

from __future__ import annotations

import functools
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from ccus_gym.core.physical import PhysicalLayer, PhysicalOutcome
from ccus_gym.sim.disruptions import DisruptionGenerator, MechanismAxes
from ccus_gym.core.network import CCUSNetwork, TransportType
from ccus_gym.sim.configs import MINIMAL_NETWORK_CONFIG
from ccus_gym.sim.case_loader import load_case


# ---------------------------------------------------------------------------
# Reward weights -- role-factored CTDE design
# ---------------------------------------------------------------------------
_DEFAULT_REWARD_WEIGHTS = {
    # Global/local mixing weights
    "w_global": 0.5,
    "w_local": 0.5,
    # System (shared) reward components
    "sys_co2_stored": 2.0,
    "sys_co2_vented": -1.0,
    "sys_pressure_violation": -5.0,  # applied to two-stage barrier; escalation built into barrier shape
    "sys_energy_use": -0.03,
    "sys_quality_violation": -3.0,
    # Emitter local reward components
    "emitter_co2_sent": 1.0,
    "emitter_co2_vented": -3.0,
    "emitter_buffer_penalty": -0.5,
    "emitter_buffer_bonus": 0.2,
    "emitter_transport_cost": -1.0,
    "emitter_carbon_tax_vent": -1.0,  # multiplied by carbon_tax
    "emitter_capture_cost": -1.0,
    "emitter_capture_energy": -0.03,
    "emitter_purity_bonus": 0.5,
    # Transport local reward components
    "transport_co2_delivered": 1.0,
    "transport_utilization": 0.5,
    "transport_co2_rejected": -1.0,
    "transport_revenue": 2.0,
    "transport_quality_bonus": 0.5,
    # Storage local reward components
    "storage_co2_injected": 2.0,
    "storage_pressure_violation": -5.0,  # two-stage barrier penalty; escalation built into barrier shape
    "storage_pressure_margin": 0.5,
    "storage_quality_penalty": -3.0,
    "storage_storage_credit": 1.0,
    # Institutional mechanism reward components
    "emitter_congestion_penalty": -1.0,   # congestion surcharge on pipeline
    "emitter_overflow_penalty": -2.0,     # overflow at storage attributed to emitter
    "storage_overflow_penalty": -2.0,     # overflow at storage attributed to storage
    "storage_obligation_penalty": -1.0,   # injection obligation shortfall
}

# Maximum values for observation normalization
_OBS_MAXES = {
    "buffer_level": 1.0,       # fraction
    "capture_rate": 1.0,       # fraction of max
    "pressure": 400.0,         # bar
    "pressure_frac": 1.5,      # can exceed 1.0 if over limit
    "capacity_frac": 1.0,
    "volume": 5.0,             # MtCO2
    "injectivity": 1.0,
    "time": 120.0,
    "purity": 1.0,
    "quality_penalty": 1.0,
}


class CCUSEnv(ParallelEnv):
    """Multi-agent CCUS network coordination environment.

    Decision layer: a PettingZoo ParallelEnv that wraps the PhysicalLayer.
    Agents submit decision intents each month; the physical layer settles
    nominations and executes the physics.
    """

    metadata = {
        "render_modes": ["human"],
        "name": "ccus_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.config = config if config is not None else deepcopy(MINIMAL_NETWORK_CONFIG)
        self.render_mode = render_mode

        # Build physical layer
        self.physical_layer = PhysicalLayer(self.config)

        # Also build the legacy CCUSNetwork for disruption generator compatibility
        self.network = CCUSNetwork.from_config(self.config)

        # Disruption generator
        dis_cfg = self.config.get("disruption", {})
        mech_pre = self.config.get("mechanism", {})
        self.disruption_gen = DisruptionGenerator(
            network=self.network,
            scenario_family=dis_cfg.get("scenario_family", "T"),
            severity_kappa=dis_cfg.get("severity", 0.5),
            cross_correlation=dis_cfg.get("cross_correlation", 0.0),
            seed=self.config.get("seed", 42),
            gamma=mech_pre.get("gamma", 1.0),
        )

        # Mechanism parameters
        mech = self.config.get("mechanism", {})
        self.alpha = mech.get("alpha", 0.5)
        self.beta = mech.get("beta", 0.3)
        self.gamma = mech.get("gamma", 1.0)

        # Structural mechanism axes
        self.disruption_forecast_horizon = int(round((1.0 - self.alpha) * 6))
        self.effective_cross_well_scale = self.beta * 5.0
        self.pressure_limit_scale = 1.0 - 0.2 * self.beta

        # Episode parameters
        self.episode_length = self.config.get("episode_length", 120)

        # Carbon tax and pricing mode
        self.carbon_tax: float = float(self.config.get("carbon_tax", 0.0))
        self._base_carbon_tax: float = self.carbon_tax
        self.electricity_price: float = float(self.config.get("electricity_price", 65.0))
        self._base_electricity_price: float = self.electricity_price
        self.capture_subsidy: float = float(self.config.get("capture_subsidy", 0.0))
        self.storage_credit: float = float(self.config.get("storage_credit", 0.0))
        self.offspec_penalty: float = float(self.config.get("offspec_penalty", 6.0))
        self.extreme_scenarios: List[Dict[str, Any]] = list(self.config.get("extreme_scenarios", []))
        self._active_extreme_context: Dict[str, float] = {
            "carbon_tax_multiplier": 1.0,
            "electricity_price_multiplier": 1.0,
        }
        # pricing_mode: "off" (no costs in rewards), "cost_only" (fixed costs in
        # rewards but no extra action dims), "full" (dynamic pricing with extra
        # action dims — original behaviour when carbon_tax > 0)
        self.pricing_mode: str = self.config.get("pricing_mode", "auto")
        if self.pricing_mode == "auto":
            # Back-compat: carbon_tax > 0 without explicit mode → "full"
            self.pricing_mode = "full" if self.carbon_tax > 0.0 else "off"
        # _cost_aware: transport costs and carbon tax enter r_local
        self._cost_aware = self.pricing_mode in ("cost_only", "full")
        # Ensure carbon_tax has a value when cost_aware
        if self._cost_aware and self.carbon_tax <= 0.0:
            self.carbon_tax = float(self.config.get("default_carbon_tax", 80.0))
        # Storage injection fee ($/tCO2) — revenue for storage operators
        self.storage_injection_fee: float = float(
            self.config.get("storage_injection_fee", 5.0))
        # Per-agent w_global overrides for heterogeneous δ_i
        self._per_agent_w_global: Dict[str, float] = self.config.get(
            "per_agent_w_global", {})
        # Institutional mechanism parameters
        self.congestion_threshold: float = float(
            self.config.get("congestion_threshold", 0.8))
        self.k_congestion: float = float(
            self.config.get("k_congestion", 5.0))
        self.injection_obligation_coverage: float = float(
            self.config.get("injection_obligation_coverage", 0.85))
        # Lagrangian multipliers (updated by training code via set_lagrangian_lambdas)
        self._lagrangian_lambdas: Dict[int, float] = {}
        self._lagrangian_target_p_frac: float = 0.90

        # Reward weights
        self.reward_weights = {
            **_DEFAULT_REWARD_WEIGHTS,
            **self.config.get("reward_weights", {}),
        }

        # RNG for observation noise
        self._obs_rng = np.random.default_rng(self.config.get("seed", 42) + 1000)

        # Transport mode list (for agent naming)
        self._transport_modes = self.config["network"].get("transport_modes",
                                                            ["pipeline", "ship"])

        # --- Build agent list ---
        self.possible_agents: List[str] = []
        self._agent_type: Dict[str, str] = {}
        self._agent_comp_id: Dict[str, Any] = {}  # int for emitter/storage, str for transport

        n_emitters = self.config["network"]["num_emitters"]
        for eid in range(n_emitters):
            name = f"emitter_{eid}"
            self.possible_agents.append(name)
            self._agent_type[name] = "emitter"
            self._agent_comp_id[name] = eid

        # Pipeline is passive infrastructure (rule-based settlement),
        # not an RL agent. Only ship/rail are active transport agents.
        self._passive_transport_modes = {"pipeline"}
        for idx, mode_name in enumerate(self._transport_modes):
            if mode_name in self._passive_transport_modes:
                continue  # passive: not an RL agent
            name = f"transport_{mode_name}"
            self.possible_agents.append(name)
            self._agent_type[name] = "transport"
            self._agent_comp_id[name] = mode_name  # store mode name

        n_storage = self.config["network"]["num_storage_sites"]
        for sid in range(n_storage):
            name = f"storage_{sid}"
            self.possible_agents.append(name)
            self._agent_type[name] = "storage"
            self._agent_comp_id[name] = sid

        self.agents = list(self.possible_agents)

        # --- Build observation and action spaces ---
        self.observation_spaces: Dict[str, spaces.Space] = {}
        self.action_spaces: Dict[str, spaces.Space] = {}

        n_transports = len(self._transport_modes)
        n_terminal_buffers = len(self.physical_layer.terminal_buffers)
        fh = self.disruption_forecast_horizon

        # Whether full pricing is active (extra action dims for posted prices/bids)
        self._pricing_active = self.pricing_mode == "full"

        # Transport action dimensions by mode
        # Pipeline is passive (no actions). Ship/Rail: [dispatch_threshold, dest_pref]
        self._transport_act_dims: Dict[str, int] = {}
        for mode_name in self._transport_modes:
            if mode_name in self._passive_transport_modes:
                continue  # passive: no action space
            if self._pricing_active:
                if mode_name == "ship":
                    # [dispatch_threshold, posted_price, destination_pref, quality_threshold, size_preference]
                    self._transport_act_dims[mode_name] = 5
                elif mode_name == "rail":
                    # [dispatch_threshold, posted_price, quality_threshold, train_load_frac]
                    self._transport_act_dims[mode_name] = 4
                else:
                    self._transport_act_dims[mode_name] = 4
            else:
                # Ship/Rail: [dispatch_threshold, destination_pref, quality_threshold]
                self._transport_act_dims[mode_name] = 3

        # Compute max emitter obs/act dims for parameter sharing
        max_emitter_obs_dim = 0
        max_emitter_act_dim = 0
        self._emitter_obs_dims: Dict[str, int] = {}
        self._emitter_act_dims: Dict[str, int] = {}

        # Extra emitter obs dims when pricing active:
        # per connected transport: +1 for posted_price_frac
        # +1 for own last transport cost (normalized)
        _price_extra_per_transport = 1 if self._pricing_active else 0
        _price_extra_global = 1 if self._pricing_active else 0

        for agent in self.possible_agents:
            if self._agent_type[agent] == "emitter":
                eid = self._agent_comp_id[agent]
                routes = self.physical_layer.get_routes_for_emitter(eid)
                n_t_conn = len(self.physical_layer.get_connected_transport_ids(eid))
                n_s_conn = len(self.physical_layer.get_connected_storage_ids(eid))
                # obs: buffer_frac, capture_rate_frac, is_disrupted,
                #   per transport: (cap_frac, is_disrupted, [posted_price_frac]),
                #   per storage: (pressure_frac, is_disrupted),
                #   terminal_buffer_frac (shared), scarcity_ratio, timestep_frac,
                #   [last_transport_cost_frac],
                #   + forecast
                obs_dim = (5
                           + n_t_conn * (3 + _price_extra_per_transport)
                           + n_s_conn * 3
                           + n_terminal_buffers + 3 + 1
                           + _price_extra_global
                           + fh)
                n_routes = len(routes)
                # action: route_logits + send_frac + capture_frac + purification_effort + [bid_premium]
                act_dim = n_routes + 3 + (1 if self._pricing_active else 0)
                self._emitter_obs_dims[agent] = obs_dim
                self._emitter_act_dims[agent] = act_dim
                max_emitter_obs_dim = max(max_emitter_obs_dim, obs_dim)
                max_emitter_act_dim = max(max_emitter_act_dim, act_dim)

        self._max_emitter_obs_dim = max_emitter_obs_dim
        self._max_emitter_act_dim = max_emitter_act_dim

        for agent in self.possible_agents:
            atype = self._agent_type[agent]

            if atype == "emitter":
                self.observation_spaces[agent] = spaces.Box(
                    low=-1.0, high=2.0, shape=(max_emitter_obs_dim,), dtype=np.float32
                )
                self.action_spaces[agent] = spaces.Box(
                    low=0.0, high=1.0, shape=(max_emitter_act_dim,), dtype=np.float32
                )

            elif atype == "transport":
                mode_name = self._agent_comp_id[agent]
                # Observation: [avail_capacity_frac, utilization, is_disrupted,
                #   disruption_remaining_frac, in_transit_frac,
                #   terminal_buffer_frac, scarcity_ratio,
                #   storage_pressure_summary,  ← NEW
                #   timestep_frac,
                #   [last_revenue_frac, posted_price_frac],
                #   + forecast]
                transport_obs_extra = 2 if self._pricing_active else 0
                obs_dim = 12 + transport_obs_extra + fh
                self.observation_spaces[agent] = spaces.Box(
                    low=-1.0, high=2.0, shape=(obs_dim,), dtype=np.float32
                )
                act_dim = self._transport_act_dims[mode_name]
                self.action_spaces[agent] = spaces.Box(
                    low=0.0, high=1.0, shape=(act_dim,), dtype=np.float32
                )

            elif atype == "storage":
                # Observation: [pressure_frac, fill_frac, injectivity,
                #   max_injectable_frac, is_disrupted, disruption_remaining_frac,
                #   per other storage: (pressure_frac),
                #   cross_well_contribution,     ← NEW
                #   injection_obligation_frac,   ← NEW
                #   timestep_frac]
                n_other = n_storage - 1
                obs_dim = 8 + n_other + 3 + 1
                self.observation_spaces[agent] = spaces.Box(
                    low=-1.0, high=2.0, shape=(obs_dim,), dtype=np.float32
                )
                act_dim = 2  # injection rate fraction + purity target adjustment
                self.action_spaces[agent] = spaces.Box(
                    low=0.0, high=1.0, shape=(act_dim,), dtype=np.float32
                )

        # Pre-generate disruption schedule for forecasting
        self._pregenerated_disruptions: List[Any] = []
        self._pregenerate_disruption_schedule()

        # Episode state
        self.timestep = 0
        self._cumulative_rewards: Dict[str, float] = {}
        self._episode_stats: Dict[str, float] = {}

        # Last outcome (for observations after step)
        self._last_outcome: Optional[PhysicalOutcome] = None

        # Reward normalization
        self._reward_running_mean: Dict[str, float] = {a: 1.0 for a in self.possible_agents}
        self._reward_running_var: Dict[str, float] = {a: 1.0 for a in self.possible_agents}
        self._reward_norm_momentum = 0.99

    # ------------------------------------------------------------------
    # Factory: create from YAML case file
    # ------------------------------------------------------------------

    @classmethod
    def from_case(cls, case_path: str, **overrides) -> "CCUSEnv":
        """Create a CCUSEnv from a YAML case definition file.

        Args:
            case_path: Path to the YAML case file (e.g. 'cases/teesside_uk.yaml').
            **overrides: Any key-value pairs to override in the loaded config.
                Example: from_case("cases/teesside_uk.yaml", seed=123)

        Returns:
            A fully initialised CCUSEnv instance.
        """
        config = load_case(case_path)
        # Apply flat overrides
        for key, value in overrides.items():
            if key in config:
                config[key] = value
            elif key in config.get("mechanism", {}):
                config["mechanism"][key] = value
            elif key in config.get("disruption", {}):
                config["disruption"][key] = value
            elif key in config.get("simulation", {}):
                config["simulation"][key] = value
            else:
                config[key] = value
        return cls(config)

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """Reset the environment to initial state."""
        if seed is not None:
            self._obs_rng = np.random.default_rng(seed + 1000)
            self.disruption_gen = DisruptionGenerator(
                network=self.network,
                scenario_family=self.config.get("disruption", {}).get("scenario_family", "T"),
                severity_kappa=self.config.get("disruption", {}).get("severity", 0.5),
                cross_correlation=self.config.get("disruption", {}).get("cross_correlation", 0.0),
                seed=seed,
                gamma=self.gamma,
            )
            self._pregenerate_disruption_schedule()

        self.physical_layer.reset()
        self.network.reset()
        self.agents = list(self.possible_agents)
        self.timestep = 0
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self._episode_stats = {
            "total_stored": 0.0,
            "total_vented": 0.0,
            "total_captured": 0.0,
            "pressure_violations": 0,
            "quality_violations": 0,
            "transport_cost": 0.0,
            "capture_cost": 0.0,
            "energy_use": 0.0,
            "disruption_events": 0,
            "total_transport_revenue": 0.0,
        }
        # Per-agent cumulative stats for IR / profit analysis
        self._per_agent_stats: Dict[str, Dict[str, float]] = {}
        for agent in self.agents:
            self._per_agent_stats[agent] = {
                "sent": 0.0, "vented": 0.0, "captured": 0.0,
                "transport_cost": 0.0, "capture_cost": 0.0,
                "capture_energy": 0.0, "revenue": 0.0,
                "injected": 0.0, "pressure_violations": 0,
                "quality_violations": 0,
                "reward_cumulative": 0.0,
            }
        self._last_outcome = None
        # Track last posted prices and revenue for observations
        self._last_posted_prices: Dict[str, float] = {m: 0.0 for m in self._transport_modes}
        self._last_transport_revenue: Dict[str, float] = {m: 0.0 for m in self._transport_modes}
        self._last_emitter_cost: Dict[int, float] = {}
        self._refresh_extreme_context()
        self._reward_running_mean = {a: 1.0 for a in self.agents}
        self._reward_running_var = {a: 1.0 for a in self.agents}

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _refresh_extreme_context(self) -> None:
        carbon_mult = 1.0
        electricity_mult = 1.0
        for scenario in self.extreme_scenarios:
            start = int(scenario.get("start_timestep", -1))
            duration = int(scenario.get("duration", 0))
            if start <= self.timestep < start + duration:
                carbon_mult *= float(scenario.get("carbon_tax_multiplier", 1.0))
                electricity_mult *= float(
                    scenario.get("electricity_price_multiplier", 1.0)
                )
        self._active_extreme_context = {
            "carbon_tax_multiplier": carbon_mult,
            "electricity_price_multiplier": electricity_mult,
        }

    def get_economic_context(self) -> Dict[str, float]:
        return {
            "carbon_tax": self._base_carbon_tax
            * self._active_extreme_context.get("carbon_tax_multiplier", 1.0),
            "electricity_price": self._base_electricity_price
            * self._active_extreme_context.get("electricity_price_multiplier", 1.0),
            "capture_subsidy": self.capture_subsidy,
            "storage_credit": self.storage_credit,
            "offspec_penalty": self.offspec_penalty,
        }

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict],
    ]:
        """Execute one timestep (1 month) of the CCUS network.

        Execution order:
        1. Generate and apply disruptions (to both network and physical layer)
        2. Parse agent actions into decision intents
        3. Nomination & Settlement (pro-rata rationing when oversubscribed)
        4. Physical execution via PhysicalLayer.settle()
        5. Compute rewards from physical outcomes
        6. Build observations from physical state
        """

        self._refresh_extreme_context()

        # ------ 1. Disruptions ------
        events = self.disruption_gen.generate_disruptions(self.timestep)
        # Apply to legacy network (for disruption generator state tracking)
        self.disruption_gen.apply_events(events)
        # Apply to physical layer
        for evt in events:
            self.physical_layer.apply_disruption(
                evt.target_type, evt.target_id, evt.severity, evt.duration
            )
        self._episode_stats["disruption_events"] += len(events)

        # ------ 2. Parse agent actions into decision intents ------
        decisions = self._parse_decisions(actions)

        # ------ 3 & 4. Physical execution (includes nomination settlement) ------
        outcome = self.physical_layer.settle(
            decisions,
            cross_well_scale=self.effective_cross_well_scale,
            pressure_limit_scale=self.pressure_limit_scale,
        )
        self._last_outcome = outcome

        # Update episode stats
        self._episode_stats["total_stored"] += outcome.total_stored
        self._episode_stats["total_vented"] += outcome.total_vented
        self._episode_stats["total_captured"] += outcome.total_captured
        self._episode_stats["transport_cost"] += outcome.transport_cost
        self._episode_stats["capture_cost"] += outcome.total_capture_cost
        self._episode_stats["energy_use"] += outcome.total_energy_use
        self._episode_stats["pressure_violations"] += outcome.pressure_violations
        self._episode_stats["quality_violations"] += outcome.quality_violations
        self._episode_stats["total_transport_revenue"] += sum(
            outcome.transport_revenue.values()
        )

        # Cache pricing info for observations
        self._last_posted_prices.update(outcome.transport_posted_prices)
        self._last_transport_revenue.update(outcome.transport_revenue)
        self._last_emitter_cost = dict(outcome.emitter_transport_cost)

        # Check for pressure-triggered disruptions
        pressure_triggers = self.physical_layer.check_pressure_triggered()
        for sid, sev, dur in pressure_triggers:
            self.physical_layer.apply_disruption("storage", sid, sev, dur)
            # Also apply to legacy network for disruption gen
            self.network.storage_sites[sid].apply_disruption(sev, dur)

        # ------ 5. Compute rewards ------
        rewards = self._compute_rewards(outcome)

        # ------ 6. Advance timestep, termination ------
        self.timestep += 1
        terminated = self.timestep >= self.episode_length
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        observations = {agent: self._get_observation(agent) for agent in self.agents}

        step_info = {
            "timestep": self.timestep,
            "step_stored": outcome.total_stored,
            "step_vented": outcome.total_vented,
            "step_captured": outcome.total_captured,
            "step_pressure_violations": outcome.pressure_violations,
            "step_quality_violations": outcome.quality_violations,
            "step_energy_use": outcome.total_energy_use,
            "disruption_events": len(events) + len(pressure_triggers),
            "scarcity_ratios": dict(outcome.scarcity_ratios),
            "terminal_buffers": dict(outcome.terminal_buffer_levels),
            "economic_context": self.get_economic_context(),
        }
        if terminated:
            step_info["episode_stats"] = dict(self._episode_stats)

        infos = {agent: step_info for agent in self.agents}

        for agent in self.agents:
            self._cumulative_rewards[agent] = (
                self._cumulative_rewards.get(agent, 0.0) + rewards[agent]
            )

        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Decision parsing: agent actions -> decision intents
    # ------------------------------------------------------------------

    def _parse_decisions(self, actions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Parse agent actions into decision intents for the physical layer.

        Returns:
            Dict with keys:
                "emitter_nominations": {eid: [(transport_idx, storage_id, volume), ...]}
                "transport_thresholds": {mode_name: threshold_frac}
                "storage_injection_fracs": {sid: frac}
                "transport_posted_prices": {mode_name: price}  (if pricing active)
                "emitter_bids": {eid: bid}                     (if pricing active)
                "transport_params": {mode_name: {extra params}} (if pricing active)
        """
        emitter_nominations: Dict[int, List[Tuple[int, int, float]]] = {}
        emitter_capture_fracs: Dict[int, float] = {}
        emitter_purification_efforts: Dict[int, float] = {}
        transport_thresholds: Dict[str, float] = {}
        transport_quality_thresholds: Dict[str, float] = {}
        storage_injection_fracs: Dict[int, float] = {}
        storage_quality_targets: Dict[int, float] = {}
        transport_posted_prices: Dict[str, float] = {}
        emitter_bids: Dict[int, float] = {}
        transport_extra_params: Dict[str, Dict[str, float]] = {}
        economic_context = self.get_economic_context()
        current_carbon_tax = economic_context["carbon_tax"]

        for agent in self.agents:
            atype = self._agent_type[agent]
            action = actions.get(agent, np.zeros(self.action_space(agent).shape))

            if atype == "emitter":
                eid = self._agent_comp_id[agent]
                emitter = self.physical_layer.emitters[eid]
                routes = self.physical_layer.get_routes_for_emitter(eid)
                n_routes = len(routes)

                # Action layout: route_logits..., send_frac, capture_frac, purification_effort, [bid_premium]
                raw_vol_frac = float(np.clip(
                    action[n_routes] if len(action) > n_routes else action[-1],
                    0.0, 1.0
                ))
                raw_capture_frac = float(np.clip(
                    action[n_routes + 1] if len(action) > n_routes + 1 else 1.0,
                    0.0, 1.0,
                ))
                raw_purification = float(np.clip(
                    action[n_routes + 2] if len(action) > n_routes + 2 else 0.0,
                    0.0, 1.0,
                ))

                # Action remapping: minimum 20%, 50% if buffer > half full
                buf_frac_now = emitter.buffer_level / max(emitter.buffer_capacity, 1e-9)
                min_vol = 0.5 if buf_frac_now > 0.5 else 0.2
                vol_frac = min_vol + (1.0 - min_vol) * raw_vol_frac
                capture_frac = 0.1 + 0.9 * raw_capture_frac

                available = emitter.get_available(capture_frac)
                total_volume = vol_frac * available
                emitter_capture_fracs[eid] = capture_frac
                emitter_purification_efforts[eid] = raw_purification

                # Route fractions: scale actions to [0,1] then normalize
                # Using temperature-scaled softmax (temp=5) so [0,1] actions
                # can express near-pure routing (e.g., [1,0] -> 99.3%/0.7%)
                route_raw = action[:n_routes].astype(np.float64)
                temperature = 5.0
                route_logits = route_raw * temperature
                route_logits = route_logits - np.max(route_logits)
                exp_logits = np.exp(route_logits)
                route_fracs = exp_logits / (np.sum(exp_logits) + 1e-8)

                routing = []
                for i, (tid, sid) in enumerate(routes):
                    vol = total_volume * route_fracs[i]
                    routing.append((tid, sid, vol))
                emitter_nominations[eid] = routing

                # Bid premium (if pricing active)
                if self._pricing_active:
                    bid_premium_idx = n_routes + 3
                    raw_bid_premium = float(np.clip(
                        action[bid_premium_idx] if len(action) > bid_premium_idx else 0.0,
                        0.0, 1.0
                    ))
                    # Bid = expected posted price + premium * carbon_tax
                    # For simplicity, we compute the bid as a willingness-to-pay
                    # that the settlement mechanism uses.
                    # actual_bid = avg_posted_price + raw_bid_premium * carbon_tax
                    avg_posted = np.mean(list(self._last_posted_prices.values())) if self._last_posted_prices else 0.0
                    bid = avg_posted + raw_bid_premium * current_carbon_tax
                    emitter_bids[eid] = bid

            elif atype == "transport":
                mode_name = self._agent_comp_id[agent]
                # Pipeline is passive — should not appear as an agent
                # Ship/Rail: [dispatch_threshold, destination_pref]

                if self._pricing_active:
                    if mode_name == "ship":
                        # [dispatch_threshold, posted_price, destination_pref, size_preference]
                        raw_dispatch = float(np.clip(action[0], 0.0, 1.0))
                        raw_price = float(np.clip(
                            action[1] if len(action) > 1 else 0.5, 0.0, 1.0))
                        raw_dest_pref = float(np.clip(
                            action[2] if len(action) > 2 else 0.5, 0.0, 1.0))
                        raw_quality = float(np.clip(
                            action[3] if len(action) > 3 else 0.5, 0.0, 1.0))
                        raw_size_pref = float(np.clip(
                            action[4] if len(action) > 4 else 0.5, 0.0, 1.0))

                        transport_thresholds[mode_name] = 0.3 + 0.7 * raw_dispatch
                        transport_posted_prices[mode_name] = raw_price * 2.0 * current_carbon_tax
                        transport_quality_thresholds[mode_name] = 0.85 + 0.12 * raw_quality
                        transport_extra_params[mode_name] = {
                            "dispatch_threshold": raw_dispatch,
                            "destination_pref": raw_dest_pref,
                            "size_preference": raw_size_pref,
                        }

                    elif mode_name == "rail":
                        # [dispatch_threshold, posted_price, train_load_frac]
                        raw_dispatch = float(np.clip(action[0], 0.0, 1.0))
                        raw_price = float(np.clip(
                            action[1] if len(action) > 1 else 0.5, 0.0, 1.0))
                        raw_quality = float(np.clip(
                            action[2] if len(action) > 2 else 0.5, 0.0, 1.0))
                        raw_load_frac = float(np.clip(
                            action[3] if len(action) > 3 else 1.0, 0.0, 1.0))

                        transport_thresholds[mode_name] = 0.3 + 0.7 * raw_dispatch
                        transport_posted_prices[mode_name] = raw_price * 2.0 * current_carbon_tax
                        transport_quality_thresholds[mode_name] = 0.85 + 0.12 * raw_quality
                        transport_extra_params[mode_name] = {
                            "dispatch_threshold": raw_dispatch,
                            "train_load_frac": raw_load_frac,
                        }
                    else:
                        raw_threshold = float(np.clip(action[0], 0.0, 1.0))
                        transport_thresholds[mode_name] = 0.3 + 0.7 * raw_threshold
                        raw_price = float(np.clip(
                            action[1] if len(action) > 1 else 0.5, 0.0, 1.0))
                        transport_posted_prices[mode_name] = raw_price * 2.0 * current_carbon_tax
                        raw_quality = float(np.clip(
                            action[2] if len(action) > 2 else 0.5, 0.0, 1.0))
                        transport_quality_thresholds[mode_name] = 0.85 + 0.12 * raw_quality
                else:
                    # Ship/Rail: [dispatch_threshold, destination_pref, quality_threshold]
                    raw_dispatch = float(np.clip(action[0], 0.0, 1.0))
                    raw_dest_pref = float(np.clip(
                        action[1] if len(action) > 1 else 0.5, 0.0, 1.0))
                    raw_quality = float(np.clip(
                        action[2] if len(action) > 2 else 0.5, 0.0, 1.0))
                    # Map dispatch_threshold: [0,1] -> [0.2, 0.9]
                    transport_thresholds[mode_name] = 0.2 + 0.7 * raw_dispatch
                    transport_quality_thresholds[mode_name] = 0.85 + 0.12 * raw_quality
                    transport_extra_params[mode_name] = {
                        "dispatch_threshold": 0.2 + 0.7 * raw_dispatch,
                        "destination_pref": raw_dest_pref,
                    }

            elif atype == "storage":
                sid = self._agent_comp_id[agent]
                raw_frac = float(np.clip(action[0], 0.0, 1.0))
                raw_quality = float(np.clip(
                    action[1] if len(action) > 1 else 0.5,
                    0.0, 1.0,
                ))
                # Hard safety cap only: no injection above 95% pressure
                # (geological integrity, non-negotiable). Lagrangian multiplier
                # handles learned constraint satisfaction below 95%.
                site = self.physical_layer.storage_sites.get(sid)
                if site is not None:
                    p_frac = site.current_pressure / max(site.pressure_limit, 1.0)
                    if p_frac > 0.95:
                        raw_frac = 0.0
                storage_injection_fracs[sid] = raw_frac
                if site is not None:
                    storage_quality_targets[sid] = np.clip(
                        site.min_purity + 0.08 * (raw_quality - 0.5),
                        0.85,
                        0.995,
                    )

        # Pipeline passive: always accept all nominations (rule-based settlement)
        if "pipeline" in self._transport_modes:
            transport_thresholds["pipeline"] = 1.0
            transport_quality_thresholds["pipeline"] = 0.0

        result: Dict[str, Any] = {
            "emitter_nominations": emitter_nominations,
            "emitter_capture_fracs": emitter_capture_fracs,
            "emitter_purification_efforts": emitter_purification_efforts,
            "transport_thresholds": transport_thresholds,
            "transport_quality_thresholds": transport_quality_thresholds,
            "storage_injection_fracs": storage_injection_fracs,
            "storage_quality_targets": storage_quality_targets,
            "economic_context": economic_context,
        }
        if self._pricing_active:
            result["transport_posted_prices"] = transport_posted_prices
            result["emitter_bids"] = emitter_bids
            result["transport_params"] = transport_extra_params
        return result

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_rewards(self, outcome: PhysicalOutcome) -> Dict[str, float]:
        """Compute role-factored CTDE rewards from physical outcomes."""
        w = self.reward_weights
        default_w_global = w["w_global"]
        default_w_local = w["w_local"]
        economic_context = self.get_economic_context()
        carbon_tax = economic_context["carbon_tax"]
        electricity_price = economic_context["electricity_price"]

        # System reward (shared)
        # Two-stage pressure barrier: mild from 75%, steep from 90%
        smooth_pressure_penalty = 0.0
        for sid, site in self.physical_layer.storage_sites.items():
            p_frac = site.current_pressure / max(site.pressure_limit, 1.0)
            if p_frac > 0.75:
                # Stage 1: mild quadratic ramp from 75%
                smooth_pressure_penalty += (p_frac - 0.75) ** 2
            if p_frac > 0.90:
                # Stage 2: steep additional penalty from 90%
                smooth_pressure_penalty += 5.0 * (p_frac - 0.90) ** 2
            if p_frac > 1.0:
                # Stage 3: hard barrier — massive penalty for actual violations
                smooth_pressure_penalty += 50.0 * (p_frac - 1.0) ** 2
        r_system = (
            w["sys_co2_stored"] * outcome.total_stored
            + w["sys_co2_vented"] * outcome.total_vented
            + w["sys_pressure_violation"] * smooth_pressure_penalty
            + w["sys_energy_use"] * outcome.total_energy_use * electricity_price
            + w["sys_quality_violation"] * outcome.quality_violations
        )

        rewards: Dict[str, float] = {}
        for agent in self.agents:
            atype = self._agent_type[agent]
            comp_id = self._agent_comp_id[agent]

            # Per-agent δ support: look up individual w_global override
            w_global_i = self._per_agent_w_global.get(agent, default_w_global)
            w_local_i = 1.0 - w_global_i

            if atype == "emitter":
                buf_frac = outcome.emitter_buffer_frac.get(comp_id, 0.0)
                r_local = (
                    w["emitter_co2_sent"] * outcome.emitter_sent.get(comp_id, 0.0)
                    + w["emitter_co2_vented"] * outcome.emitter_vented.get(comp_id, 0.0)
                    + w["emitter_buffer_penalty"] * buf_frac
                    + w["emitter_buffer_bonus"] * (1.0 - buf_frac)
                    + w["emitter_purity_bonus"] * outcome.emitter_effective_purity.get(comp_id, 0.0)
                )
                # Cost-aware terms (cost_only or full pricing mode)
                if self._cost_aware:
                    cost_norm = carbon_tax if carbon_tax > 0 else 1.0
                    transport_cost_paid = outcome.emitter_transport_cost.get(comp_id, 0.0)
                    r_local += w["emitter_transport_cost"] * transport_cost_paid / cost_norm
                    # Carbon tax on own venting
                    own_vented = outcome.emitter_vented.get(comp_id, 0.0)
                    r_local += w["emitter_carbon_tax_vent"] * carbon_tax * own_vented / cost_norm

                capture_cost = outcome.emitter_capture_cost.get(comp_id, 0.0)
                capture_energy = outcome.emitter_capture_energy.get(comp_id, 0.0)
                r_local += w["emitter_capture_cost"] * (
                    capture_cost - self.capture_subsidy * outcome.emitter_captured.get(comp_id, 0.0)
                )
                r_local += w["emitter_capture_energy"] * capture_energy * electricity_price

                # Congestion pricing penalty (institutional mechanism)
                congestion_surcharge = outcome.pipeline_congestion_surcharge.get(comp_id, 0.0)
                if congestion_surcharge > 0:
                    cost_norm = carbon_tax if carbon_tax > 0 else 1.0
                    r_local += w["emitter_congestion_penalty"] * congestion_surcharge / max(cost_norm, 1.0)

                # Overflow attribution penalty
                overflow_attributed = outcome.overflow_attributed_emitter.get(comp_id, 0.0)
                if overflow_attributed > 0:
                    r_local += w["emitter_overflow_penalty"] * overflow_attributed

                # Per-agent stats
                pa = self._per_agent_stats[agent]
                pa["sent"] += outcome.emitter_sent.get(comp_id, 0.0)
                pa["vented"] += outcome.emitter_vented.get(comp_id, 0.0)
                pa["transport_cost"] += outcome.emitter_transport_cost.get(comp_id, 0.0)
                pa["capture_cost"] += capture_cost
                pa["capture_energy"] += capture_energy

            elif atype == "transport":
                mode_name = comp_id  # string
                r_local = (
                    w["transport_co2_delivered"] * outcome.transport_delivered.get(mode_name, 0.0)
                    + w["transport_utilization"] * outcome.transport_utilization.get(mode_name, 0.0)
                    + w["transport_co2_rejected"] * outcome.transport_rejected.get(mode_name, 0.0)
                    + w["transport_quality_bonus"] * (
                        1.0 - np.mean(list(outcome.storage_quality_penalty.values()))
                        if outcome.storage_quality_penalty else 1.0
                    )
                )
                # Revenue term (cost_only: use base_cost revenue; full: use dynamic revenue)
                if self._cost_aware:
                    cost_norm = carbon_tax if carbon_tax > 0 else 1.0
                    revenue = outcome.transport_revenue.get(mode_name, 0.0)
                    r_local += w["transport_revenue"] * revenue / cost_norm

                # Per-agent stats
                pa = self._per_agent_stats[agent]
                pa["revenue"] += outcome.transport_revenue.get(mode_name, 0.0)

            elif atype == "storage":
                injected = outcome.storage_injected.get(comp_id, 0.0)
                # Two-stage per-site pressure barrier: mild from 75%, steep from 90%
                site = self.physical_layer.storage_sites.get(comp_id)
                if site is not None:
                    p_frac = site.current_pressure / max(site.pressure_limit, 1.0)
                    local_pressure_penalty = 0.0
                    if p_frac > 0.75:
                        local_pressure_penalty += (p_frac - 0.75) ** 2
                    if p_frac > 0.90:
                        local_pressure_penalty += 5.0 * (p_frac - 0.90) ** 2
                    if p_frac > 1.0:
                        local_pressure_penalty += 50.0 * (p_frac - 1.0) ** 2
                else:
                    local_pressure_penalty = 0.0
                r_local = (
                    w["storage_co2_injected"] * injected
                    + w["storage_pressure_violation"] * local_pressure_penalty
                    + w["storage_pressure_margin"] * outcome.storage_pressure_margin.get(comp_id, 0.0)
                    + w["storage_quality_penalty"] * outcome.storage_quality_penalty.get(comp_id, 0.0)
                )
                # Storage injection fee revenue (cost_only or full)
                if self._cost_aware:
                    cost_norm = carbon_tax if carbon_tax > 0 else 1.0
                    r_local += (
                        self.storage_injection_fee + self.storage_credit
                    ) * injected / cost_norm

                # Injection obligation penalty
                target_rate = site.max_injection_rate * self.injection_obligation_coverage if site else 0.0
                if target_rate > 1e-9:
                    shortfall = max(0.0, target_rate - injected) / target_rate
                    r_local += w["storage_obligation_penalty"] * shortfall

                # Overflow attribution penalty (storage couldn't inject what was delivered)
                overflow_at_storage = outcome.overflow_attributed_storage.get(comp_id, 0.0)
                if overflow_at_storage > 0:
                    r_local += w["storage_overflow_penalty"] * overflow_at_storage

                # Lagrangian constraint penalty (learned dual variable)
                lagrangian_lambda = self._lagrangian_lambdas.get(comp_id, 0.0)
                if lagrangian_lambda > 0.0 and site is not None:
                    constraint_viol = max(0.0, p_frac - self._lagrangian_target_p_frac)
                    r_local -= lagrangian_lambda * constraint_viol

                # Per-agent stats
                pa = self._per_agent_stats[agent]
                pa["injected"] += injected
                is_violated = outcome.storage_pressure_violation.get(comp_id, False)
                pa["pressure_violations"] += int(is_violated)
                pa["quality_violations"] += int(
                    outcome.storage_quality_violation.get(comp_id, False)
                )

            else:
                r_local = 0.0

            raw_reward = w_global_i * r_system + w_local_i * r_local

            # Use raw reward directly — PPO advantage normalization handles scale.
            # Reward normalization was washing out constraint penalty signals.
            rewards[agent] = raw_reward

        return rewards

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_observation(self, agent: str) -> np.ndarray:
        atype = self._agent_type[agent]
        comp_id = self._agent_comp_id[agent]
        t_frac = self.timestep / max(self.episode_length, 1)

        if atype == "emitter":
            return self._obs_emitter(comp_id, t_frac)
        elif atype == "transport":
            return self._obs_transport(comp_id, t_frac)
        elif atype == "storage":
            return self._obs_storage(comp_id, t_frac)
        else:
            raise ValueError(f"Unknown agent type: {atype}")

    def _obs_emitter(self, eid: int, t_frac: float) -> np.ndarray:
        emitter = self.physical_layer.emitters[eid]
        state = emitter.get_state()
        econ = self.get_economic_context()

        obs = [
            state["buffer_frac"],
            state["current_capture_rate"] / max(state["max_capture_rate"], 1e-9),
            state.get("capture_fraction", 1.0),
            state.get("effective_purity", getattr(emitter, "base_purity", 0.9)),
            state["is_disrupted"],
        ]

        # Connected transports (with noise based on alpha)
        for tid in self.physical_layer.get_connected_transport_ids(eid):
            mode_name = self.physical_layer.get_mode_name(tid)
            t_state = self.physical_layer.get_transport_state(mode_name)
            cap_frac = t_state["available_capacity"] / max(t_state["capacity"], 1e-9)
            noise = self._obs_rng.normal(0, self.alpha * 0.1)
            obs.append(np.clip(cap_frac + noise, 0.0, 1.5))
            obs.append(t_state["is_disrupted"])
            obs.append(np.clip(self._last_posted_prices.get(mode_name, 0.0) / max(econ["carbon_tax"] * 2.0, 1.0), 0.0, 2.0))
            # Posted price (normalized by 2*carbon_tax)
            if self._pricing_active:
                price = self._last_posted_prices.get(mode_name, 0.0)
                price_max = 2.0 * econ["carbon_tax"] if econ["carbon_tax"] > 0 else 1.0
                obs.append(np.clip(price / price_max, 0.0, 1.5))

        # Connected storage sites (with noise)
        for sid in self.physical_layer.get_connected_storage_ids(eid):
            s_state = self.physical_layer.storage_sites[sid].get_state()
            noise = self._obs_rng.normal(0, self.alpha * 0.1)
            obs.append(np.clip(s_state["pressure_frac"] + noise, 0.0, 1.5))
            obs.append(np.clip(s_state.get("last_inlet_purity", 1.0), 0.0, 1.0))
            obs.append(s_state["is_disrupted"])

        # Terminal buffer fill levels (shared resources)
        for buf_id in sorted(self.physical_layer.terminal_buffers.keys()):
            buf = self.physical_layer.terminal_buffers[buf_id]
            obs.append(buf.get_fill_frac())

        # Scarcity ratio (from last outcome, or 1.0 if no outcome yet)
        if self._last_outcome is not None:
            # Average scarcity across modes
            ratios = list(self._last_outcome.scarcity_ratios.values())
            avg_scarcity = np.mean(ratios) if ratios else 1.0
        else:
            avg_scarcity = 1.0
        obs.append(avg_scarcity)
        obs.append(np.clip(econ["carbon_tax"] / max(self._base_carbon_tax if self._base_carbon_tax > 0 else 100.0, 1.0), 0.0, 2.0))
        obs.append(np.clip(econ["electricity_price"] / max(self._base_electricity_price, 1.0), 0.0, 2.0))

        obs.append(t_frac)

        # Last transport cost paid (normalized)
        if self._pricing_active:
            cost_norm = econ["carbon_tax"] if econ["carbon_tax"] > 0 else 1.0
            last_cost = self._last_emitter_cost.get(eid, 0.0)
            obs.append(np.clip(last_cost / cost_norm, 0.0, 2.0))

        # Disruption forecast
        connected_tids = self.physical_layer.get_connected_transport_ids(eid)
        forecast = self._get_disruption_forecast(
            "transport", connected_tids[0] if connected_tids else -1
        )
        obs.extend(forecast.tolist())

        # Pad to max emitter obs dim
        result = np.zeros(self._max_emitter_obs_dim, dtype=np.float32)
        result[:len(obs)] = obs
        return result

    def _obs_transport(self, mode_name: str, t_frac: float) -> np.ndarray:
        t_state = self.physical_layer.get_transport_state(mode_name)
        tid = self.physical_layer.get_mode_index(mode_name)
        econ = self.get_economic_context()

        obs = [
            t_state["available_capacity"] / max(t_state["capacity"], 1e-9),
            t_state["utilization"],
            t_state["is_disrupted"],
            t_state["disruption_remaining_time"] / 12.0,
            t_state["in_transit_total"] / max(t_state["capacity"], 1e-9),
        ]

        # Terminal buffer fill level (if this mode uses one)
        if mode_name in self.physical_layer.terminal_buffers:
            buf = self.physical_layer.terminal_buffers[mode_name]
            obs.append(buf.get_fill_frac())
        else:
            obs.append(0.0)  # pipeline has no buffer

        # Scarcity ratio for this mode
        if self._last_outcome is not None:
            scarcity = self._last_outcome.scarcity_ratios.get(mode_name, 1.0)
        else:
            scarcity = 1.0
        obs.append(scarcity)

        # Storage pressure summary (avg p_frac across all storage sites)
        avg_p_frac = np.mean([
            s.current_pressure / max(s.pressure_limit, 1.0)
            for s in self.physical_layer.storage_sites.values()
        ]) if self.physical_layer.storage_sites else 0.0
        obs.append(np.clip(avg_p_frac, 0.0, 1.5))
        avg_purity = np.mean([
            e.last_effective_purity for e in self.physical_layer.emitters.values()
        ]) if self.physical_layer.emitters else 1.0
        obs.append(np.clip(avg_purity, 0.0, 1.0))
        obs.append(np.clip(econ["carbon_tax"] / max(self._base_carbon_tax if self._base_carbon_tax > 0 else 100.0, 1.0), 0.0, 2.0))
        obs.append(np.clip(econ["electricity_price"] / max(self._base_electricity_price, 1.0), 0.0, 2.0))

        obs.append(t_frac)

        # Pricing observations (revenue from last step, own posted price)
        if self._pricing_active:
            cost_norm = econ["carbon_tax"] if econ["carbon_tax"] > 0 else 1.0
            revenue = self._last_transport_revenue.get(mode_name, 0.0)
            obs.append(np.clip(revenue / cost_norm, 0.0, 2.0))
            price = self._last_posted_prices.get(mode_name, 0.0)
            price_max = 2.0 * econ["carbon_tax"] if econ["carbon_tax"] > 0 else 1.0
            obs.append(np.clip(price / price_max, 0.0, 1.5))

        # Disruption forecast
        forecast = self._get_disruption_forecast("transport", tid)
        obs.extend(forecast.tolist())

        return np.array(obs, dtype=np.float32)

    def _obs_storage(self, sid: int, t_frac: float) -> np.ndarray:
        site = self.physical_layer.storage_sites[sid]
        state = site.get_state()
        obs = [
            state["pressure_frac"],
            state["fill_frac"],
            state["current_injectivity"],
            state["max_injectable"] / max(site.max_injection_rate, 1e-9),
            state.get("last_inlet_purity", 1.0),
            np.clip(state.get("last_quality_penalty", 0.0), 0.0, 1.0),
            state["is_disrupted"],
            state["disruption_remaining_time"] / 12.0,
        ]

        # Other storage sites' pressure (for cross-well awareness)
        for other_sid in sorted(self.physical_layer.storage_sites.keys()):
            if other_sid != sid:
                other_state = self.physical_layer.storage_sites[other_sid].get_state()
                noise = self._obs_rng.normal(0, self.alpha * 0.1)
                obs.append(np.clip(other_state["pressure_frac"] + noise, 0.0, 1.5))

        # Cross-well pressure contribution: how much other sites' injection
        # contributed to this site's pressure (normalized by pressure limit)
        cross_well_contrib = 0.0
        if self._last_outcome is not None:
            for other_sid in self.physical_layer.storage_sites:
                if other_sid != sid:
                    other_injected = self._last_outcome.storage_injected.get(other_sid, 0.0)
                    cross_well_contrib += (
                        site.cross_well_coeff * other_injected
                        / max(site.pressure_limit, 1.0)
                    )
        obs.append(np.clip(cross_well_contrib, 0.0, 1.0))

        # Injection obligation fulfillment fraction
        obligation_coverage = self.config.get("injection_obligation_coverage", 0.85)
        target_rate = site.max_injection_rate * obligation_coverage
        if self._last_outcome is not None and target_rate > 1e-9:
            actual_injected = self._last_outcome.storage_injected.get(sid, 0.0)
            obligation_frac = actual_injected / target_rate
        else:
            obligation_frac = 1.0  # no data yet, assume fulfilled
        obs.append(np.clip(obligation_frac, 0.0, 2.0))
        obs.append(np.clip(site.min_purity, 0.0, 1.0))

        obs.append(t_frac)
        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Disruption forecast (structural alpha mechanism)
    # ------------------------------------------------------------------

    def _pregenerate_disruption_schedule(self) -> None:
        """Pre-roll disruptions for forecasting."""
        from ccus_gym.sim.disruptions import DisruptionGenerator
        preview_rng_seed = self.config.get("seed", 42) + 7777
        preview_gen = DisruptionGenerator(
            network=self.network,
            scenario_family=self.config.get("disruption", {}).get("scenario_family", "T"),
            severity_kappa=self.config.get("disruption", {}).get("severity", 0.5),
            cross_correlation=self.config.get("disruption", {}).get("cross_correlation", 0.0),
            seed=preview_rng_seed,
            gamma=self.gamma,
        )
        schedule = []
        for t in range(self.episode_length):
            evts = preview_gen.generate_disruptions(t)
            for evt in evts:
                schedule.append((evt.timestep, evt.target_type, evt.target_id,
                                 evt.severity, evt.duration))
        self._pregenerated_disruptions = schedule

    def _get_disruption_forecast(
        self, target_type: str, target_id: int
    ) -> np.ndarray:
        fh = self.disruption_forecast_horizon
        if fh == 0:
            return np.array([], dtype=np.float32)

        forecast = np.zeros(fh, dtype=np.float32)
        for t_start, ttype, tid, sev, dur in self._pregenerated_disruptions:
            if ttype != target_type or tid != target_id:
                continue
            for k in range(fh):
                future_t = self.timestep + k + 1
                if t_start <= future_t < t_start + dur:
                    forecast[k] = min(1.0, sev)
        return forecast

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> Optional[str]:
        if self.render_mode != "human":
            return None

        lines = [f"\n=== CCUS Network  t={self.timestep}/{self.episode_length} ==="]

        lines.append("\nEmitters:")
        for eid in sorted(self.physical_layer.emitters.keys()):
            e = self.physical_layer.emitters[eid]
            s = e.get_state()
            disrupt = " [DISRUPTED]" if e.is_disrupted else ""
            lines.append(
                f"  emitter_{eid}: buf={s['buffer_frac']:.1%} "
                f"cap={s['current_capture_rate']:.4f}{disrupt}"
            )

        lines.append("\nTransport:")
        for mode_name in self._transport_modes:
            t_state = self.physical_layer.get_transport_state(mode_name)
            is_dis = t_state["is_disrupted"] > 0.5
            disrupt = " [DISRUPTED]" if is_dis else ""
            lines.append(
                f"  transport_{mode_name}: util={t_state['utilization']:.1%} "
                f"in_transit={t_state['in_transit_total']:.4f}{disrupt}"
            )

        lines.append("\nTerminal Buffers:")
        for buf_id, buf in sorted(self.physical_layer.terminal_buffers.items()):
            bs = buf.get_state()
            lines.append(
                f"  {buf_id}: {bs['level']:.4f}/{bs['max_capacity']:.4f} "
                f"({bs['fill_frac']:.1%})"
            )

        lines.append("\nStorage:")
        for sid in sorted(self.physical_layer.storage_sites.keys()):
            st = self.physical_layer.storage_sites[sid]
            s = st.get_state()
            disrupt = " [DISRUPTED]" if st.is_disrupted else ""
            lines.append(
                f"  storage_{sid}: P={s['current_pressure']:.1f}bar "
                f"({s['pressure_frac']:.1%}) fill={s['fill_frac']:.1%} "
                f"inj={s['current_injectivity']:.2f}{disrupt}"
            )

        lines.append(
            f"\nCumulative: stored={self._episode_stats['total_stored']:.3f} Mt  "
            f"vented={self._episode_stats['total_vented']:.3f} Mt  "
            f"violations={self._episode_stats['pressure_violations']}"
        )

        output = "\n".join(lines)
        print(output)
        return output

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def set_lagrangian_lambdas(self, lambdas: Dict[int, float]) -> None:
        """Update Lagrangian multipliers for storage constraint.

        Called by training code after each PPO batch via dual ascent.
        """
        self._lagrangian_lambdas = lambdas

    def get_episode_stats(self) -> Dict[str, Any]:
        stats = dict(self._episode_stats)
        stats["per_agent"] = {a: dict(s) for a, s in self._per_agent_stats.items()}
        stats["pricing_mode"] = self.pricing_mode
        stats["carbon_tax"] = self.get_economic_context()["carbon_tax"]
        return stats

    def state(self) -> Dict[str, Any]:
        return {
            "timestep": self.timestep,
            "physical_state": self.physical_layer.get_state(),
            "episode_stats": dict(self._episode_stats),
            "economic_context": self.get_economic_context(),
        }

    def global_state_vector(self) -> np.ndarray:
        values: List[float] = [self.timestep / max(self.episode_length, 1)]

        physical_state = self.physical_layer.get_state()
        for eid in sorted(physical_state.get("emitters", {}).keys()):
            est = physical_state["emitters"][eid]
            values.extend([
                float(est.get("buffer_frac", 0.0)),
                float(est.get("current_capture_rate", 0.0)),
                float(est.get("capture_fraction", 0.0)),
                float(est.get("effective_purity", 0.0)),
                float(est.get("is_disrupted", 0.0)),
            ])

        for mode_name in sorted(physical_state.get("transports", {}).keys()):
            tst = physical_state["transports"][mode_name]
            values.extend([
                float(tst.get("available_capacity", 0.0)),
                float(tst.get("capacity", 0.0)),
                float(tst.get("utilization", 0.0)),
                float(tst.get("in_transit_total", 0.0)),
                float(tst.get("is_disrupted", 0.0)),
            ])

        for sid in sorted(physical_state.get("storage_sites", {}).keys()):
            sst = physical_state["storage_sites"][sid]
            values.extend([
                float(sst.get("pressure_frac", 0.0)),
                float(sst.get("fill_frac", 0.0)),
                float(sst.get("current_injectivity", 0.0)),
                float(sst.get("max_injectable", 0.0)),
                float(sst.get("last_inlet_purity", 0.0)),
                float(sst.get("last_quality_penalty", 0.0)),
                float(sst.get("is_disrupted", 0.0)),
            ])

        for buf_id in sorted(physical_state.get("terminal_buffers", {}).keys()):
            bst = physical_state["terminal_buffers"][buf_id]
            values.extend([
                float(bst.get("fill_frac", 0.0)),
                float(bst.get("level", 0.0)),
            ])

        econ = self.get_economic_context()
        values.extend([
            float(econ.get("carbon_tax", 0.0)),
            float(econ.get("electricity_price", 0.0)),
            float(econ.get("capture_subsidy", 0.0)),
            float(econ.get("storage_credit", 0.0)),
            float(econ.get("offspec_penalty", 0.0)),
        ])

        return np.asarray(values, dtype=np.float32)
