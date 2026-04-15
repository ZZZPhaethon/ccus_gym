"""Disruption generation system for the CCUS-Gym environment.

Supports three disruption families:
- Transport (T): pipeline failure, ship weather, rail conflict
- Supply (S): equipment failure, production swing, scheduled maintenance
- Storage (G): pressure exceedance, well failure, regulatory stop

Seven scenario families: T, S, G, TS, TG, SG, TSG.
Severity levels kappa in {0.3, 0.5, 0.7} scale disruption frequency and impact.
Cross-disruption correlation rho_cross couples disruption occurrence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ccus_gym.core.network import CCUSNetwork, TransportType


# ---------------------------------------------------------------------------
# Disruption event
# ---------------------------------------------------------------------------

@dataclass
class DisruptionEvent:
    """A single disruption occurrence."""
    target_type: str        # "emitter", "transport", "storage"
    target_id: int
    severity: float         # fraction of capacity lost (0..1)
    duration: float         # months
    cause: str              # e.g. "pipeline_failure", "well_failure"
    timestep: int           # when it starts


# ---------------------------------------------------------------------------
# Disruption generator
# ---------------------------------------------------------------------------

# Base monthly probabilities (before kappa scaling)
_TRANSPORT_DISRUPTION_SPECS = {
    "pipeline_failure": {
        "base_prob": 0.03,
        "duration_range": (1.0, 3.0),
        "severity_range": (0.8, 1.0),   # near-total loss
    },
    "ship_weather": {
        "base_prob": 0.08,
        "duration_range": (0.5, 1.0),
        "severity_range": (0.5, 1.0),
    },
    "rail_conflict": {
        "base_prob": 0.05,
        "duration_range": (0.5, 2.0),
        "severity_range": (0.3, 0.8),
    },
}

# Map transport type to applicable disruption causes
_TRANSPORT_TYPE_CAUSES = {
    TransportType.PIPELINE: ["pipeline_failure"],
    TransportType.SHIP: ["ship_weather"],
    TransportType.RAIL: ["rail_conflict"],
}

_SUPPLY_DISRUPTION_SPECS = {
    "equipment_failure": {
        "base_prob": 0.04,
        "duration_range": (1.0, 3.0),
        "severity_range": (0.3, 0.8),
    },
    "production_swing": {
        "base_prob": 0.06,
        "duration_range": (2.0, 6.0),
        "severity_range": (0.1, 0.4),
        "correlated": True,   # can affect multiple emitters simultaneously
    },
    "maintenance": {
        "base_prob": 0.02,
        "duration_range": (1.0, 2.0),
        "severity_range": (0.9, 1.0),
        "scheduled": True,
    },
}

_STORAGE_DISRUPTION_SPECS = {
    "well_failure": {
        "base_prob": 0.02,
        "duration_range": (2.0, 6.0),
        "severity_range": (0.8, 1.0),
    },
    "regulatory_stop": {
        "base_prob": 0.01,
        "duration_range": (3.0, 12.0),
        "severity_range": (1.0, 1.0),  # complete stop
    },
}

# Scenario family activation flags
SCENARIO_FAMILIES: Dict[str, Dict[str, bool]] = {
    "T":   {"transport": True,  "supply": False, "storage": False},
    "S":   {"transport": False, "supply": True,  "storage": False},
    "G":   {"transport": False, "supply": False, "storage": True},
    "TS":  {"transport": True,  "supply": True,  "storage": False},
    "TG":  {"transport": True,  "supply": False, "storage": True},
    "SG":  {"transport": False, "supply": True,  "storage": True},
    "TSG": {"transport": True,  "supply": True,  "storage": True},
}


class DisruptionGenerator:
    """Generates disruption events for a CCUS network simulation.

    Parameters
    ----------
    network : CCUSNetwork
        The network whose components may be disrupted.
    scenario_family : str
        One of T, S, G, TS, TG, SG, TSG.
    severity_kappa : float
        Global severity scaling in [0, 1]. Scales both probability and
        impact of disruptions.
    cross_correlation : float
        rho_cross in [0, 1]. Probability that a disruption in one domain
        triggers a correlated disruption in another active domain.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        network: CCUSNetwork,
        scenario_family: str = "T",
        severity_kappa: float = 0.5,
        cross_correlation: float = 0.0,
        seed: int = 42,
        gamma: float = 1.0,
    ) -> None:
        self.network = network
        self.scenario_family = scenario_family.upper()
        self.kappa = severity_kappa
        self.rho_cross = cross_correlation
        self.rng = np.random.default_rng(seed)

        # Gamma mechanism axis: affects disruption character
        # High gamma → shorter but more intense disruptions (need fast response)
        # Low gamma → longer but milder disruptions (centralized has time)
        self.gamma = gamma
        self.duration_scale = 1.0 + (1.0 - gamma) * 2.0   # low gamma = longer
        self.severity_scale = 0.5 + gamma * 0.5            # high gamma = more severe

        if self.scenario_family not in SCENARIO_FAMILIES:
            raise ValueError(
                f"Unknown scenario family '{self.scenario_family}'. "
                f"Choose from {list(SCENARIO_FAMILIES.keys())}."
            )
        self.active = SCENARIO_FAMILIES[self.scenario_family]

        # Pre-schedule maintenance events (deterministic)
        self._scheduled_maintenance: List[DisruptionEvent] = []
        if self.active["supply"]:
            self._schedule_maintenance()

    def _schedule_maintenance(self) -> None:
        """Pre-generate scheduled maintenance for all emitters."""
        spec = _SUPPLY_DISRUPTION_SPECS["maintenance"]
        for eid in self.network.emitters:
            # One maintenance per year on average, at roughly fixed intervals
            interval = int(12 / max(spec["base_prob"] * 12, 0.1))
            offset = self.rng.integers(0, max(interval, 1))
            t = offset
            while t < 120:  # max episode length
                dur = self.rng.uniform(*spec["duration_range"]) * self.duration_scale
                sev = self.rng.uniform(*spec["severity_range"]) * self.severity_scale
                self._scheduled_maintenance.append(DisruptionEvent(
                    target_type="emitter",
                    target_id=eid,
                    severity=min(sev * self.kappa, 1.0),
                    duration=max(dur, 0.5),
                    cause="maintenance",
                    timestep=t,
                ))
                t += interval

    def generate_disruptions(self, timestep: int) -> List[DisruptionEvent]:
        """Generate disruption events for a given timestep.

        Returns a list of DisruptionEvent that should be applied at this step.
        """
        events: List[DisruptionEvent] = []

        # --- Scheduled maintenance ---
        for evt in self._scheduled_maintenance:
            if evt.timestep == timestep:
                events.append(evt)

        # --- Stochastic transport disruptions ---
        if self.active["transport"]:
            for tid, transport in self.network.transports.items():
                if transport.is_disrupted:
                    continue  # skip if already disrupted
                causes = _TRANSPORT_TYPE_CAUSES.get(transport.mode_type, [])
                for cause in causes:
                    spec = _TRANSPORT_DISRUPTION_SPECS[cause]
                    prob = spec["base_prob"] * self.kappa
                    if self.rng.random() < prob:
                        dur = self.rng.uniform(*spec["duration_range"]) * self.duration_scale
                        sev = self.rng.uniform(*spec["severity_range"]) * self.severity_scale * self.kappa
                        sev = min(sev, 1.0)
                        events.append(DisruptionEvent(
                            target_type="transport",
                            target_id=tid,
                            severity=sev,
                            duration=max(dur, 0.5),
                            cause=cause,
                            timestep=timestep,
                        ))

        # --- Stochastic supply disruptions ---
        if self.active["supply"]:
            # Equipment failure (independent per emitter)
            spec_eq = _SUPPLY_DISRUPTION_SPECS["equipment_failure"]
            for eid, emitter in self.network.emitters.items():
                if emitter.is_disrupted:
                    continue
                prob = spec_eq["base_prob"] * self.kappa
                if self.rng.random() < prob:
                    dur = self.rng.uniform(*spec_eq["duration_range"]) * self.duration_scale
                    sev = self.rng.uniform(*spec_eq["severity_range"]) * self.severity_scale * self.kappa
                    events.append(DisruptionEvent(
                        target_type="emitter",
                        target_id=eid,
                        severity=min(sev, 1.0),
                        duration=max(dur, 0.5),
                        cause="equipment_failure",
                        timestep=timestep,
                    ))

            # Production swing (correlated across emitters)
            spec_ps = _SUPPLY_DISRUPTION_SPECS["production_swing"]
            prob_ps = spec_ps["base_prob"] * self.kappa
            if self.rng.random() < prob_ps:
                dur = self.rng.uniform(*spec_ps["duration_range"]) * self.duration_scale
                sev = self.rng.uniform(*spec_ps["severity_range"]) * self.severity_scale * self.kappa
                # Apply to all emitters (correlated)
                for eid in self.network.emitters:
                    if not self.network.emitters[eid].is_disrupted:
                        events.append(DisruptionEvent(
                            target_type="emitter",
                            target_id=eid,
                            severity=min(sev, 1.0),
                            duration=max(dur, 0.5),
                            cause="production_swing",
                            timestep=timestep,
                        ))

        # --- Stochastic storage disruptions ---
        if self.active["storage"]:
            for sid, site in self.network.storage_sites.items():
                if site.is_disrupted:
                    continue

                # Well failure
                spec_wf = _STORAGE_DISRUPTION_SPECS["well_failure"]
                prob = spec_wf["base_prob"] * self.kappa
                if self.rng.random() < prob:
                    dur = self.rng.uniform(*spec_wf["duration_range"]) * self.duration_scale
                    sev = self.rng.uniform(*spec_wf["severity_range"]) * self.severity_scale * self.kappa
                    events.append(DisruptionEvent(
                        target_type="storage",
                        target_id=sid,
                        severity=min(sev, 1.0),
                        duration=max(dur, 0.5),
                        cause="well_failure",
                        timestep=timestep,
                    ))
                    continue  # don't double-disrupt

                # Regulatory stop
                spec_rs = _STORAGE_DISRUPTION_SPECS["regulatory_stop"]
                prob = spec_rs["base_prob"] * self.kappa
                if self.rng.random() < prob:
                    dur = self.rng.uniform(*spec_rs["duration_range"]) * self.duration_scale
                    events.append(DisruptionEvent(
                        target_type="storage",
                        target_id=sid,
                        severity=1.0,
                        duration=max(dur, 0.5),
                        cause="regulatory_stop",
                        timestep=timestep,
                    ))

        # --- Cross-domain correlation ---
        if self.rho_cross > 0 and len(events) > 0:
            events = self._apply_cross_correlation(events, timestep)

        return events

    def _apply_cross_correlation(
        self, events: List[DisruptionEvent], timestep: int
    ) -> List[DisruptionEvent]:
        """When a disruption occurs in one domain, potentially trigger one
        in another active domain with probability rho_cross.
        """
        additional: List[DisruptionEvent] = []
        domains_hit = {e.target_type for e in events}

        for domain in domains_hit:
            if self.rng.random() >= self.rho_cross:
                continue
            # Pick a different active domain to trigger
            other_domains = []
            if self.active["transport"] and "transport" not in domains_hit:
                other_domains.append("transport")
            if self.active["supply"] and "emitter" not in domains_hit:
                other_domains.append("supply")
            if self.active["storage"] and "storage" not in domains_hit:
                other_domains.append("storage")

            if not other_domains:
                continue

            target_domain = self.rng.choice(other_domains)
            if target_domain == "transport":
                tid = self.rng.choice(list(self.network.transports.keys()))
                if not self.network.transports[tid].is_disrupted:
                    additional.append(DisruptionEvent(
                        target_type="transport",
                        target_id=tid,
                        severity=0.5 * self.kappa,
                        duration=1.0,
                        cause="cross_correlated",
                        timestep=timestep,
                    ))
            elif target_domain == "supply":
                eid = self.rng.choice(list(self.network.emitters.keys()))
                if not self.network.emitters[eid].is_disrupted:
                    additional.append(DisruptionEvent(
                        target_type="emitter",
                        target_id=eid,
                        severity=0.3 * self.kappa,
                        duration=1.0,
                        cause="cross_correlated",
                        timestep=timestep,
                    ))
            elif target_domain == "storage":
                sid = self.rng.choice(list(self.network.storage_sites.keys()))
                if not self.network.storage_sites[sid].is_disrupted:
                    additional.append(DisruptionEvent(
                        target_type="storage",
                        target_id=sid,
                        severity=0.5 * self.kappa,
                        duration=2.0,
                        cause="cross_correlated",
                        timestep=timestep,
                    ))

        return events + additional

    def apply_events(self, events: List[DisruptionEvent]) -> None:
        """Apply a list of disruption events to the network."""
        for evt in events:
            if evt.target_type == "emitter":
                self.network.emitters[evt.target_id].apply_disruption(
                    evt.severity, evt.duration
                )
            elif evt.target_type == "transport":
                self.network.transports[evt.target_id].apply_disruption(
                    evt.severity, evt.duration
                )
            elif evt.target_type == "storage":
                self.network.storage_sites[evt.target_id].apply_disruption(
                    evt.severity, evt.duration
                )

    def check_pressure_triggered(self) -> List[DisruptionEvent]:
        """Check for pressure-triggered storage disruptions.

        If P > P_safe at any storage site, generate a throttle event.
        """
        events: List[DisruptionEvent] = []
        if not self.active["storage"]:
            return events

        for sid, site in self.network.storage_sites.items():
            if site.current_pressure > site.pressure_limit and not site.is_disrupted:
                events.append(DisruptionEvent(
                    target_type="storage",
                    target_id=sid,
                    severity=0.5,  # throttle to 50%
                    duration=2.0,
                    cause="pressure_exceedance",
                    timestep=-1,
                ))
        return events


# ---------------------------------------------------------------------------
# Mechanism axes computation
# ---------------------------------------------------------------------------

class MechanismAxes:
    r"""Compute mechanism phase-diagram axes (alpha, beta, gamma) from scenario.

    alpha (local observability):
        Fraction of disruption state only locally observable.
        Higher alpha => agents must rely more on local information.

    beta (shared-constraint intensity):
        Degree of coupling between agents via shared resources
        (transport capacity, storage pressure).

    gamma (response-time criticality):
        Ratio tau_disruption / T_response. Higher gamma => less time
        to react relative to disruption duration.
    """

    @staticmethod
    def compute(
        scenario_family: str,
        severity_kappa: float,
        network: CCUSNetwork,
    ) -> Tuple[float, float, float]:
        """Compute (alpha, beta, gamma) for a scenario configuration.

        These are approximate heuristic values used for phase-diagram
        placement rather than exact physical quantities.
        """
        active = SCENARIO_FAMILIES.get(scenario_family.upper(), {})
        n_active = sum(active.values())

        # --- alpha: local observability ---
        # More disruption types => harder to observe everything globally
        # Transport disruptions are locally observable to transport agents
        # Supply disruptions are locally observable to emitter agents
        # Storage disruptions are locally observable to storage agents
        alpha_base = 0.3
        if n_active >= 2:
            alpha_base = 0.6
        if n_active >= 3:
            alpha_base = 0.8
        alpha = min(1.0, alpha_base * (0.5 + severity_kappa))

        # --- beta: shared-constraint intensity ---
        # Pipeline/transport sharing creates coupling
        n_emitters = len(network.emitters)
        n_transports = len(network.transports)
        n_storage = len(network.storage_sites)
        # More emitters per transport/storage => more coupling
        ratio = n_emitters / max(n_transports + n_storage, 1)
        beta = min(1.0, 0.2 * ratio * (1 + severity_kappa))

        # --- gamma: response-time criticality ---
        # Shorter disruption durations relative to transport latency =>
        # higher gamma
        avg_latency = np.mean([t.latency for t in network.transports.values()])
        # Average disruption duration (rough estimate)
        avg_dur = 2.0 / max(severity_kappa, 0.1)  # shorter at high kappa
        gamma = avg_latency / max(avg_dur, 0.1)
        gamma = min(5.0, max(0.1, gamma * (1 + severity_kappa)))

        return alpha, beta, gamma

    @staticmethod
    def from_config(config: Dict[str, Any]) -> Tuple[float, float, float]:
        """Read (alpha, beta, gamma) directly from config if provided,
        otherwise compute from scenario parameters."""
        mech = config.get("mechanism", {})
        if "alpha" in mech and "beta" in mech and "gamma" in mech:
            return mech["alpha"], mech["beta"], mech["gamma"]
        # Otherwise compute
        network = CCUSNetwork.from_config(config)
        return MechanismAxes.compute(
            config["disruption"]["scenario_family"],
            config["disruption"]["severity"],
            network,
        )
