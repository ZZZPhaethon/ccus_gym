"""Network topology and component models for the CCUS system.

Components:
- Emitter: CO2 source with capture facility and buffer storage
- TransportMode: Pipeline, ship, or rail transport link
- StorageSite: CO2 injection well with reduced-order pressure model
- CCUSNetwork: Container holding all components and their connectivity
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TransportType(enum.Enum):
    PIPELINE = "pipeline"
    SHIP = "ship"
    RAIL = "rail"


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------

@dataclass
class Emitter:
    """CO2 source with capture facility and on-site buffer storage.

    All flow quantities are in MtCO2/month.
    """

    id: int
    name: str

    # Design parameters
    max_capture_rate: float          # MtCO2/month
    buffer_capacity: float           # MtCO2
    production_rate: float           # MtCO2/month (raw CO2 produced)
    capture_method: str = "post_combustion"
    sector: str = "industrial"
    base_purity: float = 0.90
    composition: Dict[str, float] = field(default_factory=dict)
    capture_cost_per_t: float = 55.0
    capture_energy_mwh_per_t: float = 0.32
    purification_cost_factor: float = 0.35
    purification_energy_factor: float = 0.20

    # State (mutable)
    buffer_level: float = 0.0
    current_capture_rate: float = 0.0
    is_disrupted: bool = False
    disruption_remaining_time: float = 0.0
    disruption_severity: float = 0.0  # fraction of capacity lost

    def __post_init__(self) -> None:
        self.current_capture_rate = self.max_capture_rate

    def reset(self) -> None:
        self.buffer_level = 0.0
        self.current_capture_rate = self.max_capture_rate
        self.is_disrupted = False
        self.disruption_remaining_time = 0.0
        self.disruption_severity = 0.0

    def apply_disruption(self, severity: float, duration: float) -> None:
        """Apply a supply-side disruption (equipment failure, etc.)."""
        self.is_disrupted = True
        self.disruption_severity = severity
        self.disruption_remaining_time = duration
        self.current_capture_rate = self.max_capture_rate * (1.0 - severity)

    def step(self, co2_sent_out: float) -> Tuple[float, float]:
        """Advance one timestep.

        Args:
            co2_sent_out: CO2 volume dispatched to transport this step.

        Returns:
            (co2_captured, co2_vented): captured amount and any overflow vented.
        """
        # Capture CO2
        co2_captured = min(self.current_capture_rate, self.production_rate)

        # Update buffer
        self.buffer_level += co2_captured - co2_sent_out
        co2_vented = 0.0
        if self.buffer_level > self.buffer_capacity:
            co2_vented = self.buffer_level - self.buffer_capacity
            self.buffer_level = self.buffer_capacity
        if self.buffer_level < 0.0:
            # Tried to send more than available — clamp
            co2_sent_out += self.buffer_level  # reduce actual send
            self.buffer_level = 0.0

        # Tick disruption timer
        if self.is_disrupted:
            self.disruption_remaining_time -= 1.0
            if self.disruption_remaining_time <= 0.0:
                self.is_disrupted = False
                self.disruption_severity = 0.0
                self.disruption_remaining_time = 0.0
                self.current_capture_rate = self.max_capture_rate

        return co2_captured, co2_vented

    def get_state(self) -> Dict[str, float]:
        return {
            "buffer_level": self.buffer_level,
            "buffer_frac": self.buffer_level / max(self.buffer_capacity, 1e-9),
            "current_capture_rate": self.current_capture_rate,
            "max_capture_rate": self.max_capture_rate,
            "is_disrupted": float(self.is_disrupted),
            "disruption_remaining_time": self.disruption_remaining_time,
        }


# ---------------------------------------------------------------------------
# TransportMode
# ---------------------------------------------------------------------------

@dataclass
class TransportMode:
    """Transport link between emitters and storage sites.

    All capacities in MtCO2/month.
    """

    id: int
    name: str
    mode_type: TransportType

    # Design parameters
    capacity: float               # MtCO2/month
    base_cost: float              # $/tCO2
    latency: float                # months (delivery delay)

    # State (mutable)
    available_capacity: float = 0.0
    is_disrupted: bool = False
    disruption_remaining_time: float = 0.0
    disruption_severity: float = 0.0  # fraction of capacity lost

    # In-transit queue: list of (volume, remaining_latency)
    in_transit: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.available_capacity = self.capacity

    def reset(self) -> None:
        self.available_capacity = self.capacity
        self.is_disrupted = False
        self.disruption_remaining_time = 0.0
        self.disruption_severity = 0.0
        self.in_transit = []

    def apply_disruption(self, severity: float, duration: float) -> None:
        """Apply a transport disruption."""
        self.is_disrupted = True
        self.disruption_severity = severity
        self.disruption_remaining_time = duration
        self.available_capacity = self.capacity * (1.0 - severity)

    def accept_co2(self, volume: float) -> float:
        """Accept CO2 for transport, subject to available capacity.

        Returns actual volume accepted.
        """
        accepted = min(volume, self.available_capacity)
        if accepted > 0:
            self.in_transit.append((accepted, self.latency))
            self.available_capacity -= accepted
        return accepted

    def step(self) -> float:
        """Advance one timestep. Returns CO2 delivered at destination."""
        delivered = 0.0
        remaining = []
        for vol, t_rem in self.in_transit:
            t_rem -= 1.0
            if t_rem <= 0.0:
                delivered += vol
            else:
                remaining.append((vol, t_rem))
        self.in_transit = remaining

        # Refresh available capacity (monthly capacity resets)
        if self.is_disrupted:
            self.disruption_remaining_time -= 1.0
            if self.disruption_remaining_time <= 0.0:
                self.is_disrupted = False
                self.disruption_severity = 0.0
                self.disruption_remaining_time = 0.0
                self.available_capacity = self.capacity
            else:
                self.available_capacity = self.capacity * (1.0 - self.disruption_severity)
        else:
            self.available_capacity = self.capacity

        return delivered

    def get_state(self) -> Dict[str, float]:
        in_transit_total = sum(v for v, _ in self.in_transit)
        return {
            "available_capacity": self.available_capacity,
            "capacity": self.capacity,
            "utilization": 1.0 - self.available_capacity / max(self.capacity, 1e-9),
            "in_transit_total": in_transit_total,
            "is_disrupted": float(self.is_disrupted),
            "disruption_remaining_time": self.disruption_remaining_time,
        }


# ---------------------------------------------------------------------------
# StorageSite
# ---------------------------------------------------------------------------

@dataclass
class StorageSite:
    """CO2 injection site with reduced-order pressure model (ROM).

    Pressure dynamics:
        P(t) = P(t-1) + k_inj * Q_inj(t) / injectivity(t)
                       - k_diss * (P(t-1) - P_hydro)
                       + cross-well interference terms

    Injectivity decline:
        injectivity(t) = max(0.1, 1.0 - decline_rate * cumulative / capacity)

    All pressures in bar, volumes in MtCO2, rates in MtCO2/month.
    """

    id: int
    name: str

    # Design parameters
    max_injection_rate: float      # MtCO2/month
    pressure_limit: float          # bar (P_safe)
    initial_pressure: float        # bar (P_hydro)
    k_injection: float             # bar per (MtCO2/month) at unit injectivity
    k_dissipation: float           # dimensionless dissipation rate
    cross_well_coeff: float        # bar per (MtCO2/month) from other wells
    injectivity_decline_rate: float  # dimensionless
    cumulative_capacity: float     # MtCO2 total
    min_purity: float = 0.93
    max_impurities: Dict[str, float] = field(default_factory=dict)

    # State (mutable)
    current_pressure: float = 0.0
    cumulative_injected: float = 0.0
    current_injectivity: float = 1.0
    is_disrupted: bool = False
    disruption_remaining_time: float = 0.0
    disruption_severity: float = 0.0

    def __post_init__(self) -> None:
        self.current_pressure = self.initial_pressure
        self.cumulative_injected = 0.0
        self.current_injectivity = 1.0

    def reset(self) -> None:
        self.current_pressure = self.initial_pressure
        self.cumulative_injected = 0.0
        self.current_injectivity = 1.0
        self.is_disrupted = False
        self.disruption_remaining_time = 0.0
        self.disruption_severity = 0.0

    def apply_disruption(self, severity: float, duration: float) -> None:
        """Apply a storage disruption (well failure, regulatory stop, etc.)."""
        self.is_disrupted = True
        self.disruption_severity = severity
        self.disruption_remaining_time = duration

    def get_max_injectable(self) -> float:
        """Maximum injectable rate given current pressure headroom and injectivity."""
        if self.is_disrupted and self.disruption_severity >= 1.0:
            return 0.0

        pressure_headroom = self.pressure_limit - self.current_pressure
        if pressure_headroom <= 0.0:
            return 0.0

        # Inverse of the pressure equation: Q_max such that delta_P = headroom
        # delta_P = k_inj * Q / injectivity  =>  Q = headroom * injectivity / k_inj
        inj = max(self.current_injectivity, 0.1)
        q_pressure_limited = pressure_headroom * inj / max(self.k_injection, 1e-9)

        # Also limited by design max and disruption
        effective_max = self.max_injection_rate
        if self.is_disrupted:
            effective_max *= (1.0 - self.disruption_severity)

        # Also limited by remaining capacity
        remaining_capacity = max(0.0, self.cumulative_capacity - self.cumulative_injected)

        return max(0.0, min(q_pressure_limited, effective_max, remaining_capacity))

    def step(
        self,
        injection_rate: float,
        other_site_rates: Optional[List[float]] = None,
        cross_well_scale: float = 1.0,
        pressure_limit_scale: float = 1.0,
    ) -> Tuple[float, bool]:
        """Advance one timestep with given injection.

        Args:
            injection_rate: Desired injection rate (MtCO2/month).
            other_site_rates: Injection rates at other storage sites (for
                cross-well interference).
            cross_well_scale: Multiplier on cross_well_coeff from beta mechanism
                axis. Higher = stronger inter-site pressure coupling.
            pressure_limit_scale: Multiplier on pressure_limit from beta mechanism
                axis. Lower = tighter constraint (shared aquifer).

        Returns:
            (actual_injected, pressure_violation): actual amount injected and
                whether pressure exceeded P_safe.
        """
        # Effective pressure limit (tightened by beta)
        effective_pressure_limit = self.pressure_limit * pressure_limit_scale

        # Clamp to maximum injectable
        max_q = self.get_max_injectable()
        actual_q = min(injection_rate, max_q)
        actual_q = max(actual_q, 0.0)

        # Update injectivity
        self.current_injectivity = max(
            0.1,
            1.0 - self.injectivity_decline_rate
            * self.cumulative_injected / max(self.cumulative_capacity, 1e-9),
        )

        # Pressure update
        inj = max(self.current_injectivity, 0.1)
        delta_p_injection = self.k_injection * actual_q / inj
        delta_p_dissipation = self.k_dissipation * (
            self.current_pressure - self.initial_pressure
        )

        # Cross-well interference (scaled by beta mechanism axis)
        delta_p_cross = 0.0
        if other_site_rates is not None:
            effective_cross_coeff = self.cross_well_coeff * cross_well_scale
            for q_other in other_site_rates:
                delta_p_cross += effective_cross_coeff * q_other

        self.current_pressure += delta_p_injection - delta_p_dissipation + delta_p_cross

        # Accumulate
        self.cumulative_injected += actual_q

        # Check violation against effective (tightened) limit
        pressure_violation = self.current_pressure > effective_pressure_limit

        # Tick disruption timer
        if self.is_disrupted:
            self.disruption_remaining_time -= 1.0
            if self.disruption_remaining_time <= 0.0:
                self.is_disrupted = False
                self.disruption_severity = 0.0
                self.disruption_remaining_time = 0.0

        return actual_q, pressure_violation

    def get_state(self) -> Dict[str, float]:
        return {
            "current_pressure": self.current_pressure,
            "pressure_frac": self.current_pressure / max(self.pressure_limit, 1e-9),
            "cumulative_injected": self.cumulative_injected,
            "fill_frac": self.cumulative_injected / max(self.cumulative_capacity, 1e-9),
            "current_injectivity": self.current_injectivity,
            "max_injectable": self.get_max_injectable(),
            "is_disrupted": float(self.is_disrupted),
            "disruption_remaining_time": self.disruption_remaining_time,
        }


# ---------------------------------------------------------------------------
# CCUSNetwork
# ---------------------------------------------------------------------------

class CCUSNetwork:
    """Container holding all CCUS components and their connectivity.

    Connectivity is defined as a mapping:
        emitter_id -> [(transport_id, storage_id), ...]
    Each tuple represents a feasible route from an emitter through a transport
    mode to a storage site.
    """

    def __init__(
        self,
        emitters: List[Emitter],
        transports: List[TransportMode],
        storage_sites: List[StorageSite],
        connectivity: Dict[int, List[Tuple[int, int]]],
    ) -> None:
        self.emitters = {e.id: e for e in emitters}
        self.transports = {t.id: t for t in transports}
        self.storage_sites = {s.id: s for s in storage_sites}
        self.connectivity = connectivity  # emitter_id -> [(transport_id, storage_id)]

        # Build reverse maps for convenience
        self._transport_to_storage: Dict[int, List[int]] = {}
        for routes in connectivity.values():
            for t_id, s_id in routes:
                self._transport_to_storage.setdefault(t_id, [])
                if s_id not in self._transport_to_storage[t_id]:
                    self._transport_to_storage[t_id].append(s_id)

    def reset(self) -> None:
        for e in self.emitters.values():
            e.reset()
        for t in self.transports.values():
            t.reset()
        for s in self.storage_sites.values():
            s.reset()

    def get_routes_for_emitter(self, emitter_id: int) -> List[Tuple[int, int]]:
        """Return list of (transport_id, storage_id) routes for an emitter."""
        return self.connectivity.get(emitter_id, [])

    def get_connected_transport_ids(self, emitter_id: int) -> List[int]:
        """Transport IDs reachable from an emitter."""
        return list({t for t, _ in self.connectivity.get(emitter_id, [])})

    def get_connected_storage_ids(self, emitter_id: int) -> List[int]:
        """Storage site IDs reachable from an emitter (via any transport)."""
        return list({s for _, s in self.connectivity.get(emitter_id, [])})

    def get_storage_ids_for_transport(self, transport_id: int) -> List[int]:
        """Storage sites served by a transport mode."""
        return self._transport_to_storage.get(transport_id, [])

    def get_network_state(self) -> Dict[str, Any]:
        """Return full network state dictionary."""
        return {
            "emitters": {eid: e.get_state() for eid, e in self.emitters.items()},
            "transports": {tid: t.get_state() for tid, t in self.transports.items()},
            "storage_sites": {sid: s.get_state() for sid, s in self.storage_sites.items()},
        }

    @staticmethod
    def from_config(config: Dict[str, Any]) -> "CCUSNetwork":
        """Build a CCUSNetwork from a configuration dictionary.

        Expected keys under config["network"]:
            num_emitters, transport_modes, num_storage_sites,
            emitter_params, transport_params, storage_params,
            connectivity
        """
        net_cfg = config["network"]

        # --- Build emitters ---
        emitters: List[Emitter] = []
        e_defaults = net_cfg.get("emitter_defaults", {})
        e_overrides = net_cfg.get("emitter_params", {})
        for i in range(net_cfg["num_emitters"]):
            params = {**e_defaults, **e_overrides.get(i, {})}
            emitters.append(Emitter(
                id=i,
                name=f"emitter_{i}",
                max_capture_rate=params["max_capture_rate"],
                buffer_capacity=params["buffer_capacity"],
                production_rate=params["production_rate"],
                capture_method=params.get("capture_method", "post_combustion"),
                sector=params.get("sector", "industrial"),
                base_purity=params.get("base_purity", 0.90),
                composition=dict(params.get("composition", {})),
                capture_cost_per_t=params.get("capture_cost_per_t", 55.0),
                capture_energy_mwh_per_t=params.get("capture_energy_mwh_per_t", 0.32),
                purification_cost_factor=params.get("purification_cost_factor", 0.35),
                purification_energy_factor=params.get("purification_energy_factor", 0.20),
            ))

        # --- Build transports ---
        transports: List[TransportMode] = []
        t_params = net_cfg.get("transport_params", {})
        for idx, mode_name in enumerate(net_cfg["transport_modes"]):
            mode_type = TransportType(mode_name)
            params = t_params.get(mode_name, {})
            transports.append(TransportMode(
                id=idx,
                name=f"transport_{mode_name}",
                mode_type=mode_type,
                capacity=params["capacity"],
                base_cost=params["base_cost"],
                latency=params["latency"],
            ))

        # --- Build storage sites ---
        storage_sites: List[StorageSite] = []
        s_defaults = net_cfg.get("storage_defaults", {})
        s_overrides = net_cfg.get("storage_params", {})
        for i in range(net_cfg["num_storage_sites"]):
            params = {**s_defaults, **s_overrides.get(i, {})}
            storage_sites.append(StorageSite(
                id=i,
                name=f"storage_{i}",
                max_injection_rate=params["max_injection_rate"],
                pressure_limit=params["pressure_limit"],
                initial_pressure=params["initial_pressure"],
                k_injection=params["k_injection"],
                k_dissipation=params["k_dissipation"],
                cross_well_coeff=params["cross_well_coeff"],
                injectivity_decline_rate=params["injectivity_decline_rate"],
                cumulative_capacity=params["cumulative_capacity"],
                min_purity=params.get("min_purity", 0.93),
                max_impurities=dict(params.get("max_impurities", {})),
            ))

        # --- Connectivity ---
        connectivity_raw = net_cfg["connectivity"]
        connectivity: Dict[int, List[Tuple[int, int]]] = {}
        for eid_str, routes in connectivity_raw.items():
            eid = int(eid_str)
            connectivity[eid] = [(int(r[0]), int(r[1])) for r in routes]

        return CCUSNetwork(emitters, transports, storage_sites, connectivity)
