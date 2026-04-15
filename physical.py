"""Physical layer for the CCUS-Gym environment.

Simulates the actual CCUS network physics with no agent logic.
Accepts decision intents and returns physical outcomes.

Components:
- EmitterPhysics: CO2 source with on-site buffer
- PipelinePhysics: Continuous flow transport (instantaneous)
- ShipPhysics: Batch transport with fleet of ships cycling through states
- RailPhysics: Batch transport with trains
- TerminalBuffer: Shared buffer tank at port/rail terminals
- StoragePhysics: Pressure ROM for injection sites (with optional proxy model)
- PhysicalLayer: Wraps all components, executes monthly settlement

All flow rates in MtCO2/month. Pressures in bar. Costs in $/tCO2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ccus_gym.quality import (
    DEFAULT_STORAGE_QUALITY_LIMITS,
    blend_streams,
    compute_effective_stream,
    normalize_composition,
    storage_quality_penalty,
)

if TYPE_CHECKING:
    from ccus_gym.storage_proxy import StorageProxyModel

logger = logging.getLogger(__name__)


# ===================================================================
# Ship archetypes (capacities in MtCO2)
# ===================================================================

SHIP_TYPES = {
    "small": {
        "capacity": 0.0075,        # 7,500 t
        "speed_km_day": 288,       # 12 kt
        "loading_rate": 0.06,      # MtCO2/month loading throughput
        "cost_per_t": 25.0,        # $/tCO2
    },
    "medium": {
        "capacity": 0.022,         # 22,000 t
        "speed_km_day": 288,       # 12 kt
        "loading_rate": 0.06,
        "cost_per_t": 20.0,
    },
    "large": {
        "capacity": 0.050,         # 50,000 t
        "speed_km_day": 264,       # 11 kt
        "loading_rate": 0.06,
        "cost_per_t": 18.0,
    },
}

# ===================================================================
# Rail config defaults
# ===================================================================

RAIL_DEFAULTS = {
    "car_capacity": 0.00008,       # 80 tonnes = 0.00008 MtCO2
    "cars_per_train": 80,          # 80 cars standard
    "loading_time_days": 1,
    "unloading_time_days": 1,
    "transit_time_days": 3,        # configurable by distance
    "cost_per_t_km": 0.04,        # $/tCO2/km
    "distance_km": 200,           # default distance
    "num_trains": 4,              # fleet size
}


# ===================================================================
# Physical outcome
# ===================================================================

@dataclass
class PhysicalOutcome:
    """Results of one month of physical settlement."""

    # Per-emitter
    emitter_captured: Dict[int, float] = field(default_factory=dict)
    emitter_vented: Dict[int, float] = field(default_factory=dict)
    emitter_sent: Dict[int, float] = field(default_factory=dict)
    emitter_buffer_frac: Dict[int, float] = field(default_factory=dict)
    emitter_direct_vent: Dict[int, float] = field(default_factory=dict)
    emitter_capture_energy: Dict[int, float] = field(default_factory=dict)
    emitter_capture_cost: Dict[int, float] = field(default_factory=dict)
    emitter_effective_purity: Dict[int, float] = field(default_factory=dict)

    # Per-transport aggregated
    transport_accepted: Dict[str, float] = field(default_factory=dict)   # mode_name -> vol
    transport_delivered: Dict[str, float] = field(default_factory=dict)
    transport_rejected: Dict[str, float] = field(default_factory=dict)
    transport_utilization: Dict[str, float] = field(default_factory=dict)

    # Terminal buffers
    terminal_buffer_levels: Dict[str, float] = field(default_factory=dict)  # buffer_id -> frac
    terminal_vented: Dict[str, float] = field(default_factory=dict)

    # Per-storage
    storage_injected: Dict[int, float] = field(default_factory=dict)
    storage_pressure_violation: Dict[int, bool] = field(default_factory=dict)
    storage_pressure_margin: Dict[int, float] = field(default_factory=dict)
    storage_inlet_purity: Dict[int, float] = field(default_factory=dict)
    storage_quality_penalty: Dict[int, float] = field(default_factory=dict)
    storage_quality_violation: Dict[int, bool] = field(default_factory=dict)

    # Aggregates
    total_stored: float = 0.0
    total_vented: float = 0.0
    total_captured: float = 0.0
    transport_cost: float = 0.0
    pressure_violations: int = 0
    total_capture_cost: float = 0.0
    total_energy_use: float = 0.0
    quality_violations: int = 0

    # Nomination settlement info (for observations)
    scarcity_ratios: Dict[str, float] = field(default_factory=dict)  # mode -> ratio

    # Pricing & bidding
    transport_revenue: Dict[str, float] = field(default_factory=dict)       # mode -> total revenue
    emitter_transport_cost: Dict[int, float] = field(default_factory=dict)  # eid -> cost paid
    transport_posted_prices: Dict[str, float] = field(default_factory=dict) # mode -> posted price
    emitter_bids: Dict[int, float] = field(default_factory=dict)            # eid -> bid submitted

    # Overflow attribution
    overflow_attributed_emitter: Dict[int, float] = field(default_factory=dict)   # eid -> overflow vol
    overflow_attributed_storage: Dict[int, float] = field(default_factory=dict)   # sid -> overflow vol
    storage_overflow: Dict[int, float] = field(default_factory=dict)              # sid -> overflow vol

    # Congestion pricing
    pipeline_congestion_mult: float = 1.0
    pipeline_congestion_surcharge: Dict[int, float] = field(default_factory=dict)  # eid -> surcharge


# ===================================================================
# EmitterPhysics
# ===================================================================

@dataclass
class EmitterPhysics:
    """CO2 source with capture facility and on-site buffer.

    Produces CO2 each month. Excess goes to buffer, overflow is vented.
    """

    id: int
    max_capture_rate: float     # MtCO2/month
    buffer_capacity: float      # MtCO2
    production_rate: float      # MtCO2/month
    capture_method: str = "post_combustion"
    sector: str = "industrial"
    base_purity: float = 0.90
    composition: Dict[str, float] = field(default_factory=dict)
    capture_cost_per_t: float = 55.0
    capture_energy_mwh_per_t: float = 0.32
    purification_cost_factor: float = 0.35
    purification_energy_factor: float = 0.20

    # Mutable state
    buffer_level: float = 0.0
    current_capture_rate: float = 0.0
    capture_fraction: float = 1.0
    purification_effort: float = 0.0
    last_effective_purity: float = 0.90
    last_composition: Dict[str, float] = field(default_factory=dict)
    is_disrupted: bool = False
    disruption_remaining_time: float = 0.0
    disruption_severity: float = 0.0

    def __post_init__(self) -> None:
        self.current_capture_rate = self.max_capture_rate
        self.last_effective_purity = self.base_purity
        self.last_composition = normalize_composition(
            self.composition, purity_hint=self.base_purity
        )

    def reset(self) -> None:
        self.buffer_level = 0.0
        self.current_capture_rate = self.max_capture_rate
        self.capture_fraction = 1.0
        self.purification_effort = 0.0
        self.last_effective_purity = self.base_purity
        self.last_composition = normalize_composition(
            self.composition, purity_hint=self.base_purity
        )
        self.is_disrupted = False
        self.disruption_remaining_time = 0.0
        self.disruption_severity = 0.0

    def apply_disruption(self, severity: float, duration: float) -> None:
        self.is_disrupted = True
        self.disruption_severity = severity
        self.disruption_remaining_time = duration
        self.current_capture_rate = self.max_capture_rate * (1.0 - severity)

    def set_capture_controls(self, capture_fraction: float, purification_effort: float) -> None:
        self.capture_fraction = max(0.0, min(1.0, float(capture_fraction)))
        self.purification_effort = max(0.0, min(1.0, float(purification_effort)))
        purity, composition = compute_effective_stream(
            self.capture_method,
            self.purification_effort,
            base_purity=self.base_purity,
            base_composition=self.composition,
        )
        self.last_effective_purity = purity
        self.last_composition = composition

    def _captured_volume(self, capture_fraction: float | None = None) -> float:
        frac = self.capture_fraction if capture_fraction is None else float(capture_fraction)
        frac = max(0.0, min(1.0, frac))
        desired_capture = self.production_rate * frac
        return min(self.current_capture_rate, desired_capture)

    def produce_and_buffer(self, co2_sent_out: float) -> Tuple[float, float, float]:
        """Produce CO2, update buffer, return (captured, vented).

        co2_sent_out is capped at what's actually available (buffer + capture)
        to ensure mass conservation.
        """
        co2_captured = self._captured_volume()
        direct_vent = max(0.0, self.production_rate - co2_captured)
        # Cap sent_out to what's actually available (prevents negative buffer / mass loss)
        actual_sent = min(co2_sent_out, self.buffer_level + co2_captured)
        self.buffer_level += co2_captured - actual_sent

        co2_vented = direct_vent
        if self.buffer_level > self.buffer_capacity:
            co2_vented = self.buffer_level - self.buffer_capacity
            co2_vented += direct_vent
            self.buffer_level = self.buffer_capacity
        if self.buffer_level < 0.0:
            # Should not happen anymore, but safety check
            self.buffer_level = 0.0

        # Tick disruption
        if self.is_disrupted:
            self.disruption_remaining_time -= 1.0
            if self.disruption_remaining_time <= 0.0:
                self.is_disrupted = False
                self.disruption_severity = 0.0
                self.disruption_remaining_time = 0.0
                self.current_capture_rate = self.max_capture_rate

        return co2_captured, co2_vented, direct_vent

    def get_available(self, capture_fraction: float | None = None) -> float:
        """CO2 available to nominate this month (buffer + new capture)."""
        return self.buffer_level + self._captured_volume(capture_fraction)

    def get_state(self) -> Dict[str, float]:
        return {
            "buffer_level": self.buffer_level,
            "buffer_frac": self.buffer_level / max(self.buffer_capacity, 1e-9),
            "current_capture_rate": self.current_capture_rate,
            "max_capture_rate": self.max_capture_rate,
            "capture_fraction": self.capture_fraction,
            "purification_effort": self.purification_effort,
            "effective_purity": self.last_effective_purity,
            "is_disrupted": float(self.is_disrupted),
            "disruption_remaining_time": self.disruption_remaining_time,
        }


# ===================================================================
# PipelinePhysics
# ===================================================================

@dataclass
class PipelinePhysics:
    """Continuous flow pipeline. Instantaneous delivery (no latency).

    Accepts flow nominations up to capacity. Returns actual flow.
    """

    capacity: float            # MtCO2/month
    base_cost: float           # $/tCO2

    # Mutable state
    available_capacity: float = 0.0
    is_disrupted: bool = False
    disruption_remaining_time: float = 0.0
    disruption_severity: float = 0.0
    month_accepted: float = 0.0
    month_delivered: float = 0.0

    def __post_init__(self) -> None:
        self.available_capacity = self.capacity

    def reset(self) -> None:
        self.available_capacity = self.capacity
        self.is_disrupted = False
        self.disruption_remaining_time = 0.0
        self.disruption_severity = 0.0
        self.month_accepted = 0.0
        self.month_delivered = 0.0

    def apply_disruption(self, severity: float, duration: float) -> None:
        self.is_disrupted = True
        self.disruption_severity = severity
        self.disruption_remaining_time = duration
        self.available_capacity = self.capacity * (1.0 - severity)

    def accept_and_deliver(self, volume: float) -> Tuple[float, float]:
        """Accept volume, deliver immediately. Returns (accepted, cost)."""
        accepted = min(volume, self.available_capacity)
        accepted = max(accepted, 0.0)
        self.available_capacity -= accepted
        self.month_accepted += accepted
        self.month_delivered += accepted
        cost = accepted * self.base_cost
        return accepted, cost

    def end_month(self) -> None:
        """Reset monthly counters and tick disruption."""
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
        self.month_accepted = 0.0
        self.month_delivered = 0.0

    def get_state(self) -> Dict[str, float]:
        return {
            "available_capacity": self.available_capacity,
            "capacity": self.capacity,
            "utilization": self.month_accepted / max(self.capacity, 1e-9),
            "in_transit_total": 0.0,  # pipeline has no in-transit
            "is_disrupted": float(self.is_disrupted),
            "disruption_remaining_time": self.disruption_remaining_time,
        }


# ===================================================================
# ShipPhysics
# ===================================================================

class ShipState:
    """State of a single ship."""
    IDLE = "idle"
    LOADING = "loading"
    IN_TRANSIT = "in_transit"
    UNLOADING = "unloading"
    RETURN_TRANSIT = "return_transit"


@dataclass
class Ship:
    """Individual ship in the fleet."""
    ship_type: str
    capacity: float            # MtCO2
    speed_km_day: float
    loading_rate: float        # MtCO2/month throughput
    cost_per_t: float          # $/tCO2

    # State
    state: str = ShipState.IDLE
    cargo: float = 0.0
    days_remaining: float = 0.0
    target_storage_id: int = -1

    def reset(self) -> None:
        self.state = ShipState.IDLE
        self.cargo = 0.0
        self.days_remaining = 0.0
        self.target_storage_id = -1


@dataclass
class ShipPhysics:
    """Fleet of ships for batch CO2 transport.

    Ships cycle: idle -> loading -> in_transit -> unloading -> return_transit -> idle
    Within a month (30 days), we simulate day-level state transitions.
    """

    fleet: List[Ship] = field(default_factory=list)
    distance_km: float = 800.0     # one-way distance
    loading_days: float = 2.0       # days to load a full ship
    unloading_days: float = 2.0     # days to unload
    base_cost: float = 20.0         # fallback $/tCO2 if not per-ship

    # Disruption state (applied to entire fleet)
    is_disrupted: bool = False
    disruption_remaining_time: float = 0.0
    disruption_severity: float = 0.0

    # Monthly trackers
    month_accepted: float = 0.0
    month_delivered: float = 0.0
    month_cost: float = 0.0

    def reset(self) -> None:
        for ship in self.fleet:
            ship.reset()
        self.is_disrupted = False
        self.disruption_remaining_time = 0.0
        self.disruption_severity = 0.0
        self.month_accepted = 0.0
        self.month_delivered = 0.0
        self.month_cost = 0.0

    def apply_disruption(self, severity: float, duration: float) -> None:
        self.is_disrupted = True
        self.disruption_severity = severity
        self.disruption_remaining_time = duration

    def get_idle_ships(self) -> List[Ship]:
        """Return ships available for loading."""
        if self.is_disrupted and self.disruption_severity >= 1.0:
            return []
        idle = [s for s in self.fleet if s.state == ShipState.IDLE]
        if self.is_disrupted:
            # Reduce available fleet proportionally
            n_available = max(1, int(len(idle) * (1.0 - self.disruption_severity)))
            return idle[:n_available]
        return idle

    def get_monthly_capacity(self) -> float:
        """Estimate monthly throughput capacity based on fleet and cycle time."""
        if not self.fleet:
            return 0.0
        # Approximate: each idle ship can do one round trip per month
        idle = self.get_idle_ships()
        return sum(s.capacity for s in idle)

    def load_ship(self, ship: Ship, volume: float, terminal_buffer: "TerminalBuffer",
                  target_storage_id: int = 0) -> float:
        """Start loading CO2 from terminal buffer onto a ship.

        Ship enters LOADING state and will continue to load each day
        via continue_loading() until loading_days are done.
        Returns actual volume loaded in this initial call.
        """
        available_in_buffer = terminal_buffer.level
        initial_load = min(volume, ship.capacity, available_in_buffer)
        initial_load = max(initial_load, 0.0)

        if initial_load > 0 or True:  # Start loading even if buffer temporarily empty
            terminal_buffer.withdraw(initial_load)
            ship.cargo = initial_load
            ship.state = ShipState.LOADING
            ship.days_remaining = self.loading_days  # fixed loading time
            ship.target_storage_id = target_storage_id
            self.month_accepted += initial_load
            self.month_cost += initial_load * ship.cost_per_t

        return initial_load

    def continue_loading(self, ship: Ship, terminal_buffer: "TerminalBuffer") -> float:
        """Continue loading a ship that is in LOADING state.

        Called each day while ship is at berth. Tops up cargo from buffer.
        Returns additional volume loaded.
        """
        if ship.state != ShipState.LOADING:
            return 0.0
        remaining_capacity = ship.capacity - ship.cargo
        if remaining_capacity <= 0:
            return 0.0
        available = terminal_buffer.level
        top_up = min(remaining_capacity, available)
        if top_up > 0:
            terminal_buffer.withdraw(top_up)
            ship.cargo += top_up
            self.month_accepted += top_up
            self.month_cost += top_up * ship.cost_per_t
        return top_up

    def simulate_month(self, days: float = 30.0) -> Dict[int, float]:
        """Advance all ships by one month (30 days).

        Returns dict of storage_id -> CO2 delivered this month.
        """
        delivered: Dict[int, float] = {}
        time_remaining = days

        # Process in small steps to handle state transitions within a month
        dt = 1.0  # 1-day steps
        t = 0.0
        while t < days:
            for ship in self.fleet:
                if ship.state == ShipState.IDLE:
                    continue

                if ship.state == ShipState.LOADING:
                    ship.days_remaining -= dt
                    if ship.days_remaining <= 0:
                        # Start transit
                        ship.state = ShipState.IN_TRANSIT
                        transit_days = self.distance_km / max(ship.speed_km_day, 1.0)
                        ship.days_remaining = transit_days

                elif ship.state == ShipState.IN_TRANSIT:
                    ship.days_remaining -= dt
                    if ship.days_remaining <= 0:
                        # Arrive, start unloading
                        ship.state = ShipState.UNLOADING
                        ship.days_remaining = self.unloading_days

                elif ship.state == ShipState.UNLOADING:
                    ship.days_remaining -= dt
                    if ship.days_remaining <= 0:
                        # Deliver cargo
                        sid = ship.target_storage_id
                        delivered[sid] = delivered.get(sid, 0.0) + ship.cargo
                        self.month_delivered += ship.cargo
                        ship.cargo = 0.0
                        # Start return
                        ship.state = ShipState.RETURN_TRANSIT
                        return_days = self.distance_km / max(ship.speed_km_day, 1.0)
                        ship.days_remaining = return_days

                elif ship.state == ShipState.RETURN_TRANSIT:
                    ship.days_remaining -= dt
                    if ship.days_remaining <= 0:
                        ship.state = ShipState.IDLE
                        ship.days_remaining = 0.0

            t += dt

        # Tick disruption
        if self.is_disrupted:
            self.disruption_remaining_time -= 1.0
            if self.disruption_remaining_time <= 0.0:
                self.is_disrupted = False
                self.disruption_severity = 0.0
                self.disruption_remaining_time = 0.0

        return delivered

    def simulate_day(self) -> Dict[int, float]:
        """Advance all ships by 1 day. Returns delivered cargo."""
        delivered: Dict[int, float] = {}
        for ship in self.fleet:
            if ship.state == ShipState.IDLE:
                continue
            if ship.state == ShipState.LOADING:
                ship.days_remaining -= 1.0
                if ship.days_remaining <= 0:
                    ship.state = ShipState.IN_TRANSIT
                    ship.days_remaining = self.distance_km / max(ship.speed_km_day, 1.0)
            elif ship.state == ShipState.IN_TRANSIT:
                ship.days_remaining -= 1.0
                if ship.days_remaining <= 0:
                    ship.state = ShipState.UNLOADING
                    ship.days_remaining = self.unloading_days
            elif ship.state == ShipState.UNLOADING:
                ship.days_remaining -= 1.0
                if ship.days_remaining <= 0:
                    sid = ship.target_storage_id
                    delivered[sid] = delivered.get(sid, 0.0) + ship.cargo
                    self.month_delivered += ship.cargo
                    ship.cargo = 0.0
                    ship.state = ShipState.RETURN_TRANSIT
                    ship.days_remaining = self.distance_km / max(ship.speed_km_day, 1.0)
            elif ship.state == ShipState.RETURN_TRANSIT:
                ship.days_remaining -= 1.0
                if ship.days_remaining <= 0:
                    ship.state = ShipState.IDLE
                    ship.days_remaining = 0.0
        return delivered

    def end_month(self) -> None:
        """Reset monthly counters."""
        self.month_accepted = 0.0
        self.month_delivered = 0.0
        self.month_cost = 0.0

    def get_in_transit_total(self) -> float:
        """Total cargo currently in transit (loading + transit + unloading)."""
        return sum(
            s.cargo for s in self.fleet
            if s.state in (ShipState.LOADING, ShipState.IN_TRANSIT, ShipState.UNLOADING)
        )

    def get_state(self) -> Dict[str, Any]:
        total_capacity = sum(s.capacity for s in self.fleet) if self.fleet else 1e-9
        return {
            "available_capacity": self.get_monthly_capacity(),
            "capacity": total_capacity,
            "utilization": self.month_accepted / max(total_capacity, 1e-9),
            "in_transit_total": self.get_in_transit_total(),
            "is_disrupted": float(self.is_disrupted),
            "disruption_remaining_time": self.disruption_remaining_time,
            "fleet_size": len(self.fleet),
            "idle_ships": len(self.get_idle_ships()),
        }


# ===================================================================
# RailPhysics
# ===================================================================

@dataclass
class RailTrain:
    """Individual train."""
    capacity: float           # MtCO2 (n_cars * car_capacity)
    state: str = ShipState.IDLE  # reuse same state names
    cargo: float = 0.0
    days_remaining: float = 0.0
    target_storage_id: int = -1

    def reset(self) -> None:
        self.state = ShipState.IDLE
        self.cargo = 0.0
        self.days_remaining = 0.0
        self.target_storage_id = -1


@dataclass
class RailPhysics:
    """Rail transport for CO2 (batch, like ships but shorter cycle)."""

    trains: List[RailTrain] = field(default_factory=list)
    car_capacity: float = 0.00008        # MtCO2 per car
    cars_per_train: int = 80
    loading_time_days: float = 1.0
    unloading_time_days: float = 1.0
    transit_time_days: float = 3.0
    return_time_days: float = 3.0
    cost_per_t_km: float = 0.04
    distance_km: float = 200.0
    base_cost: float = 25.0  # fallback $/tCO2

    # Disruption
    is_disrupted: bool = False
    disruption_remaining_time: float = 0.0
    disruption_severity: float = 0.0

    # Monthly trackers
    month_accepted: float = 0.0
    month_delivered: float = 0.0
    month_cost: float = 0.0

    def reset(self) -> None:
        for train in self.trains:
            train.reset()
        self.is_disrupted = False
        self.disruption_remaining_time = 0.0
        self.disruption_severity = 0.0
        self.month_accepted = 0.0
        self.month_delivered = 0.0
        self.month_cost = 0.0

    def apply_disruption(self, severity: float, duration: float) -> None:
        self.is_disrupted = True
        self.disruption_severity = severity
        self.disruption_remaining_time = duration

    def get_idle_trains(self) -> List[RailTrain]:
        if self.is_disrupted and self.disruption_severity >= 1.0:
            return []
        idle = [t for t in self.trains if t.state == ShipState.IDLE]
        if self.is_disrupted:
            n_available = max(1, int(len(idle) * (1.0 - self.disruption_severity)))
            return idle[:n_available]
        return idle

    def get_monthly_capacity(self) -> float:
        idle = self.get_idle_trains()
        return sum(t.capacity for t in idle)

    def load_train(self, train: RailTrain, volume: float,
                   terminal_buffer: "TerminalBuffer",
                   target_storage_id: int = 0) -> float:
        available_in_buffer = terminal_buffer.level
        initial_load = min(volume, train.capacity, available_in_buffer)
        initial_load = max(initial_load, 0.0)

        if initial_load > 0 or True:
            terminal_buffer.withdraw(initial_load)
            train.cargo = initial_load
            train.state = ShipState.LOADING
            train.days_remaining = self.loading_time_days
            train.target_storage_id = target_storage_id
            self.month_accepted += initial_load
            cost_per_t = self.cost_per_t_km * self.distance_km
            self.month_cost += initial_load * cost_per_t

        return initial_load

    def continue_loading(self, train: RailTrain, terminal_buffer: "TerminalBuffer") -> float:
        """Continue loading a train at berth from buffer inflow."""
        if train.state != ShipState.LOADING:
            return 0.0
        remaining = train.capacity - train.cargo
        if remaining <= 0:
            return 0.0
        available = terminal_buffer.level
        top_up = min(remaining, available)
        if top_up > 0:
            terminal_buffer.withdraw(top_up)
            train.cargo += top_up
            self.month_accepted += top_up
            cost_per_t = self.cost_per_t_km * self.distance_km
            self.month_cost += top_up * cost_per_t
        return top_up

    def simulate_month(self, days: float = 30.0) -> Dict[int, float]:
        """Advance all trains by one month. Returns storage_id -> delivered."""
        delivered: Dict[int, float] = {}
        dt = 1.0

        t = 0.0
        while t < days:
            for train in self.trains:
                if train.state == ShipState.IDLE:
                    continue

                if train.state == ShipState.LOADING:
                    train.days_remaining -= dt
                    if train.days_remaining <= 0:
                        train.state = ShipState.IN_TRANSIT
                        train.days_remaining = self.transit_time_days

                elif train.state == ShipState.IN_TRANSIT:
                    train.days_remaining -= dt
                    if train.days_remaining <= 0:
                        train.state = ShipState.UNLOADING
                        train.days_remaining = self.unloading_time_days

                elif train.state == ShipState.UNLOADING:
                    train.days_remaining -= dt
                    if train.days_remaining <= 0:
                        sid = train.target_storage_id
                        delivered[sid] = delivered.get(sid, 0.0) + train.cargo
                        self.month_delivered += train.cargo
                        train.cargo = 0.0
                        train.state = ShipState.RETURN_TRANSIT
                        train.days_remaining = self.return_time_days

                elif train.state == ShipState.RETURN_TRANSIT:
                    train.days_remaining -= dt
                    if train.days_remaining <= 0:
                        train.state = ShipState.IDLE
                        train.days_remaining = 0.0

            t += dt

        # Tick disruption
        if self.is_disrupted:
            self.disruption_remaining_time -= 1.0
            if self.disruption_remaining_time <= 0.0:
                self.is_disrupted = False
                self.disruption_severity = 0.0
                self.disruption_remaining_time = 0.0

        return delivered

    def simulate_day(self) -> Dict[int, float]:
        """Advance all trains by 1 day. Returns delivered cargo."""
        delivered: Dict[int, float] = {}
        for train in self.trains:
            if train.state == ShipState.IDLE:
                continue
            if train.state == ShipState.LOADING:
                train.days_remaining -= 1.0
                if train.days_remaining <= 0:
                    train.state = ShipState.IN_TRANSIT
                    train.days_remaining = self.transit_time_days
            elif train.state == ShipState.IN_TRANSIT:
                train.days_remaining -= 1.0
                if train.days_remaining <= 0:
                    train.state = ShipState.UNLOADING
                    train.days_remaining = self.unloading_time_days
            elif train.state == ShipState.UNLOADING:
                train.days_remaining -= 1.0
                if train.days_remaining <= 0:
                    sid = train.target_storage_id
                    delivered[sid] = delivered.get(sid, 0.0) + train.cargo
                    self.month_delivered += train.cargo
                    train.cargo = 0.0
                    train.state = ShipState.RETURN_TRANSIT
                    train.days_remaining = self.return_time_days
            elif train.state == ShipState.RETURN_TRANSIT:
                train.days_remaining -= 1.0
                if train.days_remaining <= 0:
                    train.state = ShipState.IDLE
                    train.days_remaining = 0.0
        return delivered

    def end_month(self) -> None:
        self.month_accepted = 0.0
        self.month_delivered = 0.0
        self.month_cost = 0.0

    def get_in_transit_total(self) -> float:
        return sum(
            t.cargo for t in self.trains
            if t.state in (ShipState.LOADING, ShipState.IN_TRANSIT, ShipState.UNLOADING)
        )

    def get_state(self) -> Dict[str, Any]:
        total_cap = sum(t.capacity for t in self.trains) if self.trains else 1e-9
        return {
            "available_capacity": self.get_monthly_capacity(),
            "capacity": total_cap,
            "utilization": self.month_accepted / max(total_cap, 1e-9),
            "in_transit_total": self.get_in_transit_total(),
            "is_disrupted": float(self.is_disrupted),
            "disruption_remaining_time": self.disruption_remaining_time,
            "fleet_size": len(self.trains),
            "idle_trains": len(self.get_idle_trains()),
        }


# ===================================================================
# TerminalBuffer
# ===================================================================

@dataclass
class TerminalBuffer:
    """Buffer tank at port/rail terminal.

    CO2 must flow through this buffer before loading onto ship/rail.
    Pipeline transport bypasses the terminal.
    """

    id: str                        # e.g. "port_0", "rail_terminal_0"
    max_capacity: float = 0.0084   # MtCO2 (~8400 tonnes, Northern Lights scale)

    # Mutable state
    level: float = 0.0

    def reset(self) -> None:
        self.level = 0.0

    def deposit(self, volume: float) -> Tuple[float, float]:
        """Deposit CO2 into buffer. Returns (accepted, overflow_vented)."""
        space = max(0.0, self.max_capacity - self.level)
        accepted = min(volume, space)
        overflow = max(0.0, volume - space)
        self.level += accepted
        return accepted, overflow

    def withdraw(self, volume: float) -> float:
        """Withdraw CO2 from buffer. Returns actual withdrawn."""
        actual = min(volume, self.level)
        self.level -= actual
        return actual

    def get_fill_frac(self) -> float:
        return self.level / max(self.max_capacity, 1e-9)

    def get_state(self) -> Dict[str, float]:
        return {
            "level": self.level,
            "max_capacity": self.max_capacity,
            "fill_frac": self.get_fill_frac(),
        }


# ===================================================================
# StoragePhysics
# ===================================================================

@dataclass
class StoragePhysics:
    """CO2 injection site with reduced-order pressure model.

    Supports two modes:
    1. Analytical ROM (default):
        P(t+1) = P(t) + k_inj * Q_inj / injectivity - k_diss * (P - P_hydro) + cross_well
        Pressures in bar, rates in MtCO2/month.

    2. Proxy model (when proxy_model is provided):
        Uses a trained ML model to predict pressure changes (in Pa/psi).
        See StorageProxyModel for details.
    """

    id: int
    max_injection_rate: float
    pressure_limit: float
    initial_pressure: float
    k_injection: float
    k_dissipation: float
    cross_well_coeff: float
    injectivity_decline_rate: float
    cumulative_capacity: float
    min_purity: float = 0.93
    max_impurities: Dict[str, float] = field(default_factory=dict)

    # Mutable state (ROM mode: pressures in bar)
    current_pressure: float = 0.0
    cumulative_injected: float = 0.0
    current_injectivity: float = 1.0
    is_disrupted: bool = False
    disruption_remaining_time: float = 0.0
    disruption_severity: float = 0.0
    last_inlet_purity: float = 1.0
    last_quality_penalty: float = 0.0
    last_quality_violation: bool = False

    # Proxy model support (set after __init__ via attach_proxy)
    proxy_model: Optional[Any] = field(default=None, repr=False)
    use_proxy: bool = False

    # Proxy-mode state (pressures in Pa)
    _dome_pressure_pa: float = 0.0
    _bhp_pa: float = 0.0
    _cumulative_stored_t: float = 0.0
    _num_wells: int = 1

    def __post_init__(self) -> None:
        self.current_pressure = self.initial_pressure
        self.cumulative_injected = 0.0
        self.current_injectivity = 1.0
        if not self.max_impurities:
            self.max_impurities = dict(DEFAULT_STORAGE_QUALITY_LIMITS["max_impurities"])
        # Proxy state initialised later via attach_proxy()

    def attach_proxy(self, proxy_model: "StorageProxyModel") -> None:
        """Attach a StorageProxyModel and switch to proxy mode.

        This must be called after dataclass construction because dataclass
        fields with defaults cannot precede those without.
        """
        self.proxy_model = proxy_model
        self.use_proxy = proxy_model is not None and proxy_model.is_loaded

        if self.use_proxy:
            sp = proxy_model.site_params
            self._dome_pressure_pa = sp["initial_dome_pressure_mpa"] * 1.0e6
            self._bhp_pa = sp["initial_bhp_mpa"] * 1.0e6
            self._cumulative_stored_t = 0.0
            self._num_wells = int(sp.get("num_wells", 1))

            # Also update the bar-based pressure for observation compatibility
            # (current_pressure in bar for observations)
            self.current_pressure = self._dome_pressure_pa / 1.0e5  # Pa -> bar
            self.initial_pressure = self.current_pressure

            # Update pressure limit from proxy model (in bar)
            self.pressure_limit = proxy_model.dome_pressure_limit_pa / 1.0e5

            logger.info(
                "StoragePhysics[%d]: proxy attached, init dome=%.2f MPa, "
                "BHP=%.2f MPa, limit=%.2f bar",
                self.id,
                self._dome_pressure_pa / 1.0e6,
                self._bhp_pa / 1.0e6,
                self.pressure_limit,
            )

    def reset(self) -> None:
        self.current_pressure = self.initial_pressure
        self.cumulative_injected = 0.0
        self.current_injectivity = 1.0
        self.is_disrupted = False
        self.disruption_remaining_time = 0.0
        self.disruption_severity = 0.0
        self.last_inlet_purity = 1.0
        self.last_quality_penalty = 0.0
        self.last_quality_violation = False

        if self.use_proxy and self.proxy_model is not None:
            sp = self.proxy_model.site_params
            self._dome_pressure_pa = sp["initial_dome_pressure_mpa"] * 1.0e6
            self._bhp_pa = sp["initial_bhp_mpa"] * 1.0e6
            self._cumulative_stored_t = 0.0
            self.current_pressure = self._dome_pressure_pa / 1.0e5
            self.initial_pressure = self.current_pressure

    def apply_disruption(self, severity: float, duration: float) -> None:
        self.is_disrupted = True
        self.disruption_severity = severity
        self.disruption_remaining_time = duration

    def get_max_injectable(self) -> float:
        if self.is_disrupted and self.disruption_severity >= 1.0:
            return 0.0

        if self.use_proxy and self.proxy_model is not None:
            return self._get_max_injectable_proxy()

        return self._get_max_injectable_rom()

    def _get_max_injectable_rom(self) -> float:
        """Max injectable using analytical ROM."""
        pressure_headroom = self.pressure_limit - self.current_pressure
        if pressure_headroom <= 0.0:
            return 0.0

        inj = max(self.current_injectivity, 0.1)
        q_pressure_limited = pressure_headroom * inj / max(self.k_injection, 1e-9)

        effective_max = self.max_injection_rate
        if self.is_disrupted:
            effective_max *= (1.0 - self.disruption_severity)

        remaining_capacity = max(0.0, self.cumulative_capacity - self.cumulative_injected)

        return max(0.0, min(q_pressure_limited, effective_max, remaining_capacity))

    def _get_max_injectable_proxy(self) -> float:
        """Max injectable using proxy model's predicted safe rate."""
        assert self.proxy_model is not None

        # Get max safe daily rate per well (tonnes/day)
        max_rate_t_day = self.proxy_model.predict_max_safe_rate(
            self._dome_pressure_pa,
            self._bhp_pa,
            self._cumulative_stored_t / max(self._num_wells, 1),
        )

        # Convert to MtCO2/month (all wells combined)
        max_mt_month = max_rate_t_day * self._num_wells * 30.0 / 1.0e6

        # Apply disruption reduction
        if self.is_disrupted:
            max_mt_month *= (1.0 - self.disruption_severity)

        # Capacity limit
        remaining_capacity = max(
            0.0, self.cumulative_capacity - self.cumulative_injected
        )

        return max(0.0, min(max_mt_month, remaining_capacity))

    def inject(
        self,
        injection_rate: float,
        other_site_rates: Optional[List[float]] = None,
        cross_well_scale: float = 1.0,
        pressure_limit_scale: float = 1.0,
    ) -> Tuple[float, bool]:
        """Inject CO2 for one month.

        Returns (actual_injected, pressure_violation).
        """
        if self.use_proxy and self.proxy_model is not None:
            return self._inject_proxy(injection_rate)

        return self._inject_rom(
            injection_rate, other_site_rates,
            cross_well_scale, pressure_limit_scale,
        )

    def _inject_rom(
        self,
        injection_rate: float,
        other_site_rates: Optional[List[float]] = None,
        cross_well_scale: float = 1.0,
        pressure_limit_scale: float = 1.0,
    ) -> Tuple[float, bool]:
        """Inject using the analytical ROM. Original logic preserved."""
        effective_pressure_limit = self.pressure_limit * pressure_limit_scale

        max_q = self._get_max_injectable_rom()
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

        delta_p_cross = 0.0
        if other_site_rates is not None:
            effective_cross_coeff = self.cross_well_coeff * cross_well_scale
            for q_other in other_site_rates:
                delta_p_cross += effective_cross_coeff * q_other

        self.current_pressure += delta_p_injection - delta_p_dissipation + delta_p_cross
        self.cumulative_injected += actual_q

        pressure_violation = self.current_pressure > effective_pressure_limit

        # Tick disruption
        if self.is_disrupted:
            self.disruption_remaining_time -= 1.0
            if self.disruption_remaining_time <= 0.0:
                self.is_disrupted = False
                self.disruption_severity = 0.0
                self.disruption_remaining_time = 0.0

        return actual_q, pressure_violation

    def _inject_proxy(self, desired_rate_mt: float) -> Tuple[float, bool]:
        """Inject using the proxy model.

        Args:
            desired_rate_mt: Desired injection rate in MtCO2/month.

        Returns:
            (actual_stored_mt, pressure_violation)
        """
        assert self.proxy_model is not None

        # Convert MtCO2/month to tonnes/day per well
        rate_t_day = desired_rate_mt * 1.0e6 / 30.0 / max(self._num_wells, 1)

        # Check against predicted max safe rate
        max_safe_t_day = self.proxy_model.predict_max_safe_rate(
            self._dome_pressure_pa,
            self._bhp_pa,
            self._cumulative_stored_t / max(self._num_wells, 1),
        )

        # Apply disruption reduction to max safe rate
        if self.is_disrupted:
            max_safe_t_day *= (1.0 - self.disruption_severity)

        actual_rate_t_day = min(rate_t_day, max_safe_t_day)
        actual_rate_t_day = max(actual_rate_t_day, 0.0)

        # Get monthly pressure update from the proxy model
        try:
            result = self.proxy_model.predict_monthly_update(
                actual_rate_t_day,
                self._dome_pressure_pa,
                self._bhp_pa,
                self._cumulative_stored_t / max(self._num_wells, 1),
            )
        except RuntimeError:
            # Proxy not loaded -- should not happen if use_proxy is True,
            # but handle gracefully
            logger.error("Proxy prediction failed in _inject_proxy")
            self._tick_disruption()
            return 0.0, False

        # Update proxy-mode state
        self._dome_pressure_pa += result["delta_dome_pressure_pa"]
        self._bhp_pa += result["delta_bhp_pa"]

        # Actual stored amount (per well -> total, convert to tonnes)
        stored_per_well_t = result["delta_stored_t"]
        total_stored_t = stored_per_well_t * self._num_wells
        self._cumulative_stored_t += total_stored_t

        # Convert to MtCO2
        actual_stored_mt = total_stored_t / 1.0e6
        self.cumulative_injected += actual_stored_mt

        # Update bar-based state for observation compatibility
        self.current_pressure = self._dome_pressure_pa / 1.0e5  # Pa -> bar

        # Check pressure limits
        violation = (
            self._dome_pressure_pa > self.proxy_model.dome_pressure_limit_pa
            or self._bhp_pa > self.proxy_model.bhp_limit_pa
        )

        # Tick disruption
        self._tick_disruption()

        return actual_stored_mt, violation

    def _tick_disruption(self) -> None:
        """Decrement disruption timer."""
        if self.is_disrupted:
            self.disruption_remaining_time -= 1.0
            if self.disruption_remaining_time <= 0.0:
                self.is_disrupted = False
                self.disruption_severity = 0.0
                self.disruption_remaining_time = 0.0

    def get_state(self) -> Dict[str, float]:
        if self.use_proxy and self.proxy_model is not None:
            return self._get_state_proxy()
        return self._get_state_rom()

    def _get_state_rom(self) -> Dict[str, float]:
        """State dict for analytical ROM mode."""
        return {
            "current_pressure": self.current_pressure,
            "pressure_frac": self.current_pressure / max(self.pressure_limit, 1e-9),
            "cumulative_injected": self.cumulative_injected,
            "fill_frac": self.cumulative_injected / max(self.cumulative_capacity, 1e-9),
            "current_injectivity": self.current_injectivity,
            "max_injectable": self.get_max_injectable(),
            "last_inlet_purity": self.last_inlet_purity,
            "last_quality_penalty": self.last_quality_penalty,
            "is_disrupted": float(self.is_disrupted),
            "disruption_remaining_time": self.disruption_remaining_time,
        }

    def _get_state_proxy(self) -> Dict[str, float]:
        """State dict for proxy model mode.

        Includes additional proxy-specific fields (bhp, dome pressure in MPa).
        The bar-based fields remain for compatibility with the observation layer.
        """
        assert self.proxy_model is not None
        return {
            "current_pressure": self._dome_pressure_pa / 1.0e6,  # MPa
            "bhp": self._bhp_pa / 1.0e6,  # MPa
            "pressure_frac": (
                self._dome_pressure_pa / max(self.proxy_model.dome_pressure_limit_pa, 1.0)
            ),
            "cumulative_injected": self.cumulative_injected,  # Mt
            "fill_frac": self.cumulative_injected / max(self.cumulative_capacity, 1e-9),
            "current_injectivity": 1.0,  # proxy handles injectivity internally
            "max_injectable": self.get_max_injectable(),
            "last_inlet_purity": self.last_inlet_purity,
            "last_quality_penalty": self.last_quality_penalty,
            "is_disrupted": float(self.is_disrupted),
            "disruption_remaining_time": self.disruption_remaining_time,
        }


# ===================================================================
# PhysicalLayer
# ===================================================================

class PhysicalLayer:
    """Wraps all physical components. Executes monthly settlement.

    Accepts decision intents from the decision layer (env) and returns
    physical outcomes. No agent/RL logic here.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        net_cfg = config["network"]

        # --- Build emitters ---
        self.emitters: Dict[int, EmitterPhysics] = {}
        e_defaults = net_cfg.get("emitter_defaults", {})
        e_overrides = net_cfg.get("emitter_params", {})
        for i in range(net_cfg["num_emitters"]):
            params = {**e_defaults, **e_overrides.get(i, {})}
            self.emitters[i] = EmitterPhysics(
                id=i,
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
            )

        # --- Build transport ---
        self.pipeline: Optional[PipelinePhysics] = None
        self.ships: Optional[ShipPhysics] = None
        self.rail: Optional[RailPhysics] = None

        # Terminal buffers (one per port/rail terminal)
        self.terminal_buffers: Dict[str, TerminalBuffer] = {}

        t_params = net_cfg.get("transport_params", {})
        transport_modes = net_cfg.get("transport_modes", [])
        physical_cfg = config.get("physical", {})

        # Transport mode index mapping (for compatibility with connectivity)
        self._mode_index: Dict[str, int] = {}
        for idx, mode_name in enumerate(transport_modes):
            self._mode_index[mode_name] = idx

        if "pipeline" in transport_modes:
            pp = t_params.get("pipeline", {})
            self.pipeline = PipelinePhysics(
                capacity=pp.get("capacity", 0.5),
                base_cost=pp.get("base_cost", 10.0),
            )

        if "ship" in transport_modes:
            sp = t_params.get("ship", {})
            ship_cfg = physical_cfg.get("ship", {})
            fleet_spec = ship_cfg.get("fleet", [{"type": "medium", "count": 3}])
            distance = ship_cfg.get("distance_km", 800.0)
            loading_days = ship_cfg.get("loading_days", 2.0)
            unloading_days = ship_cfg.get("unloading_days", 2.0)

            fleet: List[Ship] = []
            for spec in fleet_spec:
                ship_type_name = spec.get("type", "medium")
                count = spec.get("count", 1)
                st = SHIP_TYPES.get(ship_type_name, SHIP_TYPES["medium"])
                for _ in range(count):
                    fleet.append(Ship(
                        ship_type=ship_type_name,
                        capacity=st["capacity"],
                        speed_km_day=st["speed_km_day"],
                        loading_rate=st["loading_rate"],
                        cost_per_t=st["cost_per_t"],
                    ))

            self.ships = ShipPhysics(
                fleet=fleet,
                distance_km=distance,
                loading_days=loading_days,
                unloading_days=unloading_days,
                base_cost=sp.get("base_cost", 20.0),
            )

            # Create port terminal buffer
            buf_cfg = physical_cfg.get("terminal_buffer_ship", {})
            self.terminal_buffers["ship"] = TerminalBuffer(
                id="ship",
                max_capacity=buf_cfg.get("max_capacity", 0.0084),
            )

        if "rail" in transport_modes:
            rp = t_params.get("rail", {})
            rail_cfg = physical_cfg.get("rail", {})

            car_cap = rail_cfg.get("car_capacity", RAIL_DEFAULTS["car_capacity"])
            cars = rail_cfg.get("cars_per_train", RAIL_DEFAULTS["cars_per_train"])
            n_trains = rail_cfg.get("num_trains", RAIL_DEFAULTS["num_trains"])
            train_cap = car_cap * cars

            trains: List[RailTrain] = []
            for _ in range(n_trains):
                trains.append(RailTrain(capacity=train_cap))

            self.rail = RailPhysics(
                trains=trains,
                car_capacity=car_cap,
                cars_per_train=cars,
                loading_time_days=rail_cfg.get("loading_time_days",
                                                RAIL_DEFAULTS["loading_time_days"]),
                unloading_time_days=rail_cfg.get("unloading_time_days",
                                                  RAIL_DEFAULTS["unloading_time_days"]),
                transit_time_days=rail_cfg.get("transit_time_days",
                                               RAIL_DEFAULTS["transit_time_days"]),
                return_time_days=rail_cfg.get("return_time_days",
                                              RAIL_DEFAULTS["transit_time_days"]),
                cost_per_t_km=rail_cfg.get("cost_per_t_km",
                                           RAIL_DEFAULTS["cost_per_t_km"]),
                distance_km=rail_cfg.get("distance_km",
                                         RAIL_DEFAULTS["distance_km"]),
                base_cost=rp.get("base_cost", 25.0),
            )

            # Create rail terminal buffer
            buf_cfg = physical_cfg.get("terminal_buffer_rail", {})
            self.terminal_buffers["rail"] = TerminalBuffer(
                id="rail",
                max_capacity=buf_cfg.get("max_capacity", 0.005),
            )

        # --- Build storage sites ---
        self.storage_sites: Dict[int, StoragePhysics] = {}
        s_defaults = net_cfg.get("storage_defaults", {})
        s_overrides = net_cfg.get("storage_params", {})
        proxy_models_cfg = config.get("proxy_models", {})

        for i in range(net_cfg["num_storage_sites"]):
            params = {**s_defaults, **s_overrides.get(i, {})}
            site = StoragePhysics(
                id=i,
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
            )

            # Attach proxy model if configured for this storage site
            if i in proxy_models_cfg:
                pcfg = proxy_models_cfg[i]
                try:
                    from ccus_gym.storage_proxy import StorageProxyModel
                    proxy = StorageProxyModel(
                        model_path=pcfg["path"],
                        site_params=pcfg["site_params"],
                    )
                    site.attach_proxy(proxy)
                    logger.info("Storage site %d: proxy model attached", i)
                except Exception as e:
                    logger.warning(
                        "Storage site %d: failed to create proxy model (%s). "
                        "Falling back to analytical ROM.",
                        i, e,
                    )

            self.storage_sites[i] = site

        # --- Connectivity ---
        connectivity_raw = net_cfg.get("connectivity", {})
        self.connectivity: Dict[int, List[Tuple[int, int]]] = {}
        for eid_str, routes in connectivity_raw.items():
            eid = int(eid_str)
            self.connectivity[eid] = [(int(r[0]), int(r[1])) for r in routes]

        # Reverse map: transport_idx -> [storage_ids]
        self._transport_to_storage: Dict[int, List[int]] = {}
        for routes in self.connectivity.values():
            for t_id, s_id in routes:
                self._transport_to_storage.setdefault(t_id, [])
                if s_id not in self._transport_to_storage[t_id]:
                    self._transport_to_storage[t_id].append(s_id)

        # Mode name to index and back
        self._idx_to_mode: Dict[int, str] = {v: k for k, v in self._mode_index.items()}

    def reset(self) -> None:
        """Reset all physical components."""
        for e in self.emitters.values():
            e.reset()
        if self.pipeline is not None:
            self.pipeline.reset()
        if self.ships is not None:
            self.ships.reset()
        if self.rail is not None:
            self.rail.reset()
        for buf in self.terminal_buffers.values():
            buf.reset()
        for s in self.storage_sites.values():
            s.reset()

    def settle(
        self,
        decisions: Dict[str, Any],
        cross_well_scale: float = 1.0,
        pressure_limit_scale: float = 1.0,
    ) -> PhysicalOutcome:
        """Execute one month of physical operations given settled decisions.

        decisions dict structure:
            "emitter_nominations": {eid: [(transport_idx, storage_id, volume), ...]}
            "transport_thresholds": {mode_name: threshold_frac}
            "storage_injection_fracs": {sid: frac}
            "transport_posted_prices": {mode_name: price_$/tCO2}  (optional)
            "emitter_bids": {eid: bid_$/tCO2}                     (optional)
            "transport_params": {mode_name: {extra params}}        (optional)
        """
        outcome = PhysicalOutcome()

        emitter_nominations = decisions.get("emitter_nominations", {})
        emitter_capture_fracs = decisions.get("emitter_capture_fracs", {})
        emitter_purification_efforts = decisions.get("emitter_purification_efforts", {})
        transport_thresholds = decisions.get("transport_thresholds", {})
        transport_quality_thresholds = decisions.get("transport_quality_thresholds", {})
        storage_injection_fracs = decisions.get("storage_injection_fracs", {})
        storage_quality_targets = decisions.get("storage_quality_targets", {})
        posted_prices = decisions.get("transport_posted_prices", {})
        emitter_bids = decisions.get("emitter_bids", {})
        transport_extra = decisions.get("transport_params", {})

        use_pricing = len(posted_prices) > 0

        for eid, emitter in self.emitters.items():
            emitter.set_capture_controls(
                emitter_capture_fracs.get(eid, 1.0),
                emitter_purification_efforts.get(eid, 0.0),
            )
            outcome.emitter_effective_purity[eid] = emitter.last_effective_purity

        # ---- Step 1: Nomination & Settlement ----
        # Aggregate nominations per transport mode
        mode_nominations: Dict[str, List[Tuple[int, int, float]]] = {}
        quality_rejected_by_mode: Dict[str, float] = {}
        # mode_name -> [(emitter_id, storage_id, volume)]
        for eid, routes in emitter_nominations.items():
            for tid, sid, vol in routes:
                mode_name = self._idx_to_mode.get(tid, "pipeline")
                purity = self.emitters[eid].last_effective_purity
                quality_threshold = transport_quality_thresholds.get(mode_name, 0.0)
                quality_gate = 1.0
                if quality_threshold > 0.0 and purity < quality_threshold:
                    quality_gate = max(0.0, purity / max(quality_threshold, 1e-9))
                    quality_rejected_by_mode[mode_name] = (
                        quality_rejected_by_mode.get(mode_name, 0.0)
                        + max(0.0, vol - vol * quality_gate)
                    )
                mode_nominations.setdefault(mode_name, [])
                mode_nominations[mode_name].append((eid, sid, vol * quality_gate))

        # Compute available capacity per mode
        mode_capacities: Dict[str, float] = {}
        if self.pipeline is not None:
            threshold = transport_thresholds.get("pipeline", 1.0)
            mode_capacities["pipeline"] = threshold * self.pipeline.available_capacity
        if self.ships is not None:
            threshold = transport_thresholds.get("ship", 1.0)
            mode_capacities["ship"] = threshold * self.ships.get_monthly_capacity()
        if self.rail is not None:
            threshold = transport_thresholds.get("rail", 1.0)
            mode_capacities["rail"] = threshold * self.rail.get_monthly_capacity()

        # Settlement: price-priority if pricing is active, else pro-rata
        settled_flows: Dict[str, List[Tuple[int, int, float]]] = {}
        # mode_name -> [(eid, sid, actual_volume)]

        if use_pricing:
            settled_flows = self._settle_with_pricing(
                mode_nominations, mode_capacities,
                posted_prices, emitter_bids, outcome,
            )
        else:
            # Legacy pro-rata settlement
            for mode_name, noms in mode_nominations.items():
                total_nominated = sum(v for _, _, v in noms)
                capacity = mode_capacities.get(mode_name, 0.0)

                if total_nominated <= 0:
                    scarcity = 1.0
                elif total_nominated <= capacity:
                    scarcity = 1.0
                else:
                    scarcity = capacity / total_nominated

                outcome.scarcity_ratios[mode_name] = scarcity

                settled = []
                for eid, sid, vol in noms:
                    actual = vol * scarcity
                    settled.append((eid, sid, actual))
                settled_flows[mode_name] = settled

        # ---- Step 2: Pipeline - instant flow ----
        total_transport_cost = 0.0
        pipeline_delivered: Dict[int, float] = {}  # storage_id -> vol
        pipeline_per_emitter: Dict[int, float] = {}  # eid -> accepted vol (for congestion)

        if self.pipeline is not None and "pipeline" in settled_flows:
            for eid, sid, vol in settled_flows["pipeline"]:
                accepted, cost = self.pipeline.accept_and_deliver(vol)
                pipeline_delivered[sid] = pipeline_delivered.get(sid, 0.0) + accepted
                pipeline_per_emitter[eid] = pipeline_per_emitter.get(eid, 0.0) + accepted
                total_transport_cost += cost
                outcome.emitter_sent[eid] = outcome.emitter_sent.get(eid, 0.0) + accepted
                outcome.emitter_transport_cost[eid] = (
                    outcome.emitter_transport_cost.get(eid, 0.0) + cost)
                outcome.transport_revenue["pipeline"] = (
                    outcome.transport_revenue.get("pipeline", 0.0) + cost)

            # Congestion pricing surcharge
            pipe_util = self.pipeline.month_accepted / max(self.pipeline.capacity, 1e-9)
            cong_thresh = self.config.get("congestion_threshold", 0.8)
            k_cong = self.config.get("k_congestion", 5.0)
            if pipe_util > cong_thresh:
                surcharge_rate = self.pipeline.base_cost * (pipe_util - cong_thresh) * k_cong
                outcome.pipeline_congestion_mult = 1.0 + (pipe_util - cong_thresh) * k_cong
                for eid, vol in pipeline_per_emitter.items():
                    surcharge = vol * surcharge_rate
                    total_transport_cost += surcharge
                    outcome.emitter_transport_cost[eid] = (
                        outcome.emitter_transport_cost.get(eid, 0.0) + surcharge)
                    outcome.pipeline_congestion_surcharge[eid] = surcharge
                    outcome.transport_revenue["pipeline"] = (
                        outcome.transport_revenue.get("pipeline", 0.0) + surcharge)

            outcome.transport_accepted["pipeline"] = self.pipeline.month_accepted
            outcome.transport_delivered["pipeline"] = self.pipeline.month_delivered

        # ---- Step 3: Ship/Rail - CO2 flows to terminal buffer, then loads ----

        # Ship: emitter -> terminal buffer -> ship (daily interleaved simulation)
        if self.ships is not None:
            ship_buf = self.terminal_buffers.get("ship")
            buffer_vented = 0.0
            ship_delivered: Dict[int, float] = {}

            # Get dispatch threshold from transport params
            dispatch_thresh = transport_extra.get("ship", {}).get(
                "dispatch_threshold", 0.3) if transport_extra else 0.3
            dest_pref = transport_extra.get("ship", {}).get(
                "destination_pref", 0.5) if transport_extra else 0.5

            if "ship" in settled_flows and ship_buf is not None:
                # Compute daily CO2 delivery to terminal (spread over 30 days)
                daily_deliveries: Dict[int, float] = {}  # eid -> daily volume
                daily_emitter_share: Dict[int, float] = {}  # track per-emitter fraction
                total_ship_nom = sum(v for _, _, v in settled_flows["ship"])
                for eid, sid, vol in settled_flows["ship"]:
                    daily_deliveries[eid] = vol / 30.0
                    daily_emitter_share[eid] = vol / max(total_ship_nom, 1e-12)
                    # NOTE: emitter_sent is tracked via actual buffer acceptance below

                # Day-by-day simulation: deposit CO2 + dispatch ships + advance fleet
                ship_actually_sent: Dict[int, float] = {}  # track actual accepted per emitter
                for day in range(30):
                    # Daily CO2 arrives at terminal buffer
                    for eid, daily_vol in daily_deliveries.items():
                        accepted, overflow = ship_buf.deposit(daily_vol)
                        buffer_vented += overflow
                        ship_actually_sent[eid] = ship_actually_sent.get(eid, 0.0) + accepted

                    # Continue loading ships already at berth (top up from daily buffer inflow)
                    for ship in self.ships.fleet:
                        if ship.state == ShipState.LOADING:
                            self.ships.continue_loading(ship, ship_buf)

                    # Try to load idle ships when buffer reaches dispatch threshold
                    if ship_buf.level >= 0.001:
                        idle_ships = self.ships.get_idle_ships()
                        for ship in idle_ships:
                            if ship_buf.level <= 0.0001:
                                break
                            storage_ids = list(self.storage_sites.keys())
                            if len(storage_ids) > 1:
                                target_sid = storage_ids[1] if dest_pref > 0.5 else storage_ids[0]
                            else:
                                target_sid = storage_ids[0] if storage_ids else 0
                            self.ships.load_ship(ship, ship.capacity, ship_buf, target_sid)

                    # Advance all ships by 1 day
                    day_delivered = self.ships.simulate_day()
                    for sid, vol in day_delivered.items():
                        ship_delivered[sid] = ship_delivered.get(sid, 0.0) + vol
            else:
                # No settled flows but still advance ships in transit
                for day in range(30):
                    day_delivered = self.ships.simulate_day()
                    for sid, vol in day_delivered.items():
                        ship_delivered[sid] = ship_delivered.get(sid, 0.0) + vol

            # Record actual emitter_sent for ship route (only what buffer accepted)
            if "ship" in settled_flows:
                total_ship_sent = sum(ship_actually_sent.values())
                for eid, vol_sent in ship_actually_sent.items():
                    outcome.emitter_sent[eid] = outcome.emitter_sent.get(eid, 0.0) + vol_sent
                    # Per-emitter ship cost: proportional share of fleet cost
                    if total_ship_sent > 1e-12:
                        eid_cost = self.ships.month_cost * (vol_sent / total_ship_sent)
                    else:
                        eid_cost = 0.0
                    outcome.emitter_transport_cost[eid] = (
                        outcome.emitter_transport_cost.get(eid, 0.0) + eid_cost)

            total_transport_cost += self.ships.month_cost
            outcome.transport_revenue["ship"] = (
                outcome.transport_revenue.get("ship", 0.0) + self.ships.month_cost)

            outcome.transport_accepted["ship"] = self.ships.month_accepted
            total_ship_delivered = sum(ship_delivered.values())
            outcome.transport_delivered["ship"] = total_ship_delivered

            if ship_buf is not None:
                outcome.terminal_buffer_levels["ship"] = ship_buf.get_fill_frac()
                outcome.terminal_vented["ship"] = buffer_vented

            # Add ship deliveries to storage
            for sid, vol in ship_delivered.items():
                pipeline_delivered[sid] = pipeline_delivered.get(sid, 0.0) + vol

        # Rail: emitter -> terminal buffer -> rail (daily interleaved simulation)
        if self.rail is not None:
            rail_buf = self.terminal_buffers.get("rail")
            buffer_vented = 0.0
            rail_delivered: Dict[int, float] = {}

            rail_dispatch_thresh = transport_extra.get("rail", {}).get(
                "dispatch_threshold", 0.3) if transport_extra else 0.3
            rail_load_frac = transport_extra.get("rail", {}).get(
                "train_load_frac", 0.7) if transport_extra else 0.7

            if "rail" in settled_flows and rail_buf is not None:
                daily_deliveries_rail: Dict[int, float] = {}
                for eid, sid, vol in settled_flows["rail"]:
                    daily_deliveries_rail[eid] = vol / 30.0
                    # NOTE: emitter_sent tracked via actual buffer acceptance below

                rail_actually_sent: Dict[int, float] = {}
                for day in range(30):
                    for eid, daily_vol in daily_deliveries_rail.items():
                        accepted, overflow = rail_buf.deposit(daily_vol)
                        buffer_vented += overflow
                        rail_actually_sent[eid] = rail_actually_sent.get(eid, 0.0) + accepted

                    # Continue loading trains already at berth
                    for train in self.rail.trains:
                        if train.state == ShipState.LOADING:
                            self.rail.continue_loading(train, rail_buf)

                    # Dispatch idle trains when buffer has enough
                    if rail_buf.level >= 0.001:
                        idle_trains = self.rail.get_idle_trains()
                        for train in idle_trains:
                            if rail_buf.level <= 0.0001:
                                break
                            target_sid = self._pick_storage_for_mode("rail")
                            self.rail.load_train(train, train.capacity, rail_buf, target_sid)

                    day_delivered = self.rail.simulate_day()
                    for sid, vol in day_delivered.items():
                        rail_delivered[sid] = rail_delivered.get(sid, 0.0) + vol
            else:
                for day in range(30):
                    day_delivered = self.rail.simulate_day()
                    for sid, vol in day_delivered.items():
                        rail_delivered[sid] = rail_delivered.get(sid, 0.0) + vol

            # Record actual emitter_sent for rail route
            if "rail" in settled_flows:
                total_rail_sent = sum(rail_actually_sent.values())
                for eid, vol_sent in rail_actually_sent.items():
                    outcome.emitter_sent[eid] = outcome.emitter_sent.get(eid, 0.0) + vol_sent
                    # Per-emitter rail cost: proportional share
                    if total_rail_sent > 1e-12:
                        eid_cost = self.rail.month_cost * (vol_sent / total_rail_sent)
                    else:
                        eid_cost = 0.0
                    outcome.emitter_transport_cost[eid] = (
                        outcome.emitter_transport_cost.get(eid, 0.0) + eid_cost)

            total_transport_cost += self.rail.month_cost
            outcome.transport_revenue["rail"] = (
                outcome.transport_revenue.get("rail", 0.0) + self.rail.month_cost)

            outcome.transport_accepted["rail"] = self.rail.month_accepted
            total_rail_delivered = sum(rail_delivered.values())
            outcome.transport_delivered["rail"] = total_rail_delivered

            if rail_buf is not None:
                outcome.terminal_buffer_levels["rail"] = rail_buf.get_fill_frac()
                outcome.terminal_vented["rail"] = buffer_vented

            for sid, vol in rail_delivered.items():
                pipeline_delivered[sid] = pipeline_delivered.get(sid, 0.0) + vol

        # ---- Step 4: Storage injection ----
        storage_ids_sorted = sorted(self.storage_sites.keys())
        planned_rates: Dict[int, float] = {}

        # Track emitter nomination shares per storage (for overflow attribution)
        emitter_nom_to_storage: Dict[int, Dict[int, float]] = {}  # sid -> {eid: vol}
        for mode_name, flows in settled_flows.items():
            for eid, sid, vol in flows:
                if sid not in emitter_nom_to_storage:
                    emitter_nom_to_storage[sid] = {}
                emitter_nom_to_storage[sid][eid] = (
                    emitter_nom_to_storage[sid].get(eid, 0.0) + vol)

        storage_overflow_vented = 0.0  # CO2 delivered but not injectable
        for sid in storage_ids_sorted:
            site = self.storage_sites[sid]
            frac = storage_injection_fracs.get(sid, 1.0)
            max_q = site.get_max_injectable()
            delivered = pipeline_delivered.get(sid, 0.0)
            blend_inputs = []
            if sid in emitter_nom_to_storage:
                for eid, nom_vol in emitter_nom_to_storage[sid].items():
                    blend_inputs.append((nom_vol, self.emitters[eid].last_composition))
            blended_comp = blend_streams(blend_inputs) if blend_inputs else normalize_composition({"co2": 1.0})
            purity_target = storage_quality_targets.get(sid, site.min_purity)
            quality_penalty, quality_violation = storage_quality_penalty(
                blended_comp,
                min_purity=purity_target,
                max_impurities=site.max_impurities,
            )
            quality_factor = max(0.0, 1.0 - min(0.9, 4.0 * quality_penalty))
            quality_limited_delivered = delivered * quality_factor
            desired_q = min(quality_limited_delivered, frac * max_q)
            overflow = max(0.0, delivered - desired_q)
            storage_overflow_vented += overflow
            outcome.storage_overflow[sid] = overflow
            outcome.overflow_attributed_storage[sid] = overflow
            outcome.storage_inlet_purity[sid] = blended_comp.get("co2", 1.0)
            outcome.storage_quality_penalty[sid] = quality_penalty
            outcome.storage_quality_violation[sid] = quality_violation
            site.last_inlet_purity = blended_comp.get("co2", 1.0)
            site.last_quality_penalty = quality_penalty
            site.last_quality_violation = quality_violation

            # Attribute overflow to emitters proportionally
            if overflow > 0 and sid in emitter_nom_to_storage:
                total_nom = sum(emitter_nom_to_storage[sid].values())
                if total_nom > 1e-12:
                    for eid, nom_vol in emitter_nom_to_storage[sid].items():
                        share = nom_vol / total_nom
                        outcome.overflow_attributed_emitter[eid] = (
                            outcome.overflow_attributed_emitter.get(eid, 0.0)
                            + overflow * share)

            planned_rates[sid] = desired_q

        step_stored = 0.0
        step_pressure_violations = 0

        for sid in storage_ids_sorted:
            site = self.storage_sites[sid]
            other_rates = [
                planned_rates[other_sid]
                for other_sid in storage_ids_sorted
                if other_sid != sid
            ]
            actual_q, violation = site.inject(
                planned_rates[sid],
                other_site_rates=other_rates if other_rates else None,
                cross_well_scale=cross_well_scale,
                pressure_limit_scale=pressure_limit_scale,
            )
            step_stored += actual_q
            outcome.storage_injected[sid] = actual_q
            outcome.storage_pressure_violation[sid] = violation
            if violation:
                step_pressure_violations += 1
            if outcome.storage_quality_violation.get(sid, False):
                outcome.quality_violations += 1

            s_state = site.get_state()
            outcome.storage_pressure_margin[sid] = min(
                0.3, max(0.0, 1.0 - s_state["pressure_frac"])
            )

        # ---- Step 5: Emitter buffer update ----
        step_captured = 0.0
        step_vented = 0.0

        for eid, emitter in self.emitters.items():
            total_sent = outcome.emitter_sent.get(eid, 0.0)
            captured, vented, direct_vent = emitter.produce_and_buffer(total_sent)
            step_captured += captured
            step_vented += vented

            capture_multiplier = 1.0 + emitter.purification_cost_factor * emitter.purification_effort
            energy_multiplier = 1.0 + emitter.purification_energy_factor * emitter.purification_effort
            capture_cost = captured * emitter.capture_cost_per_t * capture_multiplier
            capture_energy = captured * emitter.capture_energy_mwh_per_t * energy_multiplier
            outcome.emitter_captured[eid] = captured
            outcome.emitter_vented[eid] = vented
            outcome.emitter_direct_vent[eid] = direct_vent
            outcome.emitter_capture_cost[eid] = capture_cost
            outcome.emitter_capture_energy[eid] = capture_energy
            outcome.emitter_buffer_frac[eid] = (
                emitter.buffer_level / max(emitter.buffer_capacity, 1e-9)
            )

        # Add terminal buffer venting to total
        for mode, vented in outcome.terminal_vented.items():
            step_vented += vented

        # Add storage overflow (delivered but not injectable) to venting
        step_vented += storage_overflow_vented

        # ---- Step 6: Transport utilization & end-of-month bookkeeping ----
        if self.pipeline is not None:
            outcome.transport_utilization["pipeline"] = (
                self.pipeline.month_accepted / max(self.pipeline.capacity, 1e-9)
            )
            self.pipeline.end_month()

        if self.ships is not None:
            ship_state = self.ships.get_state()
            outcome.transport_utilization["ship"] = ship_state["utilization"]
            self.ships.end_month()

        if self.rail is not None:
            rail_state = self.rail.get_state()
            outcome.transport_utilization["rail"] = rail_state["utilization"]
            self.rail.end_month()

        # Compute rejected per mode
        for mode_name, noms in mode_nominations.items():
            total_nominated = sum(v for _, _, v in noms)
            total_accepted = outcome.transport_accepted.get(mode_name, 0.0)
            outcome.transport_rejected[mode_name] = max(
                0.0, total_nominated - total_accepted
            )
            outcome.transport_rejected[mode_name] += quality_rejected_by_mode.get(mode_name, 0.0)

        # ---- Aggregates ----
        outcome.total_stored = step_stored
        outcome.total_vented = step_vented
        outcome.total_captured = step_captured
        outcome.transport_cost = total_transport_cost
        outcome.pressure_violations = step_pressure_violations
        outcome.total_capture_cost = sum(outcome.emitter_capture_cost.values())
        outcome.total_energy_use = sum(outcome.emitter_capture_energy.values())

        return outcome

    def _settle_with_pricing(
        self,
        mode_nominations: Dict[str, List[Tuple[int, int, float]]],
        mode_capacities: Dict[str, float],
        posted_prices: Dict[str, float],
        emitter_bids: Dict[int, float],
        outcome: PhysicalOutcome,
    ) -> Dict[str, List[Tuple[int, int, float]]]:
        """Allocate transport capacity by price-priority ordering.

        For each transport mode:
        1. Get posted_price from transport agent
        2. Collect all emitter nominations with their bids
        3. Filter: only emitters willing to pay >= posted_price
        4. Sort remaining by bid (highest first)
        5. Allocate capacity top-down until exhausted
        6. Emitters pay their bid (first-price auction)
        7. Revenue goes to transport agent
        8. Rejected emitters keep CO2 in buffer

        Returns:
            settled_flows: mode_name -> [(eid, sid, actual_volume)]
        """
        settled_flows: Dict[str, List[Tuple[int, int, float]]] = {}

        for mode_name, noms in mode_nominations.items():
            capacity = mode_capacities.get(mode_name, 0.0)
            posted = posted_prices.get(mode_name, 0.0)

            outcome.transport_posted_prices[mode_name] = posted

            # Build bid list: (eid, sid, volume, bid_price)
            bid_list: List[Tuple[int, int, float, float]] = []
            for eid, sid, vol in noms:
                bid = emitter_bids.get(eid, 0.0)
                outcome.emitter_bids[eid] = bid
                bid_list.append((eid, sid, vol, bid))

            # Filter: only emitters willing to pay >= posted_price
            qualified = [(eid, sid, vol, bid) for eid, sid, vol, bid in bid_list
                         if bid >= posted]
            rejected = [(eid, sid, vol, bid) for eid, sid, vol, bid in bid_list
                        if bid < posted]

            # Sort qualified by bid (highest first) for priority
            qualified.sort(key=lambda x: x[3], reverse=True)

            # Allocate capacity top-down
            remaining_cap = capacity
            settled: List[Tuple[int, int, float]] = []
            mode_revenue = 0.0
            total_nominated = sum(v for _, _, v in noms)

            for eid, sid, vol, bid in qualified:
                if remaining_cap <= 0:
                    # Treat rest as rejected
                    break
                actual = min(vol, remaining_cap)
                settled.append((eid, sid, actual))
                remaining_cap -= actual
                # First-price: emitter pays their bid
                cost = actual * bid
                mode_revenue += cost
                outcome.emitter_transport_cost[eid] = (
                    outcome.emitter_transport_cost.get(eid, 0.0) + cost
                )

            # Rejected nominations get zero
            for eid, sid, vol, bid in rejected:
                settled.append((eid, sid, 0.0))

            settled_flows[mode_name] = settled
            outcome.transport_revenue[mode_name] = mode_revenue

            # Compute scarcity ratio for observations
            total_qualified = sum(v for _, _, v, _ in qualified)
            if total_nominated <= 0:
                scarcity = 1.0
            elif total_qualified <= capacity:
                scarcity = 1.0
            else:
                scarcity = capacity / total_qualified
            outcome.scarcity_ratios[mode_name] = scarcity

        return settled_flows

    def _pick_storage_for_mode(self, mode_name: str) -> int:
        """Pick the most common storage target for a given transport mode."""
        mode_idx = self._mode_index.get(mode_name, -1)
        storage_ids = self._transport_to_storage.get(mode_idx, [])
        if storage_ids:
            return storage_ids[0]
        # Fallback: first storage site
        if self.storage_sites:
            return min(self.storage_sites.keys())
        return 0

    def get_state(self) -> Dict[str, Any]:
        """Full physical state for observations."""
        state: Dict[str, Any] = {
            "emitters": {eid: e.get_state() for eid, e in self.emitters.items()},
            "storage_sites": {sid: s.get_state() for sid, s in self.storage_sites.items()},
            "terminal_buffers": {
                bid: b.get_state() for bid, b in self.terminal_buffers.items()
            },
        }

        # Transport state: combine into a unified dict keyed by mode name
        transports: Dict[str, Any] = {}
        if self.pipeline is not None:
            transports["pipeline"] = self.pipeline.get_state()
        if self.ships is not None:
            transports["ship"] = self.ships.get_state()
        if self.rail is not None:
            transports["rail"] = self.rail.get_state()
        state["transports"] = transports

        return state

    def get_transport_state(self, mode_name: str) -> Dict[str, Any]:
        """Get state for a specific transport mode."""
        if mode_name == "pipeline" and self.pipeline is not None:
            return self.pipeline.get_state()
        elif mode_name == "ship" and self.ships is not None:
            return self.ships.get_state()
        elif mode_name == "rail" and self.rail is not None:
            return self.rail.get_state()
        return {
            "available_capacity": 0.0,
            "capacity": 1e-9,
            "utilization": 0.0,
            "in_transit_total": 0.0,
            "is_disrupted": 0.0,
            "disruption_remaining_time": 0.0,
        }

    def apply_disruption(self, target_type: str, target_id: int,
                         severity: float, duration: float) -> None:
        """Apply a disruption to a specific component."""
        if target_type == "emitter":
            if target_id in self.emitters:
                self.emitters[target_id].apply_disruption(severity, duration)
        elif target_type == "transport":
            mode_name = self._idx_to_mode.get(target_id, "")
            if mode_name == "pipeline" and self.pipeline is not None:
                self.pipeline.apply_disruption(severity, duration)
            elif mode_name == "ship" and self.ships is not None:
                self.ships.apply_disruption(severity, duration)
            elif mode_name == "rail" and self.rail is not None:
                self.rail.apply_disruption(severity, duration)
        elif target_type == "storage":
            if target_id in self.storage_sites:
                self.storage_sites[target_id].apply_disruption(severity, duration)

    def check_pressure_triggered(self) -> List[Tuple[int, float, float]]:
        """Check for pressure-triggered storage disruptions.

        Returns list of (storage_id, severity, duration) for any site over limit.
        """
        results = []
        for sid, site in self.storage_sites.items():
            if site.current_pressure > site.pressure_limit and not site.is_disrupted:
                results.append((sid, 0.5, 2.0))
        return results

    # Connectivity helpers (matching CCUSNetwork interface)
    def get_routes_for_emitter(self, emitter_id: int) -> List[Tuple[int, int]]:
        return self.connectivity.get(emitter_id, [])

    def get_connected_transport_ids(self, emitter_id: int) -> List[int]:
        return sorted(set(t for t, _ in self.connectivity.get(emitter_id, [])))

    def get_connected_storage_ids(self, emitter_id: int) -> List[int]:
        return sorted(set(s for _, s in self.connectivity.get(emitter_id, [])))

    def get_storage_ids_for_transport(self, transport_id: int) -> List[int]:
        return self._transport_to_storage.get(transport_id, [])

    def get_mode_name(self, transport_idx: int) -> str:
        return self._idx_to_mode.get(transport_idx, "pipeline")

    def get_mode_index(self, mode_name: str) -> int:
        return self._mode_index.get(mode_name, 0)
