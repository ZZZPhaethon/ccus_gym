"""Default configurations for the CCUS-Gym environment.

Provides:
- MINIMAL_NETWORK_CONFIG: 2 emitters, pipeline + ship, 1 storage site
- FULL_NETWORK_CONFIG: 6 emitters, pipeline + ship + rail, 2 storage sites
- SCENARIO_CONFIGS: disruption + mechanism params for each scenario family
- CALIBRATED_HUB_CONFIG: Northern Lights / Humber-inspired realistic parameters

All flow rates are in MtCO2/month (annual values divided by 12).
Pressures in bar.  Costs in $/tCO2.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from ccus_gym.core.quality import DEFAULT_STORAGE_QUALITY_LIMITS, method_defaults

# ===================================================================
# Unit conversion helpers
# ===================================================================

def _annual_to_monthly(rate: float) -> float:
    """Convert MtCO2/year to MtCO2/month."""
    return rate / 12.0


def _default_emitter_meta(
    capture_method: str = "post_combustion",
    *,
    sector: str = "industrial",
    name: str = "",
) -> Dict[str, Any]:
    defaults = method_defaults(capture_method)
    return {
        "name": name,
        "sector": sector,
        "capture_method": capture_method,
        "base_purity": defaults["base_purity"],
        "composition": deepcopy(defaults["composition"]),
        "capture_cost_per_t": defaults["base_cost_per_t"],
        "capture_energy_mwh_per_t": defaults["base_energy_mwh_per_t"],
        "purification_cost_factor": 0.35,
        "purification_energy_factor": 0.20,
    }


def _default_storage_quality() -> Dict[str, Any]:
    return {
        "min_purity": DEFAULT_STORAGE_QUALITY_LIMITS["min_purity"],
        "max_impurities": deepcopy(DEFAULT_STORAGE_QUALITY_LIMITS["max_impurities"]),
    }


# ===================================================================
# MINIMAL NETWORK CONFIG
# ===================================================================
# 2 emitters, 1 pipeline, 1 ship, 1 storage site
# Inspired by a small Humber-cluster to Northern Lights setup

MINIMAL_NETWORK_CONFIG: Dict[str, Any] = {
    "network": {
        "num_emitters": 2,
        "transport_modes": ["pipeline", "ship"],
        "num_storage_sites": 1,

        "emitter_defaults": {
            "max_capture_rate": _annual_to_monthly(1.5),   # 1.5 MtCO2/yr
            "buffer_capacity": 0.15,                        # ~2 months buffer
            "production_rate": _annual_to_monthly(1.8),     # slightly above capture
            **_default_emitter_meta("post_combustion", sector="power"),
        },
        "emitter_params": {
            0: {
                "max_capture_rate": _annual_to_monthly(1.0),  # smaller plant
                "buffer_capacity": 0.10,
                "production_rate": _annual_to_monthly(1.2),
                **_default_emitter_meta("post_combustion", sector="power", name="emitter_0"),
            },
            1: {
                "max_capture_rate": _annual_to_monthly(2.0),  # larger plant
                "buffer_capacity": 0.20,
                "production_rate": _annual_to_monthly(2.4),
                **_default_emitter_meta("oxy_fuel", sector="cement", name="emitter_1"),
            },
        },

        "transport_params": {
            "pipeline": {
                "capacity": _annual_to_monthly(5.0),  # 5 MtCO2/yr
                "base_cost": 8.0,                      # $/tCO2 (onshore/short offshore)
                "latency": 0.0,                        # continuous flow, negligible delay
            },
            "ship": {
                "capacity": _annual_to_monthly(1.5),  # 1.5 MtCO2/yr (~2 ships)
                "base_cost": 18.0,                     # $/tCO2 (liquefaction + shipping)
                "latency": 1.0,                        # 1-month round trip
            },
        },

        "storage_defaults": {
            "max_injection_rate": _annual_to_monthly(3.0),  # 3 MtCO2/yr
            "pressure_limit": 300.0,                         # bar (saline aquifer)
            "initial_pressure": 100.0,                       # bar (hydrostatic)
            "k_injection": 40.0,                             # bar per (MtCO2/month)
            "k_dissipation": 0.05,                           # 5% pressure decay per month
            "cross_well_coeff": 5.0,                         # bar per (MtCO2/month) other well
            "injectivity_decline_rate": 0.15,                # 15% decline at full capacity
            "cumulative_capacity": 25.0,                     # 25 Mt total
            **_default_storage_quality(),
        },
        "storage_params": {},

        # Connectivity: emitter_id -> [(transport_id, storage_id), ...]
        # Transport IDs: 0=pipeline, 1=ship
        # Storage IDs: 0
        "connectivity": {
            0: [[0, 0], [1, 0]],  # emitter 0 -> pipeline or ship -> storage 0
            1: [[0, 0], [1, 0]],  # emitter 1 -> pipeline or ship -> storage 0
        },
    },

    # Physical layer configuration (ship fleet, rail, terminal buffers)
    "physical": {
        "ship": {
            "fleet": [
                {"type": "small", "count": 2},   # 2 small ships (~1.5 MtCO2/yr)
            ],
            "distance_km": 800.0,
            "loading_days": 4.0,  # 4 days allows ship to fill from daily CO2 inflow
            "unloading_days": 2.0,
        },
        "terminal_buffer_ship": {
            "max_capacity": 0.030,  # 30,000 tonnes (~1.4x medium ship capacity)
        },
    },

    "disruption": {
        "scenario_family": "T",
        "severity": 0.5,
        "cross_correlation": 0.0,
    },

    "mechanism": {
        "alpha": 0.5,
        "beta": 0.3,
        "gamma": 1.0,
    },

    "carbon_tax": 0.0,  # $/tCO2 (0 = pricing disabled, e.g. 80 to enable)
    "electricity_price": 65.0,  # $/MWh
    "capture_subsidy": 0.0,  # $/tCO2 captured
    "storage_credit": 0.0,  # $/tCO2 stored
    "offspec_penalty": 6.0,  # $/tCO2-equivalent penalty
    "quality_weight": 3.0,
    "energy_weight": 0.03,
    "extreme_scenarios": [],

    # Institutional mechanism parameters
    "congestion_threshold": 0.8,   # pipeline utilization above this triggers surcharge
    "k_congestion": 5.0,           # congestion surcharge multiplier
    "injection_obligation_coverage": 0.85,  # target = max_injection_rate * coverage
    "storage_obligation_penalty": 1.0,      # penalty per unit shortfall

    "episode_length": 120,
    "seed": 42,
}


# ===================================================================
# FULL NETWORK CONFIG
# ===================================================================
# 6 emitters, pipeline + ship + rail, 2 storage sites
# Represents a large industrial cluster (Humber + Teesside scale)
# connected to two offshore storage complexes

FULL_NETWORK_CONFIG: Dict[str, Any] = {
    "network": {
        "num_emitters": 6,
        "transport_modes": ["pipeline", "ship", "rail"],
        "num_storage_sites": 2,

        "emitter_defaults": {
            "max_capture_rate": _annual_to_monthly(1.0),
            "buffer_capacity": 0.10,
            "production_rate": _annual_to_monthly(1.2),
            **_default_emitter_meta("post_combustion", sector="industrial"),
        },
        "emitter_params": {
            # Steel plant — large
            0: {
                "max_capture_rate": _annual_to_monthly(2.0),
                "buffer_capacity": 0.20,
                "production_rate": _annual_to_monthly(2.5),
                **_default_emitter_meta("post_combustion", sector="steel", name="steel_plant"),
            },
            # Power station — large, flexible
            1: {
                "max_capture_rate": _annual_to_monthly(1.8),
                "buffer_capacity": 0.25,
                "production_rate": _annual_to_monthly(2.0),
                **_default_emitter_meta("post_combustion", sector="power", name="power_station"),
            },
            # Cement plant — medium
            2: {
                "max_capture_rate": _annual_to_monthly(1.0),
                "buffer_capacity": 0.10,
                "production_rate": _annual_to_monthly(1.2),
                **_default_emitter_meta("oxy_fuel", sector="cement", name="cement_plant"),
            },
            # Refinery — medium
            3: {
                "max_capture_rate": _annual_to_monthly(1.2),
                "buffer_capacity": 0.12,
                "production_rate": _annual_to_monthly(1.5),
                **_default_emitter_meta("pre_combustion", sector="refinery", name="refinery"),
            },
            # Chemical plant — small
            4: {
                "max_capture_rate": _annual_to_monthly(0.5),
                "buffer_capacity": 0.06,
                "production_rate": _annual_to_monthly(0.6),
                **_default_emitter_meta("post_combustion", sector="chemicals", name="chemical_plant"),
            },
            # Hydrogen plant — small
            5: {
                "max_capture_rate": _annual_to_monthly(0.8),
                "buffer_capacity": 0.08,
                "production_rate": _annual_to_monthly(1.0),
                **_default_emitter_meta("pre_combustion", sector="hydrogen", name="hydrogen_plant"),
            },
        },

        "transport_params": {
            "pipeline": {
                "capacity": _annual_to_monthly(5.0),   # 5 MtCO2/yr trunk pipeline
                "base_cost": 8.0,
                "latency": 0.0,
            },
            "ship": {
                "capacity": _annual_to_monthly(2.0),   # 2 MtCO2/yr fleet
                "base_cost": 18.0,
                "latency": 1.0,
            },
            "rail": {
                "capacity": _annual_to_monthly(0.8),   # 0.8 MtCO2/yr
                "base_cost": 25.0,                      # most expensive
                "latency": 0.5,                         # ~2 weeks
            },
        },

        "storage_defaults": {
            "max_injection_rate": _annual_to_monthly(3.0),
            "pressure_limit": 300.0,
            "initial_pressure": 100.0,
            "k_injection": 40.0,
            "k_dissipation": 0.05,
            "cross_well_coeff": 5.0,
            "injectivity_decline_rate": 0.15,
            "cumulative_capacity": 25.0,
            **_default_storage_quality(),
        },
        "storage_params": {
            # Primary storage — larger, well-characterized (Endurance-like)
            0: {
                "max_injection_rate": _annual_to_monthly(4.0),
                "pressure_limit": 320.0,
                "initial_pressure": 110.0,
                "k_injection": 35.0,
                "k_dissipation": 0.06,
                "cross_well_coeff": 4.0,
                "injectivity_decline_rate": 0.10,
                "cumulative_capacity": 30.0,
            },
            # Secondary storage — smaller (Aurora-like)
            1: {
                "max_injection_rate": _annual_to_monthly(2.5),
                "pressure_limit": 280.0,
                "initial_pressure": 95.0,
                "k_injection": 45.0,
                "k_dissipation": 0.04,
                "cross_well_coeff": 6.0,
                "injectivity_decline_rate": 0.20,
                "cumulative_capacity": 20.0,
            },
        },

        # Connectivity: not all emitters connect to all transport modes
        # Transport IDs: 0=pipeline, 1=ship, 2=rail
        # Storage IDs: 0 (primary), 1 (secondary)
        "connectivity": {
            # Emitters 0-3: on the pipeline trunk, can also use ship
            0: [[0, 0], [0, 1], [1, 0], [1, 1]],
            1: [[0, 0], [0, 1], [1, 0], [1, 1]],
            2: [[0, 0], [0, 1], [1, 0]],
            3: [[0, 0], [1, 0], [1, 1]],
            # Emitters 4-5: remote, use ship and rail only
            4: [[1, 0], [1, 1], [2, 0], [2, 1]],
            5: [[1, 0], [1, 1], [2, 0], [2, 1]],
        },
    },

    # Physical layer configuration
    "physical": {
        "ship": {
            "fleet": [
                {"type": "medium", "count": 3},   # 3 medium ships (~2 MtCO2/yr)
            ],
            "distance_km": 800.0,
            "loading_days": 4.0,  # 4 days allows ship to fill from daily CO2 inflow
            "unloading_days": 2.0,
        },
        "terminal_buffer_ship": {
            "max_capacity": 0.030,  # 30,000 tonnes (~1.4x medium ship)
        },
        "rail": {
            "car_capacity": 0.00008,    # 80 tonnes per car
            "cars_per_train": 80,
            "num_trains": 4,
            "loading_time_days": 1,
            "unloading_time_days": 1,
            "transit_time_days": 3,
            "distance_km": 200,
        },
        "terminal_buffer_rail": {
            "max_capacity": 0.005,  # 5,000 tonnes
        },
    },

    "disruption": {
        "scenario_family": "TSG",
        "severity": 0.5,
        "cross_correlation": 0.1,
    },

    "mechanism": {
        "alpha": 0.5,   # mixed: some disruptions predictable, some not
        "beta": 0.5,    # moderate cross-well coupling
        "gamma": 0.6,   # moderate response-time criticality
    },

    "carbon_tax": 0.0,  # $/tCO2 (0 = pricing disabled)
    "electricity_price": 70.0,
    "capture_subsidy": 0.0,
    "storage_credit": 0.0,
    "offspec_penalty": 7.0,
    "quality_weight": 3.5,
    "energy_weight": 0.03,
    "extreme_scenarios": [],

    # Institutional mechanism parameters
    "congestion_threshold": 0.8,
    "k_congestion": 5.0,
    "injection_obligation_coverage": 0.85,
    "storage_obligation_penalty": 1.0,

    "episode_length": 120,
    "seed": 42,
}


# ===================================================================
# SCENARIO FAMILY CONFIGS
# ===================================================================
# Disruption + mechanism parameters for each of the 7 scenario families.
# These represent "canonical" parameter sets at medium severity (kappa=0.5).

SCENARIO_CONFIGS: Dict[str, Dict[str, Any]] = {
    # --- Structural mechanism axes ---
    # alpha: disruption predictability (low = predictable/forecastable → MPC wins;
    #         high = unpredictable/no warning → heuristic/MARL wins)
    # beta:  storage coupling tightness (low = independent sites → heuristic OK;
    #         high = shared aquifer, strong cross-well → centralized coordination needed)
    # gamma: response-time criticality (low = slow disruptions → MPC has time;
    #         high = fast intense disruptions → decentralized reactive wins)

    "T": {
        "disruption": {
            "scenario_family": "T",
            "severity": 0.5,
            "cross_correlation": 0.0,
        },
        "mechanism": {
            "alpha": 0.7,   # transport disruptions hard to predict → favours reactive
            "beta": 0.2,    # single storage, low coupling
            "gamma": 0.8,   # moderate: pipeline failures develop slowly
        },
    },
    "S": {
        "disruption": {
            "scenario_family": "S",
            "severity": 0.5,
            "cross_correlation": 0.0,
        },
        "mechanism": {
            "alpha": 0.4,   # supply disruptions (maintenance) often scheduled → forecastable
            "beta": 0.15,   # low coupling — each emitter independent
            "gamma": 0.3,   # slow onset → MPC has time to coordinate
        },
    },
    "G": {
        "disruption": {
            "scenario_family": "G",
            "severity": 0.5,
            "cross_correlation": 0.0,
        },
        "mechanism": {
            "alpha": 0.2,   # pressure is globally measurable, forecastable → MPC regime
            "beta": 0.8,    # shared aquifer = strong coupling → needs coordination
            "gamma": 0.4,   # pressure builds predictably → centralized can plan
        },
    },
    "TS": {
        "disruption": {
            "scenario_family": "TS",
            "severity": 0.5,
            "cross_correlation": 0.1,
        },
        "mechanism": {
            "alpha": 0.6,   # mixed: some predictable, some not → hybrid territory
            "beta": 0.3,    # moderate coupling
            "gamma": 0.6,   # moderate response time
        },
    },
    "TG": {
        "disruption": {
            "scenario_family": "TG",
            "severity": 0.5,
            "cross_correlation": 0.1,
        },
        "mechanism": {
            "alpha": 0.5,   # mixed predictability
            "beta": 0.6,    # storage coupling present → centralized helps
            "gamma": 0.7,   # moderate-fast response needed
        },
    },
    "SG": {
        "disruption": {
            "scenario_family": "SG",
            "severity": 0.5,
            "cross_correlation": 0.1,
        },
        "mechanism": {
            "alpha": 0.3,   # supply+storage both forecastable → strong MPC regime
            "beta": 0.7,    # strong coupling from storage
            "gamma": 0.35,  # slow → MPC planning horizon effective
        },
    },
    "TSG": {
        "disruption": {
            "scenario_family": "TSG",
            "severity": 0.5,
            "cross_correlation": 0.2,
        },
        "mechanism": {
            "alpha": 0.5,   # mixed: transport unpredictable, others forecastable
            "beta": 0.5,    # moderate coupling
            "gamma": 0.6,   # moderate → hybrid territory
        },
    },
}


# ===================================================================
# CALIBRATED HUB CONFIG
# ===================================================================
# Northern Lights / Humber-inspired parameters with literature sources.
#
# Sources:
#   - Northern Lights Phase 1: 1.5 MtCO2/yr initial, 5 MtCO2/yr Phase 2
#     (Equinor, 2020, "Northern Lights PCI project")
#   - Humber cluster: ~17 MtCO2/yr total capture potential
#     (Element Energy, 2020, "Humber Industrial Decarbonisation Deployment")
#   - Endurance aquifer: ~300 bar fracture gradient, ~100 bar hydrostatic
#     (BP, 2021, "Net Zero Teesside Storage Development Plan")
#   - Ship transport cost: 15-25 $/tCO2 for 700-1500 km
#     (Knoope et al., 2015, IJGHGC)
#   - Pipeline cost: 5-15 $/tCO2 depending on distance and capacity
#     (IEAGHG, 2014, "CO2 Pipeline Infrastructure")
#   - Injectivity decline: 10-30% over field lifetime in saline aquifers
#     (Birkholzer et al., 2015, IJGHGC)
#   - Pressure response coefficients: calibrated to match CMG/ECLIPSE
#     reservoir simulation results for Sleipner-class aquifer
#     (Chadwick et al., 2012, Energy Procedia)

CALIBRATED_HUB_CONFIG: Dict[str, Any] = {
    "network": {
        "num_emitters": 6,
        "transport_modes": ["pipeline", "ship", "rail"],
        "num_storage_sites": 2,

        "emitter_defaults": {
            "max_capture_rate": _annual_to_monthly(1.0),
            "buffer_capacity": 0.10,
            "production_rate": _annual_to_monthly(1.2),
            **_default_emitter_meta("post_combustion", sector="industrial"),
        },
        "emitter_params": {
            # Drax BECCS — 4 MtCO2/yr (post-combustion on biomass)
            0: {
                "max_capture_rate": _annual_to_monthly(4.0),
                "buffer_capacity": 0.40,
                "production_rate": _annual_to_monthly(4.5),
                **_default_emitter_meta("post_combustion", sector="beccs", name="Drax_BECCS"),
            },
            # Keadby 3 gas CCS — 1.5 MtCO2/yr
            1: {
                "max_capture_rate": _annual_to_monthly(1.5),
                "buffer_capacity": 0.15,
                "production_rate": _annual_to_monthly(1.8),
                **_default_emitter_meta("post_combustion", sector="power", name="Keadby_3"),
            },
            # British Steel Scunthorpe — 2.5 MtCO2/yr
            2: {
                "max_capture_rate": _annual_to_monthly(2.5),
                "buffer_capacity": 0.25,
                "production_rate": _annual_to_monthly(3.0),
                **_default_emitter_meta("post_combustion", sector="steel", name="British_Steel"),
            },
            # Phillips 66 Humber Refinery — 1.0 MtCO2/yr
            3: {
                "max_capture_rate": _annual_to_monthly(1.0),
                "buffer_capacity": 0.10,
                "production_rate": _annual_to_monthly(1.2),
                **_default_emitter_meta("pre_combustion", sector="refinery", name="Phillips_66"),
            },
            # Saltend Chemicals — 0.6 MtCO2/yr
            4: {
                "max_capture_rate": _annual_to_monthly(0.6),
                "buffer_capacity": 0.06,
                "production_rate": _annual_to_monthly(0.7),
                **_default_emitter_meta("oxy_fuel", sector="chemicals", name="Saltend_Chemicals"),
            },
            # H2H Saltend Hydrogen — 0.9 MtCO2/yr
            5: {
                "max_capture_rate": _annual_to_monthly(0.9),
                "buffer_capacity": 0.09,
                "production_rate": _annual_to_monthly(1.0),
                **_default_emitter_meta("pre_combustion", sector="hydrogen", name="H2H_Saltend"),
            },
        },

        "transport_params": {
            # Trunk pipeline: Humber to Endurance (offshore, ~150 km)
            "pipeline": {
                "capacity": _annual_to_monthly(10.0),  # 10 MtCO2/yr Phase 2
                "base_cost": 10.0,                      # $/tCO2
                "latency": 0.0,
            },
            # Ship: liquefied CO2 to Northern Lights (Norway, ~800 km)
            "ship": {
                "capacity": _annual_to_monthly(3.0),   # fleet of 3-4 vessels
                "base_cost": 20.0,                      # $/tCO2
                "latency": 1.0,                         # ~1 month round trip
            },
            # Rail: for inland emitters without pipeline access
            "rail": {
                "capacity": _annual_to_monthly(0.5),
                "base_cost": 30.0,
                "latency": 0.5,
            },
        },

        "storage_defaults": {
            "max_injection_rate": _annual_to_monthly(5.0),
            "pressure_limit": 300.0,
            "initial_pressure": 100.0,
            "k_injection": 40.0,
            "k_dissipation": 0.05,
            "cross_well_coeff": 5.0,
            "injectivity_decline_rate": 0.15,
            "cumulative_capacity": 25.0,
            **_default_storage_quality(),
        },
        "storage_params": {
            # Endurance (Southern North Sea saline aquifer)
            # Estimated 450 Mt total capacity, 300 bar fracture gradient
            0: {
                "max_injection_rate": _annual_to_monthly(5.0),
                "pressure_limit": 310.0,
                "initial_pressure": 105.0,
                "k_injection": 38.0,
                "k_dissipation": 0.06,
                "cross_well_coeff": 4.0,
                "injectivity_decline_rate": 0.08,
                "cumulative_capacity": 40.0,
            },
            # Northern Lights (Johansen formation, Norwegian North Sea)
            # 100 Mt initial capacity, expandable
            1: {
                "max_injection_rate": _annual_to_monthly(3.0),
                "pressure_limit": 290.0,
                "initial_pressure": 98.0,
                "k_injection": 42.0,
                "k_dissipation": 0.04,
                "cross_well_coeff": 6.0,
                "injectivity_decline_rate": 0.12,
                "cumulative_capacity": 25.0,
            },
        },

        "connectivity": {
            # Drax: pipeline to both storages, ship to Northern Lights
            0: [[0, 0], [0, 1], [1, 1]],
            # Keadby: pipeline to Endurance, ship to both
            1: [[0, 0], [1, 0], [1, 1]],
            # British Steel: pipeline to Endurance
            2: [[0, 0], [1, 0]],
            # Phillips 66: pipeline to Endurance, ship as backup
            3: [[0, 0], [1, 0], [1, 1]],
            # Saltend Chemicals: pipeline + rail
            4: [[0, 0], [2, 0]],
            # H2H Saltend: pipeline + rail
            5: [[0, 0], [2, 0]],
        },
    },

    # Physical layer configuration (Northern Lights / Humber calibrated)
    "physical": {
        "ship": {
            "fleet": [
                {"type": "medium", "count": 2},   # 2 medium (22kt) ships
                {"type": "large", "count": 1},     # 1 large (50kt) ship
            ],
            "distance_km": 800.0,    # Humber to Northern Lights
            "loading_days": 4.0,  # 4 days allows ship to fill from daily CO2 inflow
            "unloading_days": 2.0,
        },
        "terminal_buffer_ship": {
            "max_capacity": 0.030,  # 30,000 tonnes (~1.4x medium ship) (Northern Lights Phase 1)
        },
        "rail": {
            "car_capacity": 0.00008,    # 80 tonnes per car
            "cars_per_train": 80,       # standard UK unit train
            "num_trains": 3,
            "loading_time_days": 1,
            "unloading_time_days": 1,
            "transit_time_days": 2,     # short distance (~100km)
            "distance_km": 100,
        },
        "terminal_buffer_rail": {
            "max_capacity": 0.003,  # 3,000 tonnes (smaller rail terminal)
        },
    },

    "disruption": {
        "scenario_family": "TSG",
        "severity": 0.5,
        "cross_correlation": 0.15,
    },

    "mechanism": {
        "alpha": 0.4,   # well-instrumented hub, good forecasting
        "beta": 0.6,    # shared Endurance/NL aquifer coupling
        "gamma": 0.5,   # moderate response-time requirement
    },

    "carbon_tax": 0.0,  # $/tCO2 (0 = pricing disabled)
    "electricity_price": 85.0,
    "capture_subsidy": 0.0,
    "storage_credit": 0.0,
    "offspec_penalty": 8.0,
    "quality_weight": 4.0,
    "energy_weight": 0.03,
    "extreme_scenarios": [
        {
            "name": "power_price_spike",
            "start_timestep": 24,
            "duration": 6,
            "electricity_price_multiplier": 1.4,
            "carbon_tax_multiplier": 1.0,
        },
        {
            "name": "policy_tightening",
            "start_timestep": 60,
            "duration": 12,
            "electricity_price_multiplier": 1.0,
            "carbon_tax_multiplier": 1.35,
        },
    ],

    # Institutional mechanism parameters
    "congestion_threshold": 0.8,
    "k_congestion": 5.0,
    "injection_obligation_coverage": 0.85,
    "storage_obligation_penalty": 1.0,

    "episode_length": 120,
    "seed": 42,
}


# ===================================================================
# Utility: build a config for a specific scenario family + severity
# ===================================================================

def make_config(
    base: str = "minimal",
    scenario_family: str = "T",
    severity: float = 0.5,
    cross_correlation: float = 0.0,
    seed: int = 42,
    carbon_tax: float = 0.0,
    electricity_price: float | None = None,
) -> Dict[str, Any]:
    """Create a complete config by combining a base network with a scenario.

    Args:
        base: "minimal", "full", or "calibrated".
        scenario_family: One of T, S, G, TS, TG, SG, TSG.
        severity: kappa in [0, 1].
        cross_correlation: rho_cross in [0, 1].
        seed: Random seed.
        carbon_tax: $/tCO2 carbon tax (0 = pricing disabled).
        electricity_price: Optional override for electricity price in $/MWh.

    Returns:
        Complete config dict ready for CCUSEnv.
    """
    if base == "minimal":
        config = deepcopy(MINIMAL_NETWORK_CONFIG)
    elif base == "full":
        config = deepcopy(FULL_NETWORK_CONFIG)
    elif base == "calibrated":
        config = deepcopy(CALIBRATED_HUB_CONFIG)
    else:
        raise ValueError(f"Unknown base config: {base}")

    # Apply scenario
    sc = SCENARIO_CONFIGS.get(scenario_family.upper())
    if sc is None:
        raise ValueError(f"Unknown scenario family: {scenario_family}")

    config["disruption"] = deepcopy(sc["disruption"])
    config["disruption"]["severity"] = severity
    config["disruption"]["cross_correlation"] = cross_correlation
    config["mechanism"] = deepcopy(sc["mechanism"])
    config["seed"] = seed
    config["carbon_tax"] = carbon_tax
    if electricity_price is not None:
        config["electricity_price"] = electricity_price

    return config
