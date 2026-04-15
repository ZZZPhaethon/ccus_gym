"""YAML case definition loader for the CCUS-Gym environment.

Reads a standardized YAML case file (see cases/*.yaml) and converts it into
the config dict format expected by CCUSEnv / PhysicalLayer.

Key conversions:
    - MtCO2/year -> MtCO2/month  (divide by 12)
    - tonnes -> MtCO2             (divide by 1e6)
    - proxy_model paths resolved relative to project root

Usage:
    from ccus_gym.sim.case_loader import load_case
    config = load_case("cases/teesside_uk.yaml")
    env = CCUSEnv(config)

    # Or use the convenience classmethod:
    env = CCUSEnv.from_case("cases/teesside_uk.yaml")
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ccus_gym.core.quality import DEFAULT_STORAGE_QUALITY_LIMITS, method_defaults

logger = logging.getLogger(__name__)


# ===================================================================
# Unit conversion helpers
# ===================================================================

def _mtpa_to_monthly(rate_mtpa: float) -> float:
    """Convert MtCO2/year to MtCO2/month."""
    return rate_mtpa / 12.0


def _tonnes_to_mt(tonnes: float) -> float:
    """Convert tonnes to MtCO2."""
    return tonnes / 1.0e6


# ===================================================================
# Proxy model resolution helpers
# ===================================================================

def _resolve_proxy_path(proxy_rel: Optional[str], project_root: str) -> Optional[str]:
    """Resolve a proxy model path relative to the project root.

    Handles both forward-slash and backslash paths.  Returns None if
    the input is None or empty.
    """
    if not proxy_rel:
        return None

    # Normalise path separators
    proxy_rel = proxy_rel.replace("\\", "/")

    # Also try the alternative directory name used in the project
    # (the YAML uses 'proxy_model/' but the actual dir is 'proxy model/')
    candidates = [
        os.path.join(project_root, proxy_rel),
        os.path.join(project_root, proxy_rel.replace("proxy_model/", "proxy model/")),
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)

    # Return first candidate even if not found (will fail at load time)
    logger.warning(
        "Proxy model file not found at any of %s. "
        "Will attempt to load from %s at runtime.",
        candidates, candidates[0],
    )
    return os.path.normpath(candidates[0])


# ===================================================================
# Storage builder helpers
# ===================================================================

def _build_storage_rom_config(
    storage_entry: dict,
    index: int,
) -> dict:
    """Build storage config dict for analytical ROM mode (no proxy).

    Reads rom_params from the YAML entry and converts to the format
    expected by StoragePhysics / PhysicalLayer.
    """
    rom = storage_entry.get("rom_params", {})
    return {
        "max_injection_rate": _mtpa_to_monthly(
            rom.get("max_injection_rate_mtpa", 3.0)
        ),
        "pressure_limit": rom.get("pressure_limit_bar", 300.0),
        "initial_pressure": rom.get("initial_pressure_bar", 100.0),
        "k_injection": rom.get("k_injection", 40.0),
        "k_dissipation": rom.get("k_dissipation", 0.05),
        "cross_well_coeff": rom.get("cross_well_coeff", 5.0),
        "injectivity_decline_rate": rom.get("injectivity_decline_rate", 0.15),
        "cumulative_capacity": storage_entry.get("capacity_mt", 25.0),
    }


def _build_storage_proxy_params(
    storage_entry: dict,
) -> dict:
    """Extract site_params dict for StorageProxyModel from a YAML entry."""
    return {
        "fracture_pressure_gradient": storage_entry.get(
            "fracture_pressure_gradient", 0.746
        ),
        "transmissibility_multiplier": storage_entry.get(
            "transmissibility_multiplier", 1.0
        ),
        "aquifer_pv_multiplier": storage_entry.get("aquifer_pv_multiplier", 1.0),
        "num_wells": storage_entry.get("num_wells", 1),
        "dome_depth_m": storage_entry.get("dome_depth_m", 1080.0),
        "bottom_hole_depth_m": storage_entry.get("bottom_hole_depth_m", 1450.0),
        "initial_dome_pressure_mpa": storage_entry.get(
            "initial_dome_pressure_mpa", 11.5
        ),
        "initial_bhp_mpa": storage_entry.get("initial_bhp_mpa", 15.87),
        "capacity_mt": storage_entry.get("capacity_mt", 25.0),
        "pore_volume_1e6m3": storage_entry.get("pore_volume_1e6m3", 10.0),
    }


# ===================================================================
# Default ROM fallback parameters for proxy-mode storage
# (used when only proxy is specified, no rom_params)
# ===================================================================

def _proxy_storage_rom_defaults(sp: dict, capacity_mt: float) -> dict:
    """Generate sensible analytical ROM defaults from proxy site params.

    These are only used if the proxy model fails to load and we need to
    fall back to the analytical ROM.
    """
    from ccus_gym.core.storage_proxy import meters_to_feet, psi_to_pa

    dome_depth_m = sp.get("dome_depth_m", 1080.0)
    fpg = sp.get("fracture_pressure_gradient", 0.746)
    init_dome_mpa = sp.get("initial_dome_pressure_mpa", 11.5)

    # Approximate pressure limit from fracture gradient (90% safety margin)
    dome_depth_ft = meters_to_feet(dome_depth_m)
    frac_psi = dome_depth_ft * fpg
    safe_psi = 0.9 * frac_psi
    safe_bar = psi_to_pa(safe_psi) / 1e5  # Pa -> bar

    initial_bar = init_dome_mpa * 10.0  # MPa -> bar

    return {
        "max_injection_rate": _mtpa_to_monthly(3.0),
        "pressure_limit": safe_bar,
        "initial_pressure": initial_bar,
        "k_injection": 40.0,
        "k_dissipation": 0.05,
        "cross_well_coeff": 5.0,
        "injectivity_decline_rate": 0.15,
        "cumulative_capacity": capacity_mt,
    }


# ===================================================================
# Main loader
# ===================================================================

def load_case(case_path: str) -> dict:
    """Load a YAML case file and convert to an env config dict.

    The returned dict is compatible with CCUSEnv(config).

    Args:
        case_path: Path to the YAML case definition file.

    Returns:
        Config dict with keys: network, physical, disruption, mechanism,
        episode_length, seed, and optionally proxy_models.
    """
    case_path = os.path.normpath(case_path)
    project_root = _find_project_root(case_path)

    with open(case_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # ------ Emitters ------
    emitters_raw: List[dict] = raw.get("emitters", [])
    num_emitters = len(emitters_raw)

    emitter_params: Dict[int, dict] = {}
    for i, em in enumerate(emitters_raw):
        capture_method = em.get("capture_method", "post_combustion")
        defaults = method_defaults(capture_method)
        emitter_params[i] = {
            "name": em.get("name", f"emitter_{i}"),
            "sector": em.get("sector", "industrial"),
            "capture_method": capture_method,
            "max_capture_rate": _mtpa_to_monthly(em.get("capture_rate_mtpa", 1.0)),
            "buffer_capacity": _tonnes_to_mt(em.get("buffer_capacity_t", 2000)),
            "production_rate": _mtpa_to_monthly(em.get("production_rate_mtpa", 1.2)),
            "base_purity": float(em.get("purity", defaults["base_purity"])),
            "composition": deepcopy(em.get("composition", defaults["composition"])),
            "capture_cost_per_t": float(
                em.get("capture_cost_per_t", defaults["base_cost_per_t"])
            ),
            "capture_energy_mwh_per_t": float(
                em.get("capture_energy_mwh_per_t", defaults["base_energy_mwh_per_t"])
            ),
            "purification_cost_factor": float(em.get("purification_cost_factor", 0.35)),
            "purification_energy_factor": float(em.get("purification_energy_factor", 0.20)),
        }

    # ------ Transport ------
    transport_raw = raw.get("transport", {})
    transport_modes: List[str] = []
    transport_params: Dict[str, dict] = {}
    physical_cfg: Dict[str, Any] = {}

    # Pipeline
    pipe_raw = transport_raw.get("pipeline")
    if pipe_raw is not None:
        transport_modes.append("pipeline")
        transport_params["pipeline"] = {
            "capacity": _mtpa_to_monthly(pipe_raw.get("capacity_mtpa", 5.0)),
            "base_cost": pipe_raw.get("cost_per_t", 8.0),
            "latency": 0.0,
        }

    # Ships
    ship_raw = transport_raw.get("ships")
    if ship_raw is not None:
        transport_modes.append("ship")
        fleet_raw = ship_raw.get("fleet", [{"type": "medium", "count": 2}])
        distance_km = ship_raw.get("distance_km", 800.0)
        loading_days = ship_raw.get("loading_days", 2.0)
        unloading_days = ship_raw.get("unloading_days", 2.0)

        # Estimate ship capacity in MtCO2/month for the network layer
        from ccus_gym.core.physical import SHIP_TYPES
        total_ship_capacity_mt = 0.0
        for fs in fleet_raw:
            stype = SHIP_TYPES.get(fs.get("type", "medium"), SHIP_TYPES["medium"])
            total_ship_capacity_mt += stype["capacity"] * fs.get("count", 1)

        transport_params["ship"] = {
            "capacity": total_ship_capacity_mt,
            "base_cost": 20.0,
            "latency": 1.0,
        }
        physical_cfg["ship"] = {
            "fleet": fleet_raw,
            "distance_km": distance_km,
            "loading_days": loading_days,
            "unloading_days": unloading_days,
        }

    # Rail
    rail_raw = transport_raw.get("rail")
    if rail_raw is not None:
        transport_modes.append("rail")
        car_cap_t = rail_raw.get("car_capacity_t", 80)
        cars = rail_raw.get("cars_per_train", 80)
        n_trains = rail_raw.get("num_trains", 4)
        train_cap_mt = _tonnes_to_mt(car_cap_t * cars)
        total_rail_cap_mt = train_cap_mt * n_trains

        transport_params["rail"] = {
            "capacity": total_rail_cap_mt,
            "base_cost": 25.0,
            "latency": 0.5,
        }
        physical_cfg["rail"] = {
            "car_capacity": _tonnes_to_mt(car_cap_t),
            "cars_per_train": cars,
            "num_trains": n_trains,
            "loading_time_days": rail_raw.get("loading_time_days", 1),
            "unloading_time_days": rail_raw.get("unloading_time_days", 1),
            "transit_time_days": rail_raw.get("transit_time_days", 3),
            "distance_km": rail_raw.get("distance_km", 200),
        }

    # ------ Terminal Buffers ------
    tb_raw = raw.get("terminal_buffers", {})
    if "ship" in tb_raw:
        physical_cfg["terminal_buffer_ship"] = {
            "max_capacity": _tonnes_to_mt(tb_raw["ship"].get("capacity_t", 8400)),
        }
    if "rail" in tb_raw:
        physical_cfg["terminal_buffer_rail"] = {
            "max_capacity": _tonnes_to_mt(tb_raw["rail"].get("capacity_t", 5000)),
        }

    # ------ Storage ------
    storage_raw: List[dict] = raw.get("storage", [])
    num_storage = len(storage_raw)

    storage_params: Dict[int, dict] = {}
    proxy_models_cfg: Dict[int, dict] = {}  # sid -> {path, site_params}
    has_any_proxy = False

    for i, st in enumerate(storage_raw):
        proxy_path_rel = st.get("proxy_model")
        use_proxy = proxy_path_rel is not None and proxy_path_rel != ""

        if use_proxy:
            has_any_proxy = True
            resolved_path = _resolve_proxy_path(proxy_path_rel, project_root)
            site_params = _build_storage_proxy_params(st)
            proxy_models_cfg[i] = {
                "path": resolved_path,
                "site_params": site_params,
            }
            # Also provide ROM fallback parameters
            storage_params[i] = _proxy_storage_rom_defaults(
                site_params, st.get("capacity_mt", 25.0)
            )
        else:
            # Pure analytical ROM
            storage_params[i] = _build_storage_rom_config(st, i)
        storage_params[i]["min_purity"] = float(
            st.get("min_purity", DEFAULT_STORAGE_QUALITY_LIMITS["min_purity"])
        )
        storage_params[i]["max_impurities"] = deepcopy(
            st.get("max_impurities", DEFAULT_STORAGE_QUALITY_LIMITS["max_impurities"])
        )

    # ------ Connectivity ------
    conn_raw = raw.get("connectivity", {})
    connectivity: Dict[int, List[List[int]]] = {}
    for eid_key, routes in conn_raw.items():
        eid = int(eid_key)
        connectivity[eid] = [list(r) for r in routes]

    # ------ Disruptions ------
    dis_raw = raw.get("disruptions", {})
    disruption_cfg = {
        "scenario_family": dis_raw.get("scenario_family", "T"),
        "severity": dis_raw.get("severity", 0.5),
        "cross_correlation": dis_raw.get("cross_correlation", 0.0),
    }

    # ------ Mechanism ------
    mech_raw = raw.get("mechanism", {})
    mechanism_cfg = {
        "alpha": mech_raw.get("alpha", 0.5),
        "beta": mech_raw.get("beta", 0.3),
        "gamma": mech_raw.get("gamma", 1.0),
    }

    # ------ Simulation ------
    sim_raw = raw.get("simulation", {})
    episode_length = sim_raw.get("episode_length_months", 120)
    seed = sim_raw.get("seed", 42)
    # Carbon tax: can be a scalar, a dict with price_per_t_co2, or absent (0)
    ct_raw = sim_raw.get("carbon_tax", raw.get("carbon_tax", 0.0))
    if isinstance(ct_raw, dict):
        carbon_tax = float(ct_raw.get("price_per_t_co2", 0.0))
    elif ct_raw is None:
        carbon_tax = 0.0
    else:
        carbon_tax = float(ct_raw)
    electricity_price = float(sim_raw.get("electricity_price", raw.get("electricity_price", 65.0)))
    capture_subsidy = float(sim_raw.get("capture_subsidy", raw.get("capture_subsidy", 0.0)))
    storage_credit = float(sim_raw.get("storage_credit", raw.get("storage_credit", 0.0)))
    offspec_penalty = float(sim_raw.get("offspec_penalty", raw.get("offspec_penalty", 6.0)))
    energy_weight = float(sim_raw.get("energy_weight", raw.get("energy_weight", 0.03)))
    quality_weight = float(sim_raw.get("quality_weight", raw.get("quality_weight", 3.0)))
    extreme_scenarios = deepcopy(
        sim_raw.get("extreme_scenarios", raw.get("extreme_scenarios", []))
    )

    # ------ Assemble config dict ------
    # Storage defaults: use the first storage site's ROM params as defaults
    storage_defaults = {
        "max_injection_rate": _mtpa_to_monthly(3.0),
        "pressure_limit": 300.0,
        "initial_pressure": 100.0,
        "k_injection": 40.0,
        "k_dissipation": 0.05,
        "cross_well_coeff": 5.0,
        "injectivity_decline_rate": 0.15,
        "cumulative_capacity": 25.0,
    }

    config: Dict[str, Any] = {
        "network": {
            "num_emitters": num_emitters,
            "transport_modes": transport_modes,
            "num_storage_sites": num_storage,
            "emitter_defaults": {},
            "emitter_params": emitter_params,
            "transport_params": transport_params,
            "storage_defaults": storage_defaults,
            "storage_params": storage_params,
            "connectivity": connectivity,
        },
        "physical": physical_cfg,
        "disruption": disruption_cfg,
        "mechanism": mechanism_cfg,
        "carbon_tax": carbon_tax,
        "electricity_price": electricity_price,
        "capture_subsidy": capture_subsidy,
        "storage_credit": storage_credit,
        "offspec_penalty": offspec_penalty,
        "energy_weight": energy_weight,
        "quality_weight": quality_weight,
        "extreme_scenarios": extreme_scenarios,
        "episode_length": episode_length,
        "seed": seed,
    }

    # Attach proxy model configuration for PhysicalLayer to pick up
    if has_any_proxy:
        config["proxy_models"] = proxy_models_cfg

    # Attach original case metadata
    case_meta = raw.get("case", {})
    config["case_meta"] = {
        "name": case_meta.get("name", "Unknown"),
        "description": case_meta.get("description", ""),
        "region": case_meta.get("region", ""),
        "yaml_path": case_path,
    }

    return config


def _find_project_root(case_path: str) -> str:
    """Find the project root directory by walking up from the case file.

    Looks for a directory containing 'ccus_gym/' or 'CLAUDE.md'.
    Falls back to the parent of the case file's directory.
    """
    current = os.path.dirname(os.path.abspath(case_path))

    for _ in range(10):
        if (os.path.isdir(os.path.join(current, "ccus_gym"))
                or os.path.isfile(os.path.join(current, "CLAUDE.md"))):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    # Fallback: assume cases/ is one level below project root
    return os.path.dirname(os.path.dirname(os.path.abspath(case_path)))
