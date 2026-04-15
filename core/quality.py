"""Utilities for composition-aware CCUS modelling.

This module provides a lightweight abstraction for representing
capture-method-specific CO2 stream quality, mixing streams, and checking
simple purity / impurity acceptance constraints.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple


COMPONENT_KEYS: Tuple[str, ...] = (
    "co2",
    "n2",
    "o2",
    "h2o",
    "h2s",
    "ch4",
    "co",
)


CAPTURE_METHOD_LIBRARY: Dict[str, Dict[str, object]] = {
    "post_combustion": {
        "base_purity": 0.90,
        "base_cost_per_t": 55.0,
        "base_energy_mwh_per_t": 0.32,
        "composition": {
            "co2": 0.90,
            "n2": 0.05,
            "o2": 0.02,
            "h2o": 0.015,
            "h2s": 0.002,
            "ch4": 0.01,
            "co": 0.003,
        },
    },
    "pre_combustion": {
        "base_purity": 0.965,
        "base_cost_per_t": 72.0,
        "base_energy_mwh_per_t": 0.38,
        "composition": {
            "co2": 0.965,
            "n2": 0.005,
            "o2": 0.002,
            "h2o": 0.008,
            "h2s": 0.01,
            "ch4": 0.008,
            "co": 0.002,
        },
    },
    "oxy_fuel": {
        "base_purity": 0.975,
        "base_cost_per_t": 68.0,
        "base_energy_mwh_per_t": 0.36,
        "composition": {
            "co2": 0.975,
            "n2": 0.002,
            "o2": 0.01,
            "h2o": 0.006,
            "h2s": 0.002,
            "ch4": 0.003,
            "co": 0.002,
        },
    },
}


DEFAULT_STORAGE_QUALITY_LIMITS: Dict[str, object] = {
    "min_purity": 0.93,
    "max_impurities": {
        "h2s": 0.01,
        "h2o": 0.02,
        "o2": 0.03,
        "n2": 0.08,
        "ch4": 0.04,
        "co": 0.01,
    },
}


def normalize_composition(
    composition: Mapping[str, float] | None,
    *,
    purity_hint: float | None = None,
) -> Dict[str, float]:
    """Return a valid composition dict over COMPONENT_KEYS summing to 1.0."""
    if composition is None:
        purity = max(0.0, min(1.0, purity_hint if purity_hint is not None else 0.95))
        remainder = max(0.0, 1.0 - purity)
        split = remainder / max(len(COMPONENT_KEYS) - 1, 1)
        return {
            key: purity if key == "co2" else split
            for key in COMPONENT_KEYS
        }

    cleaned = {key: max(0.0, float(composition.get(key, 0.0))) for key in COMPONENT_KEYS}
    total = sum(cleaned.values())
    if total <= 1e-12:
        return normalize_composition(None, purity_hint=purity_hint)
    return {key: value / total for key, value in cleaned.items()}


def method_defaults(capture_method: str) -> Dict[str, object]:
    """Look up default parameters for a capture method."""
    return CAPTURE_METHOD_LIBRARY.get(capture_method, CAPTURE_METHOD_LIBRARY["post_combustion"])


def composition_to_vector(composition: Mapping[str, float]) -> List[float]:
    return [float(composition.get(key, 0.0)) for key in COMPONENT_KEYS]


def compute_effective_stream(
    capture_method: str,
    purification_effort: float,
    *,
    base_purity: float | None = None,
    base_composition: Mapping[str, float] | None = None,
) -> Tuple[float, Dict[str, float]]:
    """Apply a simple purification model to a stream.

    Purification effort in [0, 1] increases CO2 purity and proportionally
    reduces non-CO2 components while preserving a normalized composition.
    """
    defaults = method_defaults(capture_method)
    raw_purity = float(
        base_purity if base_purity is not None else defaults["base_purity"]
    )
    raw_comp = normalize_composition(
        base_composition if base_composition is not None else defaults["composition"],
        purity_hint=raw_purity,
    )

    effort = max(0.0, min(1.0, float(purification_effort)))
    target_purity = min(0.995, raw_purity + 0.03 * effort)
    raw_co2 = raw_comp.get("co2", raw_purity)
    if raw_co2 >= 1.0 - 1e-9:
        return raw_co2, raw_comp

    impurity_scale = max(0.0, (1.0 - target_purity) / max(1.0 - raw_co2, 1e-9))
    adjusted = {}
    for key, value in raw_comp.items():
        adjusted[key] = value if key == "co2" else value * impurity_scale
    adjusted["co2"] = target_purity
    return target_purity, normalize_composition(adjusted, purity_hint=target_purity)


def blend_streams(streams: Iterable[Tuple[float, Mapping[str, float]]]) -> Dict[str, float]:
    """Blend a collection of (volume, composition) streams."""
    totals = {key: 0.0 for key in COMPONENT_KEYS}
    total_volume = 0.0
    for volume, composition in streams:
        vol = max(0.0, float(volume))
        if vol <= 0.0:
            continue
        total_volume += vol
        normalized = normalize_composition(composition)
        for key in COMPONENT_KEYS:
            totals[key] += vol * normalized.get(key, 0.0)

    if total_volume <= 1e-12:
        return normalize_composition({"co2": 1.0})
    return {key: totals[key] / total_volume for key in COMPONENT_KEYS}


def storage_quality_penalty(
    composition: Mapping[str, float],
    *,
    min_purity: float,
    max_impurities: Mapping[str, float] | None = None,
) -> Tuple[float, bool]:
    """Return a soft penalty score and violation flag for a blended stream."""
    comp = normalize_composition(composition)
    limits = max_impurities or DEFAULT_STORAGE_QUALITY_LIMITS["max_impurities"]
    penalty = max(0.0, float(min_purity) - comp.get("co2", 0.0))
    violated = penalty > 0.0
    for key, limit in limits.items():
        excess = max(0.0, comp.get(key, 0.0) - float(limit))
        penalty += excess
        violated = violated or excess > 0.0
    return penalty, violated
