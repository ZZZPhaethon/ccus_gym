"""Read-only physics tools for LLM agents.

Provides ground-truth queries against the PhysicalLayer, enabling
tool-augmented agents to make physically informed decisions.
"""

from __future__ import annotations

from typing import Any, Dict


class PhysicsToolkit:
    """Thin read-only wrapper over PhysicalLayer."""

    __slots__ = ("pl",)

    def __init__(self, physical_layer: Any) -> None:
        self.pl = physical_layer

    def route_quote(self, mode: str) -> dict[str, Any]:
        """Available capacity and cost for a transport mode."""
        if mode == "pipeline" and self.pl.pipeline is not None:
            p = self.pl.pipeline
            return {
                "available_Mt": round(p.available_capacity, 4),
                "cost_per_t": round(p.base_cost, 1),
                "disrupted": p.is_disrupted,
            }
        if mode == "ship" and self.pl.ships is not None:
            s = self.pl.ships
            return {
                "available_Mt": round(s.get_monthly_capacity(), 4),
                "cost_per_t": round(s.base_cost, 1),
                "disrupted": s.is_disrupted,
            }
        if mode == "rail" and self.pl.rail is not None:
            r = self.pl.rail
            return {
                "available_Mt": round(r.get_monthly_capacity(), 4),
                "cost_per_t": round(r.base_cost, 1),
                "disrupted": r.is_disrupted,
            }
        return {"available_Mt": 0.0, "cost_per_t": 0.0, "disrupted": True}

    def storage_headroom(self, site_id: int) -> dict[str, Any]:
        """Pressure state and injectable capacity for a storage site."""
        s = self.pl.storage_sites.get(site_id)
        if s is None:
            return {"error": f"no site {site_id}"}
        return {
            "pressure_bar": round(s.current_pressure, 1),
            "limit_bar": round(s.pressure_limit, 1),
            "headroom_pct": round(
                (1 - s.current_pressure / s.pressure_limit) * 100, 1
            ),
            "max_injectable_Mt": round(s.get_max_injectable(), 4),
            "injectivity": round(s.current_injectivity, 3),
        }

    def feasibility_check(
        self, planned: Dict[int, float]
    ) -> dict[str, Any]:
        """Check if planned injection rates would exceed pressure limits."""
        violations = {}
        for sid, rate in planned.items():
            s = self.pl.storage_sites.get(sid)
            if s is None:
                continue
            dp = s.k_injection * rate / max(s.current_injectivity, 0.01)
            for other_sid, other_rate in planned.items():
                if other_sid != sid:
                    dp += s.cross_well_coeff * other_rate
            projected = (
                s.current_pressure
                + dp
                - s.k_dissipation * (s.current_pressure - s.initial_pressure)
            )
            if projected > s.pressure_limit:
                violations[sid] = {
                    "projected_bar": round(projected, 1),
                    "over_by": round(projected - s.pressure_limit, 1),
                }
        return {"feasible": len(violations) == 0, "violations": violations}
