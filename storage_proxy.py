"""Storage proxy model wrapper for the CCUS-Gym environment.

Wraps the ML-based storage proxy model (.pkl) from the Teesside CCS project
and provides a clean interface for pressure prediction and safe injection rate
calculation.

The proxy model predicts monthly pressure changes (dome pressure and BHP) as
well as actual stored volume, given current state and injection rate.

Input features (7):
    [avg_injection_rate_MSCF, fracture_pressure_gradient,
     transmissibility_multiplier, aquifer_pv_multiplier,
     dome_pressure_psi, bhp_psi, cumulative_stored_per_well_MSCF]

Output (3):
    [delta_dome_pressure_psi, delta_bhp_psi, delta_stored_MSCF]

All unit conversions between SI (tonnes, Pa, meters) and the model's units
(MSCF, psi, feet) are handled internally.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ===================================================================
# Numpy compatibility shim for older sklearn/pickle models
# ===================================================================
# Models pickled with numpy < 2.0 may reference numpy.core.numeric.ComplexWarning
# which was removed in numpy 2.0+.  Patch it so unpickling does not fail.

try:
    import numpy.core.numeric as _ncn
    if not hasattr(_ncn, "ComplexWarning"):
        try:
            from numpy.exceptions import ComplexWarning as _CW
        except ImportError:
            _CW = DeprecationWarning  # type: ignore[misc]
        _ncn.ComplexWarning = _CW  # type: ignore[attr-defined]
except Exception:
    pass


# ===================================================================
# Unit conversion functions
# Preserved from the original Teesside CCS codebase.
# ===================================================================

# CO2 properties:
#   Molar mass M = 44.01 g/mol
#   At STP (0 C, 1 atm): 1 mol = 22.414 L
#   1 MSCF = 1000 standard cubic feet = 28.3168 m^3
#
# Conversion chain:
#   tonnes -> kg -> mol -> Nm3 -> SCF -> MSCF
#   tonnes * 1000 / 44.01 * 22.414e-3 * 35.3147 / 1000
#   Simplified: tonnes * 1000 / 44.01 * 22.4 / 1e3
#   (using 22.4 L/mol approximation, and the factor 35.3147 ft3/m3
#    is implicitly folded into the original code's approximation)

def ton_to_MSCF(tons: float) -> float:
    """Convert tonnes of CO2 to thousands of standard cubic feet (MSCF).

    Uses the same approximation as the original Teesside proxy model code.
    """
    return tons * 1000.0 / 44.01 * 22.4 / 1e3


def MSCF_to_ton(mscf: float) -> float:
    """Convert MSCF of CO2 back to tonnes."""
    return mscf * 1e3 / 22.4 * 44.01 / 1000.0


def pa_to_psi(pa: float) -> float:
    """Convert Pascal to pounds per square inch."""
    return pa * 0.000145038


def psi_to_pa(psi: float) -> float:
    """Convert psi to Pascal."""
    return psi / 0.000145038


def meters_to_feet(m: float) -> float:
    """Convert meters to feet."""
    return m * 3.28084


# ===================================================================
# StorageProxyModel
# ===================================================================

class StorageProxyModel:
    """Wrapper for the ML storage proxy model from the Teesside CCS project.

    Provides two main functions:
    1. predict_monthly_update: given current state + injection rate, predict
       pressure changes and actual stored volume for one month.
    2. predict_max_safe_rate: find the maximum injection rate that keeps
       dome pressure and BHP within safety limits (90% of fracture pressure).
    """

    def __init__(self, model_path: str, site_params: dict) -> None:
        """
        Args:
            model_path: Path to the .pkl file containing the trained proxy model.
            site_params: Dict with keys:
                fracture_pressure_gradient: float (psi/ft)
                transmissibility_multiplier: float
                aquifer_pv_multiplier: float
                num_wells: int
                dome_depth_m: float (meters)
                bottom_hole_depth_m: float (meters)
                initial_dome_pressure_mpa: float (MPa)
                initial_bhp_mpa: float (MPa)
                capacity_mt: float (Mt total capacity)
        """
        self.model: Any = None
        self.model_path = model_path
        self.site_params = dict(site_params)
        self._model_loaded = False

        # Try to load the model
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self._model_loaded = True
            logger.info("Loaded storage proxy model from %s", model_path)
        except Exception as e:
            logger.warning(
                "Failed to load proxy model from %s: %s. "
                "Will fall back to analytical ROM.",
                model_path, e,
            )
            self._model_loaded = False

        # Store key physical parameters
        self.fracture_pressure_gradient = site_params["fracture_pressure_gradient"]
        self.transmissibility_multiplier = site_params["transmissibility_multiplier"]
        self.aquifer_pv_multiplier = site_params["aquifer_pv_multiplier"]
        self.num_wells = int(site_params["num_wells"])
        self.dome_depth_m = site_params["dome_depth_m"]
        self.bottom_hole_depth_m = site_params["bottom_hole_depth_m"]

        # Compute pressure limits: 90% of fracture pressure
        # Fracture pressure = depth_ft * fracture_gradient (psi)
        # Safety limit = 90% of fracture pressure, converted to Pa
        self.dome_pressure_limit_pa = self._compute_limit(self.dome_depth_m)
        self.bhp_limit_pa = self._compute_limit(self.bottom_hole_depth_m)

        logger.info(
            "Proxy model pressure limits: dome=%.2f MPa, BHP=%.2f MPa",
            self.dome_pressure_limit_pa / 1e6,
            self.bhp_limit_pa / 1e6,
        )

    @property
    def is_loaded(self) -> bool:
        """Whether the proxy model pkl was successfully loaded."""
        return self._model_loaded

    def _compute_limit(self, depth_m: float) -> float:
        """Compute 90% fracture pressure limit in Pa for a given depth.

        fracture_pressure_psi = depth_ft * fracture_pressure_gradient
        safety_limit = 0.9 * fracture_pressure
        """
        depth_ft = meters_to_feet(depth_m)
        frac_pressure_psi = depth_ft * self.fracture_pressure_gradient
        safe_pressure_psi = 0.9 * frac_pressure_psi
        return psi_to_pa(safe_pressure_psi)

    def _build_input_array(
        self,
        avg_daily_injection_rate_t: float,
        current_dome_pressure_pa: float,
        current_bhp_pa: float,
        cumulative_stored_per_well_t: float,
    ) -> list:
        """Build the 7-feature input array for the proxy model.

        Features:
            0: avg injection rate (MSCF/day)
            1: fracture pressure gradient (psi/ft)
            2: transmissibility multiplier
            3: aquifer pore volume multiplier
            4: dome pressure (psi)
            5: BHP (psi)
            6: cumulative stored per well (MSCF)
        """
        return [
            ton_to_MSCF(avg_daily_injection_rate_t),
            self.fracture_pressure_gradient,
            self.transmissibility_multiplier,
            self.aquifer_pv_multiplier,
            pa_to_psi(current_dome_pressure_pa),
            pa_to_psi(current_bhp_pa),
            ton_to_MSCF(cumulative_stored_per_well_t),
        ]

    def predict_monthly_update(
        self,
        avg_daily_injection_rate_t: float,
        current_dome_pressure_pa: float,
        current_bhp_pa: float,
        cumulative_stored_per_well_t: float,
    ) -> Dict[str, float]:
        """Predict pressure changes for one month of injection.

        Args:
            avg_daily_injection_rate_t: Average daily injection rate per well
                (tonnes/day).
            current_dome_pressure_pa: Current dome pressure (Pa).
            current_bhp_pa: Current bottom-hole pressure (Pa).
            cumulative_stored_per_well_t: Cumulative CO2 stored per well
                (tonnes).

        Returns:
            Dict with keys:
                delta_dome_pressure_pa: Dome pressure change this month (Pa).
                delta_bhp_pa: BHP change this month (Pa).
                delta_stored_t: Actual CO2 stored this month per well (tonnes).
                    Negative values are clipped to 0.
                max_injection_rate_t_per_year: Predicted max safe rate for
                    next month (tonnes/year per well), based on high-rate
                    forward prediction.

        Raises:
            RuntimeError: If the proxy model is not loaded.
        """
        if not self._model_loaded:
            raise RuntimeError(
                "Proxy model not loaded. Cannot call predict_monthly_update."
            )

        input_array = self._build_input_array(
            avg_daily_injection_rate_t,
            current_dome_pressure_pa,
            current_bhp_pa,
            cumulative_stored_per_well_t,
        )

        try:
            prediction = self.model.predict([input_array])
        except Exception as e:
            logger.error("Proxy model prediction failed: %s", e)
            # Return zero changes as safe fallback
            return {
                "delta_dome_pressure_pa": 0.0,
                "delta_bhp_pa": 0.0,
                "delta_stored_t": 0.0,
                "max_injection_rate_t_per_year": 0.0,
            }

        # prediction shape: (1, 3)
        # [delta_dome_pressure_psi, delta_bhp_psi, delta_stored_MSCF]
        delta_dome_psi = float(prediction[0][0])
        delta_bhp_psi = float(prediction[0][1])
        delta_stored_mscf = float(prediction[0][2])

        # Convert back to SI
        delta_dome_pa = psi_to_pa(delta_dome_psi)
        delta_bhp_pa = psi_to_pa(delta_bhp_psi)
        delta_stored_t = max(0.0, MSCF_to_ton(delta_stored_mscf))

        # Forward prediction for max safe rate next month
        # Use a high injection rate to get the proxy model's estimate of
        # max throughput (stored volume per month at high rate)
        # 6e6 tonnes/year / 365 = ~16438 tonnes/day as high test rate
        high_rate_t_day = 6e6 / 365.0
        # Updated pressures after this month
        new_dome_pa = current_dome_pressure_pa + delta_dome_pa
        new_bhp_pa = current_bhp_pa + delta_bhp_pa
        new_cumulative_t = cumulative_stored_per_well_t + delta_stored_t

        high_input = self._build_input_array(
            high_rate_t_day, new_dome_pa, new_bhp_pa, new_cumulative_t,
        )
        try:
            high_pred = self.model.predict([high_input])
            # The stored volume at high rate gives max monthly capacity
            max_stored_mscf = float(high_pred[0][2])
            max_stored_t_month = max(0.0, MSCF_to_ton(max_stored_mscf))
            max_rate_t_year = max_stored_t_month * 12.0
        except Exception:
            max_rate_t_year = 0.0

        return {
            "delta_dome_pressure_pa": delta_dome_pa,
            "delta_bhp_pa": delta_bhp_pa,
            "delta_stored_t": delta_stored_t,
            "max_injection_rate_t_per_year": max_rate_t_year,
        }

    def predict_max_safe_rate(
        self,
        current_dome_pressure_pa: float,
        current_bhp_pa: float,
        cumulative_stored_per_well_t: float,
        days_per_month: float = 30.0,
    ) -> float:
        """Predict maximum safe injection rate for the next month.

        Uses iterative reduction: start from a high candidate rate, predict
        pressures, and reduce if limits would be exceeded. This mirrors
        the adjust_max_injection_rate_new_proxy() logic from the original code.

        Args:
            current_dome_pressure_pa: Current dome pressure (Pa).
            current_bhp_pa: Current bottom-hole pressure (Pa).
            cumulative_stored_per_well_t: Cumulative CO2 stored per well (t).
            days_per_month: Days in the current month (default 30).

        Returns:
            Maximum safe daily injection rate per well (tonnes/day).
            Returns 0.0 if the model is not loaded or pressures are already
            at the limit.
        """
        if not self._model_loaded:
            return 0.0

        # Start from a high candidate: ~16k t/day per well (6 Mt/yr)
        candidate_t_day = 6e6 / 365.0

        for _ in range(200):
            # Predict what would happen if we inject at this rate for a month
            input_array = self._build_input_array(
                candidate_t_day,
                current_dome_pressure_pa,
                current_bhp_pa,
                # Add projected monthly injection to cumulative
                cumulative_stored_per_well_t + candidate_t_day * days_per_month,
            )

            try:
                prediction = self.model.predict([input_array])
            except Exception:
                return 0.0

            predicted_dome_pa = current_dome_pressure_pa + psi_to_pa(
                float(prediction[0][0])
            )
            predicted_bhp_pa = current_bhp_pa + psi_to_pa(
                float(prediction[0][1])
            )

            # Check if pressures are within limits
            if (predicted_dome_pa <= self.dome_pressure_limit_pa
                    and predicted_bhp_pa <= self.bhp_limit_pa):
                return candidate_t_day

            # Reduce by 1% and try again
            candidate_t_day *= 0.99

            # Floor: if rate is negligible, return zero
            if candidate_t_day <= 1.0:
                return 0.0

        # Did not converge -- return zero as safe fallback
        logger.warning(
            "predict_max_safe_rate did not converge. "
            "Dome=%.2f MPa (limit %.2f), BHP=%.2f MPa (limit %.2f)",
            current_dome_pressure_pa / 1e6,
            self.dome_pressure_limit_pa / 1e6,
            current_bhp_pa / 1e6,
            self.bhp_limit_pa / 1e6,
        )
        return 0.0
