"""Deterministic dimension/unit checks, not physics-model selection."""
from __future__ import annotations
import math
from .models import PhysicalQuantity

# SI base dimensions: mass, length, time, temperature, amount, current, luminous intensity.
UNITS = {
    "1": (1.0, 0.0, (0,0,0,0,0,0,0)),
    "m": (1.0, 0.0, (0,1,0,0,0,0,0)), "mm": (0.001, 0.0, (0,1,0,0,0,0,0)),
    "s": (1.0, 0.0, (0,0,1,0,0,0,0)),
    "K": (1.0, 0.0, (0,0,0,1,0,0,0)), "degC": (1.0, 273.15, (0,0,0,1,0,0,0)),
    "Pa": (1.0, 0.0, (1,-1,-2,0,0,0,0)), "bar": (100000.0, 0.0, (1,-1,-2,0,0,0,0)),
    "m/s": (1.0, 0.0, (0,1,-1,0,0,0,0)), "m2/s": (1.0, 0.0, (0,2,-1,0,0,0,0)),
    "kg/m3": (1.0, 0.0, (1,-3,0,0,0,0,0)),
    "W": (1.0, 0.0, (1,2,-3,0,0,0,0)), "W/m3": (1.0, 0.0, (1,-1,-3,0,0,0,0)),
}


def si_value(quantity: PhysicalQuantity) -> float:
    if quantity.unit not in UNITS:
        raise ValueError(f"Unsupported unit {quantity.unit!r}; explicit unit support is required, not a guessed conversion.")
    scale, offset, dimensions = UNITS[quantity.unit]
    if quantity.dimensions != dimensions:
        raise ValueError("Quantity dimensions disagree with its unit.")
    return quantity.value * scale + offset


def assert_equivalent(source: PhysicalQuantity, normalized: PhysicalQuantity) -> None:
    for key in ("quantity", "dimensions", "region", "patch", "material", "time_dependence", "reference_definition"):
        if getattr(source, key) != getattr(normalized, key):
            raise ValueError(f"Physical quantity target/meaning changed: {key}.")
    if not math.isclose(si_value(source), si_value(normalized), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Physical quantity conversion does not preserve the value.")
