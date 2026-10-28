"""Shared ``Literal`` types for string options and the respective checking function."""
from __future__ import annotations

__all__ = [
    "Aggregation", "BoxKind", "DensityFallback", "DensityQuantity", "GridMode", "InterpMethod",
    "JointKind", "LoadOutput", "OnMissing", "SalinityType", "ScatterKind", "TemperatureTarget",
    "TemperatureType", "TimeMode", "check_choice",
]

from typing import Literal, get_args

# Output container of load_comfort
LoadOutput = Literal["xarray", "dict"]

# Policy options of the unit conversion
OnMissing = Literal["drop", "keep", "raise"]
DensityFallback = Literal["constant", "drop", "raise"]

# Depth interpolation
InterpMethod = Literal["pchip", "linear"]

# Grid time axis: Y(ear), M(onth), D(ay) and their combinations
GridMode = Literal["Y", "M", "D", "YM", "YD", "MD"]

# Aggregation name(s) understood by DataFrame.agg (plus "seasonal_mean" in Grid.bin)
Aggregation = str | list[str]

# Counting by year or by month
TimeMode = Literal["year", "month"]

# TEOS-10 conversions
TemperatureTarget = Literal["CT", "pt0"]
DensityQuantity = Literal["sigma0", "sigma1", "sigma2", "rho"]

# Plot options
BoxKind = Literal["box", "violin"]
ScatterKind = Literal["scatter", "hexbin"]
JointKind = Literal["scatter", "hist", "kde", "hex", "reg"]
TemperatureType = Literal["CT", "pt", "pt0", "t"]
SalinityType = Literal["SA", "SP"]


def check_choice(value, choices, option: str) -> None:
    """Raise ``ValueError`` if *value* is not one of the options of the ``Literal`` type *choices*."""
    options = get_args(choices)
    if value not in options:
        raise ValueError(f"{option} must be one of {options}, got {value!r}")
