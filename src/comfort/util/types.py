"""Shared ``Literal`` types for string options and the respective checking function."""
from __future__ import annotations

__all__ = [
    "Aggregation", "BoxKind", "DensityFallback", "DensityQuantity", "GridMode", "InterpMethod",
    "JointKind", "LoadOutput", "OnMissing", "SalinityType", "ScatterKind", "TemperatureTarget",
    "TemperatureType", "TimeMode", "check_choice", "check_timestamp",
]

from datetime import date, datetime
from typing import Literal, get_args

# Long timestamp format that all date and time bounds are given in
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

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


def check_timestamp(value, option: str) -> None:
    """Raise ``ValueError`` if *value* is not in the :data:`_TIMESTAMP_FORMAT` format."""
    # Accept datetime and Timestamp objects (datetime.date does not support time of day)
    valid = not isinstance(value, date) or isinstance(value, datetime)
    if isinstance(value, str):
        # Str Object
        try:
            # Apply time format
            datetime.strptime(value, _TIMESTAMP_FORMAT)
            valid = True
        except ValueError:
            valid = False
    if not valid:
        # Invalid format
        raise ValueError(f"{option} must be given as 'YYYY-MM-DD HH:MM:SS', got {value!r}")
