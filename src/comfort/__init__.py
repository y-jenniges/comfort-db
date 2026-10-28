"""comfort-db - helper library for the COMFORT oceanographic dataset."""
from __future__ import annotations

__version__ = "0.1.0"

from .io import (
    connect,
    describe_variables,
    info,
    list_parameters,
    load_comfort,
    read_parameter,
    read_station,
    subset_region,
)
from .qc import QC_ALL, QC_GOOD, QCFilter
from .gridding import Grid, GridManager
from .units import UnitsConverter

__all__ = [
    "__version__",
    "connect",
    "describe_variables",
    "info",
    "list_parameters",
    "load_comfort",
    "read_parameter",
    "read_station",
    "subset_region",
    "QC_ALL",
    "QC_GOOD",
    "QCFilter",
    "Grid",
    "GridManager",
    "UnitsConverter",
]
