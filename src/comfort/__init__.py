"""comfort-db - helper library for the COMFORT oceanographic dataset."""
from __future__ import annotations

__version__ = "0.1.0"

from .io import connect, info, load_comfort, describe_variables, subset_region, list_parameters, read_parameter
from .qc import QC_ALL, QC_GOOD, QCFilter, apply_qc_flags, flag_salinity_like_oxygen
from .database.information import get_table_as_df, does_table_exist
from .profile_analysis import average_duplicate_records_per_profile, detect_depth_col
from .physics import convert_to_potential_temperature
from .gridding import Grid, GridManager, average_duplicate_locations, seasonal_mean
from .units import UnitsConverter
from .scaling import ParamScaler
from .util.sqlite_utils import vacuum

__all__ = [
    "connect",
    "info",
    "load_comfort",
    "describe_variables",
    "subset_region",
    "list_parameters",
    "read_parameter",
    "QC_ALL",
    "QC_GOOD",
    "QCFilter",
    "apply_qc_flags",
    "get_table_as_df",
    "does_table_exist",
    "Grid",
    "GridManager",
    "UnitsConverter",
    "ParamScaler",
    "average_duplicate_locations",
    "average_duplicate_records_per_profile",
    "convert_to_potential_temperature",
    "detect_depth_col",
    "seasonal_mean",
    "vacuum",
]
