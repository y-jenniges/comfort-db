"""Database communication, querying and schema management for COMFORT."""
from __future__ import annotations

from .information import (
    count_negative_samples,
    count_samples_over_time,
    count_samples_per_parameter,
    does_table_exist,
    find_spatiotemporal_duplicates,
    get_min_max_dates,
    get_minmax,
    get_names_of_all_parameter_tables,
    get_num_samples,
    get_table_as_df,
    param_label,
    param_table_name,
)
from .structure import create_combined_parameter_table

__all__ = [
    "count_negative_samples",
    "count_samples_over_time",
    "count_samples_per_parameter",
    "create_combined_parameter_table",
    "does_table_exist",
    "find_spatiotemporal_duplicates",
    "get_min_max_dates",
    "get_minmax",
    "get_names_of_all_parameter_tables",
    "get_num_samples",
    "get_table_as_df",
    "param_label",
    "param_table_name",
]
