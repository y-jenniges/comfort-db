"""Visualisation functions for COMFORT data.

Requires:
    pip install "comfort-db[plot]"
"""
from __future__ import annotations
try:
    import matplotlib
except ImportError:
    raise ImportError(
        "matplotlib is required for comfort.plot; "
        'install it with: pip install "comfort-db[plot]"'
    )

# --- profiles & sections: how does this vary with depth? ---
from .profile import plot_profile, plot_section

# --- relationships: how do two variables relate to each other? ---
from .relationships import plot_correlation, plot_joint, plot_ts_diagram

# --- distributions: what's the spread of values? ---
from .distribution import (
    boxplot,
    compare_stats,
    plot_histogram,
    plot_monthly_histogram,
)

# --- spatial: where is the data, and where is it missing? ---
from .spatial import (
    plot_lon_lat_range,
    plot_parameter_map,
    plot_missing_value_info_map_joint,
    plot_missing_value_info_map_over_depth,
    plot_spatial_distribution,
)

# --- coverage: how much data is there, and how complete is it? ---
from .coverage import (
    plot_annual_coverage,
    plot_counts_bar,
    plot_depth_coverage,
    plot_duplicates,
    plot_missing_value_info,
)

__all__ = [
    # profiles & sections
    "plot_profile",
    "plot_section",
    # relationships
    "plot_correlation",
    "plot_joint",
    "plot_ts_diagram",
    # distributions
    "boxplot",
    "compare_stats",
    "plot_histogram",
    "plot_monthly_histogram",
    # spatial
    "plot_spatial_distribution",
    "plot_lon_lat_range",
    "plot_parameter_map",
    "plot_missing_value_info_map_joint",
    "plot_missing_value_info_map_over_depth",
    # coverage
    "plot_annual_coverage",
    "plot_counts_bar",
    "plot_depth_coverage",
    "plot_duplicates",
    "plot_missing_value_info",
]
