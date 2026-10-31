"""Along-track section/transect analysis for COMFORT oceanographic data."""
from __future__ import annotations

__all__ = [
    "bin_section",
    "section_difference",
]

import pandas as pd
from numpy.typing import ArrayLike

from .gridding import _cut_cells
from .util.columns import find_col


def bin_section(df: pd.DataFrame,
                value_col: str,
                depth_col: str,
                along_col: str,
                along_bins: int | ArrayLike = 60,
                depth_bins: int | ArrayLike = 40,
                agg: str = "mean",
                ) -> pd.DataFrame:
    """Bin scattered section observations onto an (along-track x depth) grid.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        value_col (str): Column with measured values.
        depth_col (str): Column with depth [m].
        along_col (str): Column with along-track position, e.g. the output
            of :func:`~comfort.geo.along_track_distance`.
        along_bins (int or array-like): Number of equal-width bins or
            explicit bin edges, along *along_col*. Bins are left-closed,
            ``[a, b)``, with the maximum edge included (as in gridding).
        depth_bins (int or array-like): Same as *along_bins*, for *depth_col*.
            Unlike :class:`~comfort.gridding.Grid`, the bin is labelled by its centre here (not by top node).
        agg (str): Aggregation applied to *value_col* within each cell.
    Returns:
        pandas.DataFrame: One row per (along, depth) cell with columns
            [*along_col*, *depth_col*, *value_col*], holding the bin centres
            (also for depth, see *depth_bins*) and aggregated value. Cells with no observations get NaN.
    """
    # Resolve column names (insensitive to casing)
    value_col = find_col(df, value_col)
    depth_col = find_col(df, depth_col)
    along_col = find_col(df, along_col)

    # Drop nans
    plot_df = df[[value_col, depth_col, along_col]].dropna()
    if plot_df.empty:
        return pd.DataFrame(columns=[along_col, depth_col, value_col])

    # Binning (left-closed like the gridding), labelled by bin centre
    along_bin = _cut_to_centres(plot_df[along_col], along_bins)
    depth_bin = _cut_to_centres(plot_df[depth_col], depth_bins)
    binned = (
        plot_df.assign(_along_bin=along_bin, _depth_bin=depth_bin)
        .groupby(["_along_bin", "_depth_bin"], observed=False)[value_col]
        .agg(agg)
        .reset_index()
    )

    # Over-write old along and depth columns with the bin centres
    binned[along_col] = binned["_along_bin"].astype(float)
    binned[depth_col] = binned["_depth_bin"].astype(float)
    return binned[[along_col, depth_col, value_col]]


def _cut_to_centres(values: pd.Series, bins: int | ArrayLike) -> pd.Series:
    """Bin *values* into left-closed bins ``[a, b)`` and label each bin by its centre."""
    _, edges = pd.cut(values, bins=bins, right=False, retbins=True)  # get bin edges
    centres = (edges[:-1] + edges[1:]) / 2  # define centres
    return _cut_cells(values, edges, centres)  # map the values


def section_difference(section_a: pd.DataFrame,
                       section_b: pd.DataFrame,
                       value_col: str,
                       depth_col: str,
                       along_col: str,
                       ) -> pd.DataFrame:
    """Difference of two binned sections cell-by-cell (*section_a* minus *section_b*).

    Both inputs must be :func:`bin_section` output computed with identical
    *along_bins*/*depth_bins* (same explicit bin edges).

    Args:
        section_a (pandas.DataFrame): Output of :func:`bin_section`.
        section_b (pandas.DataFrame): Output of :func:`bin_section`, same
            binning as *section_a*.
        value_col (str): Column with measured values.
        depth_col (str): Column with depth [m].
        along_col (str): Column with along-track position.
    Returns:
        pandas.DataFrame: One row per (along, depth) cell present in both
            inputs, with *value_col* holding ``section_a - section_b``.
    """
    # Resolve column names (insensitive to casing)
    along_col = find_col(section_a, along_col)
    depth_col = find_col(section_a, depth_col)
    value_col = find_col(section_a, value_col)

    # Merge section a and b
    merged = section_a.merge(section_b, on=[along_col, depth_col], suffixes=("_a", "_b"))

    # Compute differences wrt the parameter column
    merged[value_col] = merged[f"{value_col}_a"] - merged[f"{value_col}_b"]
    return merged[[along_col, depth_col, value_col]]
