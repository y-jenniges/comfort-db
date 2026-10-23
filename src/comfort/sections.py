"""Along-track section/transect analysis for COMFORT oceanographic data."""
from __future__ import annotations

__all__ = [
    "bin_section",
    "section_difference",
]

import pandas as pd
from numpy.typing import ArrayLike


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
        along_bins (int or array-like): Number of equal-width bins, or
            explicit bin edges, along *along_col*.
        depth_bins (int or array-like): Same as *along_bins*, for *depth_col*.
        agg (str): Aggregation applied to *value_col* within each cell.
    Returns:
        pandas.DataFrame: One row per (along, depth) cell with columns
            [*along_col*, *depth_col*, *value_col*], holding the bin centres
            and aggregated value. Cells with no observations get NaN.
    """
    # Drop nans
    plot_df = df[[value_col, depth_col, along_col]].dropna()
    if plot_df.empty:
        return pd.DataFrame(columns=[along_col, depth_col, value_col])

    # Binning
    along_bin = pd.cut(plot_df[along_col], bins=along_bins)
    depth_bin = pd.cut(plot_df[depth_col], bins=depth_bins)
    binned = (
        plot_df.assign(_along_bin=along_bin, _depth_bin=depth_bin)
        .groupby(["_along_bin", "_depth_bin"], observed=False)[value_col]
        .agg(agg)
        .reset_index()
    )

    # Over-write old along and depth columns with the binned ones
    binned[along_col] = binned["_along_bin"].apply(lambda iv: iv.mid).astype(float)
    binned[depth_col] = binned["_depth_bin"].apply(lambda iv: iv.mid).astype(float)
    return binned[[along_col, depth_col, value_col]]


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
    # Merge section a and b
    merged = section_a.merge(section_b, on=[along_col, depth_col], suffixes=("_a", "_b"))

    # Compute differences wrt the parameter column
    merged[value_col] = merged[f"{value_col}_a"] - merged[f"{value_col}_b"]
    return merged[[along_col, depth_col, value_col]]
