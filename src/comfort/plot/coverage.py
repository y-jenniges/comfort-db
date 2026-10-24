"""Data-coverage and completeness diagnostics, i.e. how much data is there,
how complete is it (per parameter/year/month)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..database.information import param_label
from ..util.columns import find_col, resolve_param_columns
from ._helpers import _finish, _get_or_create_ax, _save_fig

if TYPE_CHECKING:
    import matplotlib.axes

logger = logging.getLogger(__name__)


def plot_annual_coverage(
    dfs: dict[str, pd.DataFrame],
    *,
    date_col: str = "DATEANDTIME",
    normalise: bool = False,
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes:
    """Line plot of observation count per year for multiple parameters.

    Counts the loaded DataFrames. To count straight from the database (without loading
    the data) use :func:`comfort.database.count_samples_over_time` and
    :func:`plot_counts_bar`.

    Args:
        dfs (dict): Dictionary mapping parameter names to DataFrames, each containing
            a datetime column.
        date_col: Column name containing datetime values.
        normalise: If ``True`` show counts relative to the per-parameter
            maximum (0-1 scale).
        ax: Axes to draw into.
        save_as: Save path.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib Axes.
    """
    ax, standalone = _get_or_create_ax(ax)

    # Iterate over parameter dfs
    for parameter, df in dfs.items():
        # Per-year counts
        counts = pd.to_datetime(df[find_col(df, date_col)]).dt.year.value_counts().sort_index()

        # Normalise
        if normalise:
            counts = counts / counts.max()

        # Plot
        label = parameter.capitalize() if parameter.isupper() else parameter
        ax.plot(counts.index, counts.values, label=label, marker=".", markersize=3)

    # Labels and format
    ax.set_xlabel("Year")
    ax.set_ylabel("Relative coverage" if normalise else "Number of observations")
    ax.set_title("Annual data coverage per parameter")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return _finish(ax, standalone, save_as, dpi)


def plot_depth_coverage(
    df_wide: pd.DataFrame,
    parameters: list[str],
    *,
    depth_col: str = "LEV_M",
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes:
    """Line plot of value count per depth level, for multiple parameters.

    Args:
        df_wide: Wide table with a depth column and one column per parameter.
        parameters: Parameter column names to plot.
        depth_col: Column name containing depth values.
        ax: Axes to draw into.
        save_as: Save path.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib Axes.
    """
    # Resolve column names (insensitive to casing)
    depth_col = find_col(df_wide, depth_col)

    ax, standalone = _get_or_create_ax(ax)

    # Get sorted depth levels
    depths = sorted(df_wide[depth_col].dropna().unique())

    # Plot count-vs-depth per parameter
    for col in resolve_param_columns(df_wide, parameters):
        counts = [df_wide.loc[df_wide[depth_col] == d, col].notna().sum() for d in depths]
        label = param_label(col).capitalize()
        ax.plot(counts, depths, marker="o", label=label, markersize=4)

    # Labels and format
    ax.invert_yaxis()
    ax.set_xlabel("Number of non-NaN values")
    ax.set_ylabel("Depth [m]")
    ax.set_title("Data coverage per depth level")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return _finish(ax, standalone, save_as, dpi)


def plot_counts_bar(
    df_count: pd.DataFrame,
    tick_style: str | None = "year",
    *,
    title: str = "Number of samples per year",
    month_ticks: list[str] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes | None:
    """Horizontal bar chart with value labels for any category/count table.

    Plots the output of the counting functions in :mod:`comfort.database`:
    :func:`~comfort.database.count_samples_over_time` (per year or month, set
    *tick_style* to ``'year'`` or ``'month'``),
    :func:`~comfort.database.count_samples_per_parameter` and
    :func:`~comfort.database.count_negative_samples` (select the ``parameter`` and
    ``proportion of negative samples`` columns first; *tick_style* ``None``).

    Args:
        df_count: First column is the category (year, month, parameter, ...),
            second is the count.
        tick_style: ``'year'``, ``'month'`` or ``None`` (no special tick
            formatting, e.g. for per-parameter counts).
        title: Plot title.
        month_ticks: 12 month labels.
        ax: Axes to draw into.
        save_as: Save path.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib Axes, or ``None`` when data is empty.
    """
    # Define months labels
    if month_ticks is None:
        month_ticks = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Check input df
    if df_count is None or df_count.empty:
        logger.warning("plot_counts_bar: passed dataframe is empty")
        return None

    ax, standalone = _get_or_create_ax(ax)

    # Determine x and y
    cols = df_count.columns
    x = df_count[cols[0]].map(str).tolist()
    y = df_count[cols[1]].tolist()

    # Barplot
    ax.barh(x, y)
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.text(yi, i, str(yi), ha="left", va="center")
    ax.set_title(title)
    ax.set_ylabel("Count")

    # Draw ticks and labels
    if tick_style == "year" and len(df_count) > 0:
        min_y = df_count.iloc[0, 0]
        max_y = df_count.iloc[-1, 0]
        min_y_rounded = np.round(min_y, -1)
        min_tick = min_y_rounded if min_y_rounded >= min_y else min_y_rounded + 10
        ticks = ([str(min_y)]
                 + [str(t) for t in range(int(min_tick), int(max_y), 10)]
                 + [str(max_y)])
        ax.set_xticks([float(t) for t in ticks])
        ax.set_xticklabels(ticks, rotation=90)
    elif tick_style == "month":
        ax.set_yticks(range(len(month_ticks)))
        ax.set_yticklabels(month_ticks)

    return _finish(ax, standalone, save_as, dpi)


def plot_duplicates(
    summary: pd.DataFrame,
    details: dict[str, dict[str, pd.DataFrame]],
    *,
    save_as_prefix: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes | None:
    """Plot spatio-temporal duplicates per parameter.

    Creates per-parameter diagnostic plots (scatter of duplicates,
    statistics and std distribution) and a summary bar chart.

    Args:
        summary: Summary DataFrame from
            :func:`comfort.database.find_spatiotemporal_duplicates`.
        details: Per-parameter duplicate tables from
            :func:`comfort.database.find_spatiotemporal_duplicates`.
        save_as_prefix: Path prefix for saved plots.
        dpi: Resolution (dots per inch) used for saving.

    Returns:
        The Axes of the summary bar chart, or ``None`` when there are no duplicates.
    """
    # Nothing to plot
    if summary.empty:
        return None

    for parameter, dups in details.items():
        dups_time_loc = dups["time_loc"]
        dups_row = dups["rows"]

        # Per-parameter diagnostic plots
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
        sns.scatterplot(ax=axs[0], data=dups_time_loc, x="ID", y="LEV_M", hue="size", size="size")
        axs[0].set_title(f"{parameter} - duplicates in time and location")
        sns.scatterplot(ax=axs[1], data=dups_row, x="ID", y="LEV_M", hue="size", size="size")
        axs[1].set_title(f"{parameter} - duplicates in all columns")
        plt.xticks(rotation=90)
        plt.tight_layout()
        if save_as_prefix:
            _save_fig(f"{save_as_prefix}_{parameter}_number_of_duplicates.png", dpi)
        plt.close()

        sns.scatterplot(data=dups_time_loc, x="size", y="mean", size="std", hue="std")
        plt.title(f"{parameter} - time/loc duplicate statistics")
        plt.tight_layout()
        if save_as_prefix:
            _save_fig(f"{save_as_prefix}_{parameter}_duplicates_statistics.png", dpi)
        plt.close()

        sns.boxplot(data=dups_time_loc, x="size", y="std")
        plt.title(f"{parameter} - time/loc duplicate std")
        plt.tight_layout()
        if save_as_prefix:
            _save_fig(f"{save_as_prefix}_{parameter}_duplicates_std.png", dpi)
        plt.close()

    # Summary bar chart comparing both duplicate types
    width = 0.35
    x = np.arange(len(summary))
    fig, ax = plt.subplots()
    b0 = ax.bar(x - width / 2, summary["proportion of row duplicates"], width, label="Row duplicates")
    b1 = ax.bar(x + width / 2, summary["proportion of time loc duplicates"], width, label="Time/location duplicates")
    ax.set_ylabel("%")
    ax.set_title("Proportion of duplicates per parameter")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["parameter"], rotation=90)
    ax.bar_label(b0, padding=3)
    ax.bar_label(b1, padding=3)
    ax.legend()
    plt.tight_layout()
    if save_as_prefix:
        _save_fig(f"{save_as_prefix}_duplicates_summary.png", dpi)

    return ax


def plot_missing_value_info(
    num_nulls: pd.DataFrame,
    *,
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes:
    """Bar chart of missing-value fractions per parameter.

    Args:
        num_nulls: DataFrame with columns ``'parameter'`` and ``'relative'``.
        ax: Axes to draw into.
        save_as: Save path.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib Axes.
    """
    ax, standalone = _get_or_create_ax(ax)

    # Bar chart
    num_nulls.plot(
        kind="bar", title="Fraction of missing values",
        x="parameter", y="relative", legend=False,
        ylabel="%", xlabel="", grid=True, ax=ax,
    )

    # Percentage labels on each bar
    for p in ax.patches:
        ax.annotate(str(round(p.get_height())),
                     (p.get_x() * 1.01, p.get_height() * 1.01))
    return _finish(ax, standalone, save_as, dpi)
