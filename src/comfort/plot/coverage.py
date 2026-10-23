"""Data-coverage and completeness diagnostics, i.e. how much data is there,
how complete is it (per parameter/year/month)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..database.information import get_names_of_all_parameter_tables, get_num_samples
from ..qc import build_where_clause
from ..util.sqlite_utils import validate_identifier
from ._helpers import _finish, _get_or_create_ax, _save_fig

if TYPE_CHECKING:
    import sqlite3
    import matplotlib.axes
    from ..qc import QCFilter

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
    for name, df in dfs.items():
        counts = count_samples_over_time(df, date_col, time_mode="year").set_index("YEAR")["COUNT"]
        counts = counts[counts > 0].sort_index()

        # Normalise
        if normalise:
            counts = counts / counts.max()

        # Plot
        label = name.capitalize() if name.isupper() else name
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
    param_cols: list[str],
    *,
    depth_col: str = "LEV_M",
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes:
    """Line plot of value count per depth level, for multiple parameters.

    Args:
        df_wide: Wide table with a depth column and one column per parameter.
        param_cols: Parameter column names to plot.
        depth_col: Column name containing depth values.
        ax: Axes to draw into.
        save_as: Save path.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib Axes.
    """
    ax, standalone = _get_or_create_ax(ax)

    # Get sorted depth levels
    depths = sorted(df_wide[depth_col].dropna().unique())

    # Plot count-vs-depth per parameter
    for col in param_cols:
        counts = [df_wide.loc[df_wide[depth_col] == d, col].notna().sum() for d in depths]
        label = col[2:] if col.startswith("P_") else col
        label = label.capitalize() if label.isupper() else label
        ax.plot(counts, depths, marker="o", label=label, markersize=4)

    # Labels and format
    ax.invert_yaxis()
    ax.set_xlabel("Number of non-NaN values")
    ax.set_ylabel("Depth [m]")
    ax.set_title("Data coverage per depth level")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return _finish(ax, standalone, save_as, dpi)


def count_samples_over_time(
    df: pd.DataFrame,
    date_column: str,
    time_mode: str = "year",
) -> pd.DataFrame:
    """Count the number of samples per year or month.

    Args:
        df: Must contain a date column.
        date_column: Column name containing the dates.
        time_mode: ``'year'`` or ``'month'``.

    Returns:
        DataFrame with columns ``[YEAR|MONTH, COUNT]``.
    """
    temp = df.copy()
    temp[date_column] = pd.to_datetime(temp[date_column])

    if time_mode == "year":
        # No dated rows -> No year range to count over
        if temp[date_column].dropna().empty:
            return pd.DataFrame(columns=["YEAR", "COUNT"])

        # Get min/max years
        min_y = temp[date_column].dt.year.min()
        max_y = temp[date_column].dt.year.max()

        # Count samples per year
        counts = (temp[date_column].dt.year.value_counts()
                  .reindex(np.arange(min_y, max_y + 1), fill_value=0)
                  .reset_index())
        counts.columns = ["YEAR", "COUNT"]
        return counts

    if time_mode == "month":
        # Count samples per month
        counts = (temp[date_column].dt.month.value_counts()
                  .reindex(np.arange(1, 13), fill_value=0)
                  .reset_index())
        counts.columns = ["MONTH", "COUNT"]
        return counts

    logger.error("count_samples_over_time: unsupported time_mode %r", time_mode)
    return pd.DataFrame(columns=np.array([time_mode.upper(), "COUNT"]))


def plot_counts_bar(
    df_count: pd.DataFrame,
    tick_style: str | None = "year",
    *,
    plot_title: str = "Number of samples per year",
    month_ticks: list[str] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes | None:
    """Horizontal bar chart with value labels for any category/count table.

    Used for per-year and per-month counts (with axis-tick formatting for
    those cases), as well as generic per-parameter counts.

    Args:
        df_count: First column is the category (year, month, parameter, ...),
            second is the count.
        tick_style: ``'year'``, ``'month'`` or ``None`` (no special tick
            formatting, e.g. for per-parameter counts).
        plot_title: Plot title.
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
    ax.set_title(plot_title)
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


def count_samples_over_time_from_db(
    conn: sqlite3.Connection,
    param_name: str,
    *,
    chunk_size: int = 1_000_000,
    date_column: str = "DATEANDTIME",
    time_mode: str = "year",
    quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Count samples per time unit from the database (chunk-wise).

    Args:
        conn: Database connection.
        param_name: Parameter name (without ``P_`` prefix).
        chunk_size: Rows to fetch per chunk.
        date_column: Date column name.
        time_mode: ``'year'`` or ``'month'``.
        quality_flags: QC filters.

    Returns:
        DataFrame with columns ``[YEAR|MONTH, COUNT]``.
    """
    table = f"P_{param_name}"
    validate_identifier(table)
    quality_statement = build_where_clause(quality_flags)

    # Chunk-wise fetch to bound memory and join station for DATEANDTIME
    cursor = conn.cursor()
    ex = cursor.execute(
        f"SELECT s.DATEANDTIME FROM {table} p "
        f"JOIN station s ON p.ID = s.ID {quality_statement};"
    )
    cols = np.array([desc[0].upper() for desc in cursor.description])
    df_count = pd.DataFrame(columns=np.array(["COUNT"]))

    # Fetch data chunk-wise
    while True:
        result = ex.fetchmany(chunk_size)
        if not result:
            break
        # Accumulate counts across chunks
        sample_count = count_samples_over_time(
            pd.DataFrame(result, columns=cols), date_column, time_mode,
        )
        sample_count = sample_count.set_index(time_mode.upper())
        df_count = df_count.reindex(
            set(list(sample_count.index) + list(df_count.index)), fill_value=0,
        )
        df_count = df_count.add(sample_count, axis=0, fill_value=0)

    df_count = df_count.reset_index()
    df_count.columns = [time_mode.upper(), "COUNT"]
    return df_count


def count_and_plot_samples_over_time(
    conn: sqlite3.Connection,
    param_name: str,
    *,
    chunk_size: int = 1_000_000,
    date_column: str = "DATEANDTIME",
    time_mode: str = "year",
    plot_title: str = "Number of samples per year",
    quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> pd.DataFrame:
    """Count samples per time unit from the database and plot.

    Convenience wrapper around :func:`count_samples_over_time_from_db` +
    :func:`plot_counts_bar`.

    Args:
        conn: Database connection.
        param_name: Parameter name (without ``P_`` prefix).
        chunk_size: Rows to fetch per chunk.
        date_column: Date column name.
        time_mode: ``'year'`` or ``'month'``.
        plot_title: Plot title.
        quality_flags: QC filters.
        ax: Axes to draw into.
        save_as: Save path.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        DataFrame of sample counts per time unit.
    """
    df_count = count_samples_over_time_from_db(
        conn, param_name,
        chunk_size=chunk_size, date_column=date_column,
        time_mode=time_mode, quality_flags=quality_flags,
    )
    plot_counts_bar(df_count, time_mode, plot_title=plot_title, ax=ax, save_as=save_as, dpi=dpi)
    return df_count


def count_samples_per_parameter(
    conn: sqlite3.Connection,
    param_names: list[str] | None = None,
    *,
    quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Count the total number of samples per parameter.

    Args:
        conn: Database connection.
        param_names: Parameter names or ``None`` (all parameters).
        quality_flags: QC filters.

    Returns:
        DataFrame with columns ``[parameter, count]``.
    """
    # Default to all parameter tables in the database
    if not param_names:
        param_names = [x[2:] for x in get_names_of_all_parameter_tables(conn)]

    # Count samples per parameter via SQL COUNT(*)
    df_count = pd.DataFrame({"parameter": param_names, "count": 0})
    for i, name in enumerate(param_names):
        df_count.loc[i, "count"] = get_num_samples(
            conn, f"P_{name}", table_type="table", quality_flags=quality_flags,
        )
    return df_count


def count_and_plot_samples_per_parameter(
    conn: sqlite3.Connection,
    param_names: list[str] | None,
    *,
    quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
    plot_title: str = "Number of samples per parameter",
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> pd.DataFrame:
    """Count samples per parameter and plot as a bar chart.

    Convenience wrapper around :func:`count_samples_per_parameter` +
    :func:`plot_counts_bar`.

    Args:
        conn: Database connection.
        param_names: Parameter names or ``None`` (all parameters).
        quality_flags: QC filters.
        plot_title: Plot title.
        ax: Axes to draw into.
        save_as: Save path.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        DataFrame of sample counts per parameter.
    """
    df_count = count_samples_per_parameter(conn, param_names, quality_flags=quality_flags)
    plot_counts_bar(
        df_count.sort_values("count"), tick_style=None,
        plot_title=plot_title, ax=ax, save_as=save_as, dpi=dpi,
    )
    return df_count


def count_negative_samples(
    conn: sqlite3.Connection,
    param_names: list[str],
    *,
    quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Count the proportion of negative values per parameter.

    Args:
        conn: Database connection.
        param_names: Parameter names (without ``P_`` prefix).
        quality_flags: QC filters.

    Returns:
        DataFrame with columns ``[parameter, number of samples,
        number of negative samples, proportion of negative samples]``.
    """
    cursor = conn.cursor()
    quality_statement = build_where_clause(quality_flags, keyword="AND")

    # Query negative and total counts for each parameter
    rows = []
    for param_name in param_names:
        table = f"P_{param_name}"
        validate_identifier(table)
        logger.info("count_negative_samples: %s", param_name)
        negative_count = cursor.execute(
            f"SELECT COUNT(*) FROM {table} WHERE VAL<0 {quality_statement};"
        ).fetchone()[0]
        count = cursor.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
        rows.append({
            "parameter": param_name,
            "number of samples": count,
            "number of negative samples": negative_count,
        })

    # Compute proportions and sort for display
    df = pd.DataFrame(rows, columns=["parameter", "number of samples", "number of negative samples"])
    df["proportion of negative samples"] = df["number of negative samples"] / df["number of samples"] * 100

    return df.sort_values("proportion of negative samples")


def count_and_plot_negative_samples(
    conn: sqlite3.Connection,
    param_names: list[str],
    *,
    quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
    plot_title: str = "Proportion of negative samples per parameter [%]",
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> pd.DataFrame:
    """Count and plot the proportion of negative values per parameter.

    Convenience wrapper around :func:`count_negative_samples` +
    :func:`plot_counts_bar`.

    Args:
        conn: Database connection.
        param_names: Parameter names (without ``P_`` prefix).
        quality_flags: QC filters.
        plot_title: Plot title.
        ax: Axes to draw into.
        save_as: Save path.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        DataFrame with negative-sample statistics.
    """
    df = count_negative_samples(conn, param_names, quality_flags=quality_flags)

    df_pct = df[["parameter", "proportion of negative samples"]].round(2)
    plot_counts_bar(df_pct, tick_style=None, plot_title=plot_title, ax=ax, save_as=save_as, dpi=dpi)
    return df


def detect_and_plot_spatiotemporal_duplicates(
    conn: sqlite3.Connection,
    param_names: list[str],
    *,
    quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
    save_as_prefix: str | None = None,
    dpi: int = 300,
) -> pd.DataFrame:
    """Investigate spatiotemporal duplicates per parameter.

    Creates per-parameter diagnostic plots (scatter of duplicates,
    statistics and std distribution). Plots per parameter while iterating,
    so full parameter tables don't need to be held in memory.

    Args:
        conn: Database connection.
        param_names: Parameter names (without ``P_`` prefix).
        quality_flags: QC filters.
        save_as_prefix: Path prefix for saved plots.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        Summary DataFrame with columns ``[parameter, number of samples, number of row duplicates,
        number of time loc duplicates, proportion of row duplicates, proportion of time loc duplicates]``.
    """
    cursor = conn.cursor()
    quality_statement = build_where_clause(quality_flags)

    rows = []
    for param_name in param_names:
        table = f"P_{param_name}"
        validate_identifier(table)

        # Fetch all rows for this parameter
        logger.info("detect_and_plot_spatiotemporal_duplicates: %s", param_name)
        ex = cursor.execute(f"SELECT * FROM {table} {quality_statement};")
        df_param = pd.DataFrame(ex.fetchall(), columns=np.array([x[0] for x in cursor.description]))

        # Spatiotemporal duplicates (same station and depth)
        dups_time_loc = (
            df_param.groupby(["ID", "LEV_M"], as_index=False, dropna=False)
            .agg(size=("VAL", "size"), mean=("VAL", "mean"), std=("VAL", "std"))
        )
        dups_time_loc = dups_time_loc[dups_time_loc["size"] > 1]

        if len(dups_time_loc) < 1:
            logger.info("  %s: no duplicates", param_name)
            continue

        # Exact row duplicates (identical across all columns)
        dups_row = (
            df_param.groupby(df_param.columns.tolist(), as_index=False, dropna=False)
            .size()
        )
        dups_row = dups_row[dups_row["size"] > 1]

        rows.append({
            "parameter": param_name,
            "number of samples": len(df_param),
            "number of row duplicates": len(dups_row),
            "number of time loc duplicates": len(dups_time_loc),
        })

        # Per-parameter diagnostic plots
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
        sns.scatterplot(ax=axs[0], data=dups_time_loc, x="ID", y="LEV_M",
                        hue="size", size="size")
        axs[0].set_title(f"{param_name} - duplicates in time and location")
        sns.scatterplot(ax=axs[1], data=dups_row, x="ID", y="LEV_M",
                        hue="size", size="size")
        axs[1].set_title(f"{param_name} - duplicates in all columns")
        plt.xticks(rotation=90)
        plt.tight_layout()
        if save_as_prefix:
            _save_fig(f"{save_as_prefix}_{param_name}_number_of_duplicates.png", dpi)
        plt.close()

        sns.scatterplot(data=dups_time_loc, x="size", y="mean", size="std", hue="std")
        plt.title(f"{param_name} - time/loc duplicate statistics")
        plt.tight_layout()
        if save_as_prefix:
            _save_fig(f"{save_as_prefix}_{param_name}_duplicates_statistics.png", dpi)
        plt.close()

        sns.boxplot(data=dups_time_loc, x="size", y="std")
        plt.title(f"{param_name} - time/loc duplicate std")
        plt.tight_layout()
        if save_as_prefix:
            _save_fig(f"{save_as_prefix}_{param_name}_duplicates_std.png", dpi)
        plt.close()

    if not rows:
        # Return empty dataframe
        return pd.DataFrame(columns=[
            "parameter", "number of samples", "number of row duplicates", "number of time loc duplicates",
            "proportion of row duplicates", "proportion of time loc duplicates",
        ])

    # Compute duplicate proportions
    df = pd.DataFrame(rows)
    df["proportion of row duplicates"] = (
        df["number of row duplicates"] / df["number of samples"] * 100
    )
    df["proportion of time loc duplicates"] = (
        df["number of time loc duplicates"] / df["number of samples"] * 100
    )

    # Summary bar chart comparing both duplicate types
    width = 0.35
    x = np.arange(len(df))
    fig, ax = plt.subplots()
    b0 = ax.bar(x - width / 2, df["proportion of row duplicates"],
                width, label="Row duplicates")
    b1 = ax.bar(x + width / 2, df["proportion of time loc duplicates"],
                width, label="Time/location duplicates")
    ax.set_ylabel("%")
    ax.set_title("Proportion of duplicates per parameter")
    ax.set_xticks(x)
    ax.set_xticklabels(df["parameter"], rotation=90)
    ax.bar_label(b0, padding=3)
    ax.bar_label(b1, padding=3)
    ax.legend()
    plt.tight_layout()
    if save_as_prefix:
        _save_fig(f"{save_as_prefix}_duplicates_summary.png", dpi)

    return df


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
