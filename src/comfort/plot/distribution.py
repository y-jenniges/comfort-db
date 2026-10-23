"""Distribution plots - histograms, boxplots and group comparison.

The ``compare_stats`` function is adapted from:

    Yvonne Jenniges (2025).
    y-jenniges/ocean_clustering_and_validation:
    Biogeochemical Ocean Regions - Code Base (v1.0.1).
    Zenodo. https://doi.org/10.5281/zenodo.15827777
"""
from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ._helpers import _finish, _get_or_create_ax, _resolve_data, _save_fig

if TYPE_CHECKING:
    import sqlite3
    import matplotlib.axes
    import matplotlib.figure
    from ..qc import QCFilter

logger = logging.getLogger(__name__)


def boxplot(
    df: pd.DataFrame | None = None,
    value_col: str = "VAL",
    *,
    kind: str = "box",
    title: str | None = None,
    conn: sqlite3.Connection | None = None,
    parameter: str | None = None,
    quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes:
    """Boxplot or violin plot for a single column.

    Accepts either a *df* or a database *conn* along with a *parameter* name.

    Args:
        df: DataFrame with the value column.
        value_col: Column to plot.
        kind: Plot type (``'box'`` or ``'violin'``).
        title: Plot title (defaults to *value_col*).
        conn: Database connection (alternative to *df*).
        parameter: Parameter name without ``P_`` prefix.
        quality_flags: QC filters.
        ax: Axes to draw into.
        save_as: File path to save the figure.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib Axes.
    """
    df = _resolve_data(df, conn, parameter, quality_flags)
    ax, standalone = _get_or_create_ax(ax)

    # Draw the chosen distribution plot
    if kind == "violin":
        sns.violinplot(x=value_col, data=df, ax=ax)
    else:
        sns.boxplot(x=value_col, data=df, ax=ax)
    ax.set_title(title or value_col)
    ax.set_xlabel("")

    return _finish(ax, standalone, save_as, dpi)


def plot_histogram(
    df: pd.DataFrame | None = None,
    value_col: str = "VAL",
    *,
    title: str | None = None,
    kind: str = "hist",
    conn: sqlite3.Connection | None = None,
    parameter: str | None = None,
    quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes:
    """Histogram for a single value column.

    Accepts either a *df* or a database *conn* along with a *parameter* name.

    Args:
        df: DataFrame with the value column.
        value_col: Column to plot.
        title: Plot title.
        kind: Plot kind accepted by ``DataFrame.plot()``.
        conn: Database connection (alternative to *df*).
        parameter: Parameter name without ``P_`` prefix.
        quality_flags: QC filters.
        ax: Axes to draw into.
        save_as: File path to save the figure.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib Axes.
    """
    df = _resolve_data(df, conn, parameter, quality_flags)
    ax, standalone = _get_or_create_ax(ax)

    df[[value_col]].plot(
        kind=kind, title=title or f"Number of {value_col} samples",
        legend=False, ax=ax,
    )

    return _finish(ax, standalone, save_as, dpi)


def plot_monthly_histogram(
    df: pd.DataFrame | None = None,
    value_col: str = "VAL",
    *,
    date_col: str = "DATEANDTIME",
    conn: sqlite3.Connection | None = None,
    parameter: str | None = None,
    quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.figure.Figure:
    """Twelve-panel histogram (one subplot per month).

    Accepts either a *df* or a database *conn* along with a *parameter* name.

    Args:
        df: DataFrame with value and date columns.
        value_col: Column with measured values.
        date_col: Column with datetime values.
        conn: Database connection (alternative to *df*).
        parameter: Parameter name without ``P_`` prefix.
        quality_flags: QC filters.
        save_as: File path to save the figure.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib Figure.
    """
    df = _resolve_data(df, conn, parameter, quality_flags)
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col])

    # Create a 3x4 grid of subplots, one per calendar month
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))
    month = 0
    for i in range(3):
        for j in range(4):
            month += 1
            data = temp[temp[date_col].dt.month == month]
            month_name = datetime.datetime.strptime(str(month), "%m").strftime("%B")
            sns.histplot(data=data, x=value_col, kde=True, color="teal", ax=axs[i, j])
            axs[i, j].set_title(month_name)
            axs[i, j].set_xlabel(None)
            axs[i, j].set_ylabel(None)

    fig.supylabel("Count")
    fig.supxlabel(value_col)
    plt.tight_layout()
    _save_fig(save_as, dpi)
    return fig


def compare_stats(
    df: pd.DataFrame,
    group_col: str,
    value_cols: list[str] | None = None,
    *,
    labels: list | None = None,
    value_labels: dict[str, str] | None = None,
    scale: bool = True,
    figsize: tuple[float, float] = (8, 6),
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes:
    """Comparative boxplots of multiple parameters across groups.

    Useful for comparing parameter distributions across clusters, water
    masses or regions. Values are MinMax-scaled by default so that
    parameters with different units are visually comparable.

    Adapted from Jenniges (2025), doi:10.5281/zenodo.15827777.

    Args:
        df: DataFrame containing *group_col* and all *value_cols*.
        group_col: Column that identifies the group/cluster/region.
        value_cols: Columns to compare. ``None`` auto-detects numeric
            columns (excluding *group_col*).
        labels: Subset of group values to include. ``None`` uses all.
        value_labels: Rename mapping for display (e.g.
            ``{"P_TEMPERATURE": "Temperature"}``).
        scale: MinMax-scale values so different parameters are comparable.
        figsize: Figure size (used only when no *ax* is given).
        ax: Axes to draw into.
        save_as: File path to save the figure.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib Axes.
    """
    # Auto-detect numeric columns when none are specified
    if value_cols is None:
        value_cols = sorted(
            c for c in df.select_dtypes("number").columns if c != group_col
        )

    # Filter to the requested group labels
    temp = df.copy()
    if labels is not None:
        temp = temp[temp[group_col].isin(labels)]
    else:
        labels = sorted(temp[group_col].unique())

    # Scale to [0, 1] for cross-parameter comparability
    if scale:
        try:
            from sklearn.preprocessing import MinMaxScaler
        except ImportError:
            raise ImportError(
                "scikit-learn is required for compare_stats(scale=True). "
                'Install it with: pip install "comfort-db[scale]"'
            )
        scaler = MinMaxScaler().fit(temp[value_cols])
        scaled = pd.DataFrame(
            scaler.transform(temp[value_cols]),
            columns=value_cols, index=temp.index,
        )
        scaled[group_col] = temp[group_col].values
        temp = scaled

    # Reshape from wide to long format for seaborn
    melted = pd.melt(temp, id_vars=[group_col], value_vars=value_cols)
    if value_labels:
        melted["variable"] = melted["variable"].map(
            lambda v: value_labels.get(v, v)
        )

    melted = melted.rename(columns={group_col: "Group"})

    ax, standalone = _get_or_create_ax(ax, figsize=figsize)

    # Grouped boxplot with one hue per group label
    bp = sns.boxplot(
        data=melted, x="variable", y="value", hue="Group",
        hue_order=labels, flierprops={"marker": "."},
        ax=ax,
    )
    sns.move_legend(bp, "lower left")
    ax.set_xlabel("")
    ax.set_ylabel("Scaled value" if scale else "Value")

    return _finish(ax, standalone, save_as, dpi)
