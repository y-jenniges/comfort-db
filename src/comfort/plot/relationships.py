"""Bivariate relationship plots, i.e. how do two variables relate to each other.

The T-S diagram with density contours is adapted from:

    Yvonne Jenniges (2025).
    y-jenniges/ocean_clustering_and_validation:
    Biogeochemical Ocean Regions - Code Base (v1.0.1).
    Zenodo. https://doi.org/10.5281/zenodo.15827777
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import gsw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..database.information import param_label, param_table_name
from ..qc import build_where_clause
from ..util.sqlite_utils import validate_identifier
from ..util.columns import find_col
from ._helpers import _finish, _get_or_create_ax, _pretty_label, _save_fig

if TYPE_CHECKING:
    import sqlite3
    import matplotlib.axes
    from ..qc import QCFilter

logger = logging.getLogger(__name__)


def plot_correlation(
    df: pd.DataFrame | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
    *,
    colour_col: str | None = None,
    cmap: str = "viridis",
    kind: str = "scatter",
    gridsize: int = 40,
    regression: bool = True,
    annotate_r2: bool = True,
    marker_size: float = 8,
    alpha: float = 0.4,
    xlabel: str | None = None,
    ylabel: str | None = None,
    conn: sqlite3.Connection | None = None,
    parameters: tuple[str, str] | None = None,
    quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes | None:
    """Scatter plot of two variables with optional regression line and r2.

    Accepts either a *df* with two named columns, or a database *conn* with
    a *(param_x, param_y)* tuple to join two parameter tables.

    Args:
        df: DataFrame containing *x_col* and *y_col*.
        x_col: Column name for the x-axis.
        y_col: Column name for the y-axis.
        colour_col: Optional column for colour coding.
        cmap: Matplotlib colormap (used when *colour_col* is set).
        kind: ``"scatter"`` (default) or ``"hexbin"``.
        gridsize: Number of hexagons across the x-axis when ``kind="hexbin"``.
        regression: Overlay a linear regression line.
        annotate_r2: Annotate the plot with the r2 value.
        marker_size: Scatter marker size.
        alpha: Marker opacity.
        xlabel: x-axis label (defaults to *x_col*).
        ylabel: y-axis label (defaults to *y_col*).
        conn: Database connection (alternative to *df*).
        parameters: ``(param_x, param_y)`` names. Insensitive to casing and ``P_`` prefix.
        quality_flags: QC filters forwarded to the SQL query.
        ax: Axes to draw into.
        save_as: File path to save the figure.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib axes or ``None`` when the data is empty.
    """
    # Load from database when no DataFrame is given
    if df is None:
        df, x_col, y_col = _load_pair(conn, parameters, quality_flags)

    if x_col is None or y_col is None:
        raise ValueError("x_col and y_col are required")

    # Resolve column names (insensitive to casing)
    x_col = find_col(df, x_col)
    y_col = find_col(df, y_col)
    colour_col = find_col(df, colour_col, required=False) or colour_col

    # Drop rows with missing values
    valid = df[[x_col, y_col]].dropna()
    if colour_col and colour_col in df.columns:
        # Ensure colour_col is present
        valid = df[[x_col, y_col, colour_col]].dropna()
    if valid.empty:
        logger.warning("plot_correlation: No valid data after dropping NaNs")
        return None

    ax, standalone = _get_or_create_ax(ax)

    if kind == "hexbin":
        # Hexbin plot
        has_color = colour_col and colour_col in valid.columns
        hb = ax.hexbin(
            valid[x_col], valid[y_col],
            C=valid[colour_col] if has_color else None,
            reduce_C_function=np.mean, gridsize=gridsize, cmap=cmap, mincnt=1,
        )

        # Colorbar
        ax.figure.colorbar(hb, ax=ax, label=_pretty_label(colour_col) if has_color else "Count")
    else:
        # Scatter plot with optional colour coding
        scatter_kw = dict(s=marker_size, alpha=alpha, edgecolors="none")
        if colour_col and colour_col in valid.columns:
            sc = ax.scatter(valid[x_col], valid[y_col], c=valid[colour_col],
                            cmap=cmap, **scatter_kw)
            ax.figure.colorbar(sc, ax=ax, label=_pretty_label(colour_col))
        else:
            ax.scatter(valid[x_col], valid[y_col], **scatter_kw)

    # Linear regression and coefficient of determination (R2)
    if regression or annotate_r2:
        x = valid[x_col].values.astype(float)
        y = valid[y_col].values.astype(float)
        coeffs = np.polyfit(x, y, 1)
        r2 = np.corrcoef(x, y)[0, 1] ** 2

        if regression:
            # Draw regression line
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, np.polyval(coeffs, x_line), color="red",
                    linewidth=1.5, label=f"y = {coeffs[0]:.3g}x + {coeffs[1]:.3g}")
        if annotate_r2:
            # Add R2 annotation
            ax.annotate(
                f"$r^2$ = {r2:.3f}",
                xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    # Axes labels
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)

    # Legend
    if regression:
        ax.legend(loc="lower right")

    return _finish(ax, standalone, save_as, dpi)


def plot_joint(
    df: pd.DataFrame | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
    *,
    kind: str = "hist",
    conn: sqlite3.Connection | None = None,
    parameters: tuple[str, str] | None = None,
    quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> sns.JointGrid | None:
    """Joint plot of two variables (scatter with marginal distributions).

    Accepts either a *df* or a database *conn* with *(param_x, param_y)*.

    Args:
        df: DataFrame containing *x_col* and *y_col*.
        x_col: Column for x-axis.
        y_col: Column for y-axis.
        kind: Joint-plot kind (``'scatter'``, ``'hist'``, ``'kde'``, ``'hex'``, ``'reg'``).
        conn: Database connection (alternative to *df*).
        parameters: ``(param_x, param_y)`` names. Insensitive to casing and ``P_`` prefix.
        quality_flags: QC filters.
        save_as: File path to save the figure.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        :class:`seaborn.JointGrid`, or ``None`` when the data is empty.
    """
    # Load from database when no DataFrame is given
    if df is None:
        df, x_col, y_col = _load_pair(conn, parameters, quality_flags)

    if x_col is None or y_col is None:
        raise ValueError("x_col and y_col are required")

    # Resolve column names (insensitive to casing)
    x_col = find_col(df, x_col)
    y_col = find_col(df, y_col)

    # Drop rows with missing values
    valid = df[[x_col, y_col]].dropna()
    if valid.empty:
        logger.warning("plot_joint: no valid data after dropping NaNs")
        return None

    # Joint plot
    g = sns.jointplot(x=x_col, y=y_col, data=valid, kind=kind)
    plt.tight_layout()
    _save_fig(save_as, dpi)
    return g


def _load_pair(conn, parameters, quality_flags):
    """Join two parameter tables and return (df, x_col, y_col)."""
    if conn is None or parameters is None or len(parameters) != 2:
        raise ValueError(
            "Provide either a DataFrame (df) or conn + parameters=(param_x, param_y)"
        )
    x_name, y_name = (param_label(p) for p in parameters)

    # Validate table identifiers
    x_table, y_table = param_table_name(x_name), param_table_name(y_name)
    validate_identifier(x_table)
    validate_identifier(y_table)

    # Build quality-flag clauses for both tables
    where = build_where_clause(quality_flags, table_alias="p")
    and_clause = ""
    if quality_flags:
        and_clause = " " + build_where_clause(quality_flags, table_alias="q", keyword="AND")

    # Join the two parameter tables on station ID and depth
    sql = (
        f"SELECT p.VAL, q.VAL FROM {x_table} AS p "
        f"LEFT JOIN {y_table} AS q USING(ID, LEV_M) "
        f"{where}{and_clause};"
    )
    cur = conn.cursor()
    rows = cur.execute(sql).fetchall()
    df = pd.DataFrame(rows, columns=np.array([x_name, y_name]))
    return df, x_name, y_name


_SAL_LABELS: dict[str, str] = {
    "SA": r"$S_A$ [g kg$^{-1}$]",
    "SP": r"$S_P$ [1]",
}
_TEMP_LABELS: dict[str, str] = {
    "CT": r"$\Theta$ [°C]",
    "pt": r"$\theta$ [°C]",
    "pt0": r"$\theta$ [°C]",
    "t": r"$T$ [°C]",
}
_TEOS10_SAL = {"SA"}
_TEOS10_TEMP = {"CT"}


def plot_ts_diagram(
    df: pd.DataFrame,
    temperature_col: str,
    salinity_col: str,
    *,
    temperature_type: str = "CT",
    salinity_type: str = "SA",
    colour_col: str | None = None,
    depth_col: str = "LEV_M",
    cmap: str = "viridis_r",
    colourbar_label: str | None = None,
    density_contours: bool = True,
    contour_levels: int | list[float] | None = None,
    kind: str = "scatter",
    gridsize: int = 40,
    marker_size: float = 9,
    alpha: float = 0.6,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes | None:
    """Temperature-Salinity diagram with optional isopycnal contours.
    Density contours are only accurate when ``salinity_type="SA"`` and ``temperature_type="CT"``.

    Axis labels are derived from *salinity_type* / *temperature_type* when *xlabel* /
    *ylabel* are not supplied explicitly:

    * ``salinity_type="SA"``  -> :math:`S_A` [g kg-1]  - Absolute Salinity (TEOS-10)
    * ``salinity_type="SP"``  -> :math:`S_P` [1]  - Practical Salinity (dimensionless)
    * ``temperature_type="CT"`` -> :math:`\\Theta` [°C] - Conservative Temperature (TEOS-10)
    * ``temperature_type="pt"`` / ``"pt0"`` -> :math:`\\theta` [°C] - Potential Temperature
    * ``temperature_type="t"``  -> :math:`T` [°C] - In-situ Temperature

    Args:
        df: DataFrame containing temperature and salinity.
        temperature_col: Column with temperature values.
        salinity_col: Column with salinity values.
        temperature_type: Variable type for *temperature_col* - ``"CT"``, ``"pt"``,
            ``"pt0"``, or ``"t"``.  Controls the y-axis label and
            density contour accuracy.
        salinity_type: Variable type for *salinity_col* - ``"SA"`` or ``"SP"``.
            Controls the x-axis label and density contour accuracy.
        colour_col: Column to colour the scatter by. Falls back to
            *depth_col* when ``None``.
        depth_col: Depth column used as default colour variable.
        cmap: Matplotlib colormap name.
        colourbar_label: Colour-bar label (defaults to the colour column).
        density_contours: Overlay sigma0 contour lines.
        contour_levels: Explicit contour levels or number of levels.
        kind: ``"scatter"`` (default) or ``"hexbin"``.
        gridsize: Number of hexagons across the x-axis when ``kind="hexbin"``.
        marker_size: Scatter marker size.
        alpha: Scatter marker opacity.
        xlabel: x-axis label (defaults to label derived from *salinity_type*).
        ylabel: y-axis label (defaults to label derived from *temperature_type*).
        xlim: Salinity axis limits.
        ylim: Temperature axis limits.
        ax: Axes to draw into.
        save_as: File path to save the figure.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib Axes, or ``None`` when the data is empty.
    """
    # Resolve column names (insensitive to casing)
    temperature_col = find_col(df, temperature_col)
    salinity_col = find_col(df, salinity_col)
    colour_col = find_col(df, colour_col, required=False) or colour_col
    depth_col = find_col(df, depth_col, required=False) or depth_col

    # Determine colour column
    if colour_col is not None:
        # Use passed colour_col
        use_color = colour_col
    elif kind != "hexbin":
        # Default (for scatter) is depth observation count
        use_color = depth_col
    else:
        # Default (for hexbin) is observation count
        use_color = None

    # Define which columns need full data
    needed = [temperature_col, salinity_col]
    if use_color in df.columns:
        needed.append(use_color)

    # Extract valid data
    valid = df[needed].dropna()
    if valid.empty:
        logger.warning("plot_ts_diagram: No valid data after dropping NaNs.")
        return None

    # Warning if density contours need different sal/temp to be accurate
    if density_contours and (salinity_type not in _TEOS10_SAL or temperature_type not in _TEOS10_TEMP):
        logger.warning(
            "Density contours use gsw.rho(SA, CT, 0) but salinity_type=%r, temperature_type=%r - "
            "contours will be inaccurate. Pass SA and CT for correct isopycnals.",
            salinity_type, temperature_type,
        )

    ax, standalone = _get_or_create_ax(ax)

    # Isopycnal contour grid
    if density_contours:
        s_min = xlim[0] if xlim else valid[salinity_col].min()
        s_max = xlim[1] if xlim else valid[salinity_col].max()
        t_min = ylim[0] if ylim else valid[temperature_col].min()
        t_max = ylim[1] if ylim else valid[temperature_col].max()

        s_pad = 0.01 * abs(s_max - s_min) or 0.5
        t_pad = 0.1 * abs(t_max - t_min) or 0.5
        s_min -= s_pad
        s_max += s_pad
        t_min -= t_pad
        t_max += t_pad

        # Build an S-T grid and compute sigma-0 at each node
        si = np.linspace(s_min, s_max, max(int(round((s_max - s_min) / 0.1)) + 1, 10))
        ti = np.linspace(t_min, t_max, max(int(round((t_max - t_min) / 0.1)) + 1, 10))
        s_grid, t_grid = np.meshgrid(si, ti)
        sigma0 = gsw.rho(s_grid, t_grid, 0) - 1000

        # Overlay dashed isopycnal lines
        levels = contour_levels if contour_levels is not None else 12
        cs = ax.contour(si, ti, sigma0, levels=levels, linestyles="dashed", colors="k", alpha=0.5)

        # Place each density label at the midpoint of its line's longest segment
        manual_positions = []
        for seg_list in cs.allsegs:
            segs = [seg for seg in seg_list if len(seg) > 0]
            if not segs:
                continue
            longest = max(segs, key=len)
            manual_positions.append(tuple(longest[len(longest) // 2]))
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f", manual=manual_positions)

    # Plot observations on top of the contours
    has_color_data = use_color in valid.columns
    if kind == "hexbin":
        hb = ax.hexbin(valid[salinity_col], valid[temperature_col], C=valid[use_color] if has_color_data else None, reduce_C_function=np.mean, gridsize=gridsize, cmap=cmap, mincnt=1)
        ax.figure.colorbar(hb, ax=ax, label=(colourbar_label or _pretty_label(use_color)) if has_color_data else "Count")
    else:
        scatter_kw = dict(s=marker_size, alpha=alpha, marker=".", edgecolors="none")
        if has_color_data:
            sc = ax.scatter(valid[salinity_col], valid[temperature_col], c=valid[use_color], cmap=cmap, **scatter_kw)
            ax.figure.colorbar(sc, ax=ax, label=colourbar_label or _pretty_label(use_color))
        else:
            ax.scatter(valid[salinity_col], valid[temperature_col], **scatter_kw)

    # Axes labels and limits
    ax.set_xlabel(xlabel or _SAL_LABELS.get(salinity_type, salinity_col))
    ax.set_ylabel(ylabel or _TEMP_LABELS.get(temperature_type, temperature_col))
    ax.set_title("T-S diagram")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    return _finish(ax, standalone, save_as, dpi)
