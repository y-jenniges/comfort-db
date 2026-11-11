"""Spatial visualisation, i.e. where the data is and where it is missing."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..util.columns import find_col, resolve_param_columns
from ._helpers import _finish, _get_or_create_ax, _require_cartopy, _resolve_data

if TYPE_CHECKING:
    import sqlite3
    import matplotlib.axes
    from ..qc import QualityFlags

logger = logging.getLogger(__name__)


def plot_spatial_distribution(
    df: pd.DataFrame | None = None,
    *,
    lon_col: str = "LONGITUDE",
    lat_col: str = "LATITUDE",
    resolution: float = 0.5,
    clim: tuple[float, float] | None = None,
    cmap: str = "Spectral_r",
    colourbar_label: str | None = None,
    conn: sqlite3.Connection | None = None,
    parameter: str | None = None,
    quality_flags: QualityFlags | None = None,
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes:
    """Count observations per lon/lat bin and display on a world map.

    Accepts either a *df* (must contain *lon_col* and *lat_col*) or a
    database *conn* along with a *parameter* name.

    Args:
        df: DataFrame with latitude and longitude columns.
        lon_col: Longitude column name.
        lat_col: Latitude column name.
        resolution: Bin size in degrees.
        clim: Colour scale limits.
        cmap: Matplotlib colormap name.
        colourbar_label: Colour-bar label.
        conn: Database connection (alternative to *df*).
        parameter: Parameter name. Insensitive to casing and ``P_`` prefix.
        quality_flags: QC filters.
        ax: Axes to draw into. Must already use a cartopy projection when
            passed explicitly. Creates a new PlateCarree figure when ``None``.
        save_as: Save path.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib Axes.
    """
    ccrs, cfeature = _require_cartopy()

    # Coordinates live in the station table, so read through the joined reader
    df = _resolve_data(df, conn, parameter, quality_flags)

    # Resolve column names (insensitive to casing)
    lon_col = find_col(df, lon_col)
    lat_col = find_col(df, lat_col)

    # Bin observations into a 2D lat/lon grid
    lonbins = np.arange(-180, 180, resolution)
    latbins = np.arange(-90, 90, resolution)
    hist = stats.binned_statistic_2d(
        df[lat_col], df[lon_col], None,
        bins=[latbins, lonbins], statistic="count",
    )

    # Mask empty bins so they appear transparent on the map
    hist.statistic[hist.statistic == 0] = np.nan

    # Plot on a PlateCarree projection
    ax, standalone = _get_or_create_ax(ax, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="grey")
    ax.gridlines(draw_labels=True)
    image = ax.pcolormesh(
        lonbins, latbins, hist.statistic,
        cmap=plt.colormaps[cmap].resampled(64), shading="flat",
        transform=ccrs.PlateCarree(),
    )
    ax.figure.colorbar(
        image, ax=ax, orientation="horizontal", fraction=0.1, aspect=40,
        pad=0.08, label=colourbar_label or f"Number of {parameter or ''} samples",
    )
    if clim:
        image.set_clim(*clim)

    return _finish(ax, standalone, save_as, dpi)


def plot_lon_lat_range(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    *,
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes:
    """Draw a bounding box on a world map.

    Args:
        lon_min: Western longitude boundary.
        lon_max: Eastern longitude boundary.
        lat_min: Southern latitude boundary.
        lat_max: Northern latitude boundary.
        ax: Axes to draw into.
        save_as: Save path.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib Axes.
    """
    ccrs, _ = _require_cartopy()
    ax, standalone = _get_or_create_ax(ax, projection=ccrs.PlateCarree())

    ax.coastlines()
    ax.set_global()
    ax.gridlines(draw_labels=True)

    # Draw horizontal and vertical edges of the bounding box
    for x1, x2, y in [(lon_min, lon_max, lat_min), (lon_min, lon_max, lat_max)]:
        ax.plot([x1, x2], [y, y], color="blue", linewidth=2)
    for x, y1, y2 in [(lon_min, lat_min, lat_max), (lon_max, lat_min, lat_max)]:
        ax.plot([x, x], [y1, y2], color="blue", linewidth=2)

    return _finish(ax, standalone, save_as, dpi)


def plot_parameter_map(
    df_wide: pd.DataFrame,
    parameters: list[str],
    *,
    depths: list | None = None,
    times: list | None = None,
    savefig_folder: str | None = None,
    dpi: int = 300,
) -> list[matplotlib.axes.Axes]:
    """Geographic scatter map(s) of parameter values by depth and time slice.
    Missing cells are rendered as translucent black.

    Args:
        df_wide: Wide table with ``LATITUDE``, ``LONGITUDE``, ``LEV_M`` and one column per parameter.
            ``DATEANDTIME`` is optional: Without it (a grid without time axis) all rows form one time slice.
        parameters: Parameter column names to plot.
        depths: Depth levels to show (``None`` = all).
        times: Time values to show (``None`` = all). Requires a ``DATEANDTIME`` column.
        savefig_folder: Folder to save figures. If given, each figure is closed after saving.
        dpi: Resolution (dots per inch) used when saving.
    Returns:
        list[matplotlib.axes.Axes]: One Axes per parameter x time x depth combination.
    """
    ccrs, _ = _require_cartopy()
    df_wide = df_wide.copy()

    # Default to all available depth levels and time steps
    depths = depths or list(df_wide["LEV_M"].value_counts().index)
    has_time = "DATEANDTIME" in df_wide.columns
    if has_time:
        df_wide["DATEANDTIME"] = pd.to_datetime(df_wide["DATEANDTIME"])
        times = times or list(df_wide["DATEANDTIME"].value_counts().index)
    elif times:
        raise ValueError("times requires a DATEANDTIME column")
    else:
        # A single slice that holds all rows
        times = [None]

    # NaN bins rendered as translucent black
    cmap = plt.colormaps["viridis"].with_extremes(bad=mcolors.to_rgba("black", alpha=0.2))

    # One map per parameter x time x depth combination
    parameters = resolve_param_columns(df_wide, parameters)
    axes = []
    for param in parameters:
        # Shared colour norm for all times and depths of a parameter
        norm = plt.Normalize(df_wide[param].min(), df_wide[param].max())
        for t in times:
            for depth in depths:
                # Filter df for depth and time
                dft = df_wide[df_wide["LEV_M"] == depth]
                if has_time:
                    dft = dft[dft["DATEANDTIME"] == t]

                # Plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.set_global()
                ax.gridlines(draw_labels=True)
                points = ax.scatter(
                    dft["LONGITUDE"], dft["LATITUDE"], c=dft[param],
                    cmap=cmap, norm=norm, s=0.5, marker="s", plotnonfinite=True,
                )
                # Use the scatter for the colorbar
                fig.colorbar(points, ax=ax, location="bottom", pad=0.05)
                title = f"{param} - {depth}m"
                file_name = f"{param}_depth{depth}"
                if has_time:
                    title += f" - {t.year}"
                    file_name += f"_time{t.year}"
                ax.set_title(title)
                plt.tight_layout()
                axes.append(ax)
                if savefig_folder:
                    # Close saved figures
                    os.makedirs(savefig_folder, exist_ok=True)
                    plt.savefig(f"{savefig_folder}/{file_name}.png", dpi=dpi)
                    plt.close(fig)
    return axes


def plot_missing_value_info_map_over_depth(
    df_wide: pd.DataFrame,
    parameters: list[str],
    *,
    times: list | None = None,
    savefig_folder: str | None = None,
    relative: bool = True,
    dpi: int = 300,
) -> None:
    """Geographical map(s) of missingness averaged over depths.

    Args:
        df_wide: Wide table with ``LATITUDE``, ``LONGITUDE``, ``LEV_M``,
            ``DATEANDTIME`` and one column per parameter.
        parameters: Parameter column names to plot.
        times: Time values to show (``None`` = all).
        savefig_folder: Folder to save figures.
        relative: Plot the proportion of missing depth levels in percent. Otherwise, plot the number
            of depth levels with data.
        dpi: Resolution (dots per inch) used when saving.
    """
    ccrs, _ = _require_cartopy()
    df_wide = df_wide.copy()
    df_wide["DATEANDTIME"] = pd.to_datetime(df_wide["DATEANDTIME"])
    times = times or list(df_wide["DATEANDTIME"].value_counts().index)

    # Colour settings
    cmap = plt.colormaps["viridis"]

    parameters = resolve_param_columns(df_wide, parameters)
    for param in parameters:
        # Shared colour norm for all times of a parameter
        if relative:
            # Percent
            norm = plt.Normalize(0, 100)
        else:
            # 0 to the largest count of depth levels with data
            max_count = df_wide.groupby(["DATEANDTIME", "LATITUDE", "LONGITUDE"])[param].count().max()
            norm = plt.Normalize(0, max(max_count, 1))
        for t in times:
            dft = df_wide[df_wide["DATEANDTIME"] == t]

            # Count depth levels with data per lat/lon cell
            grouped = (
                dft[["LATITUDE", "LONGITUDE", "LEV_M", param]]
                .groupby(["LATITUDE", "LONGITUDE"]).agg("count").reset_index()
            )

            # Compute missingness
            grouped["missing_relative"] = 100 - grouped[param] / grouped["LEV_M"] * 100
            plot_col = "missing_relative" if relative else param

            # Plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.gridlines(draw_labels=True)
            points = ax.scatter(
                grouped["LONGITUDE"], grouped["LATITUDE"],
                c=grouped[plot_col], cmap=cmap, norm=norm,
                marker="s", plotnonfinite=True,
            )
            # Use the scatter for the colorbar
            fig.colorbar(points, ax=ax, location="bottom", pad=0.05)
            ax.set_title(f"% missing values for {param}" if relative else param)
            plt.tight_layout()
            if savefig_folder:
                os.makedirs(savefig_folder, exist_ok=True)
                plt.savefig(f"{savefig_folder}/{param}_time{t.year}.png", dpi=dpi)
            plt.close()


def plot_missing_value_info_map_joint(
    df_wide: pd.DataFrame,
    parameters: list[str],
    *,
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes:
    """Geographical map of missingness averaged across parameters, depths and times.

    Args:
        df_wide: Wide table with ``LATITUDE``, ``LONGITUDE`` and one column
            per parameter.
        parameters: Parameter column names to check jointly.
        ax: Axes to draw into. Must already use a cartopy projection when
            passed explicitly. Creates a new PlateCarree figure when ``None``.
        save_as: Save path.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib Axes.
    """
    ccrs, _ = _require_cartopy()
    ax, standalone = _get_or_create_ax(ax, projection=ccrs.PlateCarree())

    # Compute missingness
    parameters = resolve_param_columns(df_wide, parameters)
    spatial = df_wide.groupby(["LATITUDE", "LONGITUDE"])[parameters].apply(
        lambda g: g.isna().all(axis=1).mean() * 100,
    ).reset_index(name="pct_missing")

    # Geo plot
    ax.coastlines()
    ax.set_global()
    ax.gridlines(draw_labels=True)
    sc = ax.scatter(
        spatial["LONGITUDE"], spatial["LATITUDE"],
        c=spatial["pct_missing"], cmap="YlOrRd", s=1, marker="s", vmin=0, vmax=100,
    )
    ax.figure.colorbar(
        sc, ax=ax, orientation="horizontal", pad=0.05,
        label="% rows with all params missing",
    )
    ax.set_title("Spatial coverage")

    return _finish(ax, standalone, save_as, dpi)
