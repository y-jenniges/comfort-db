"""Vertical profile and section plots for COMFORT oceanographic data."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import pandas as pd

from ._helpers import _finish, _get_or_create_ax, _pretty_label, _resolve_data
from ..sections import bin_section

if TYPE_CHECKING:
    import sqlite3
    import matplotlib.axes
    from numpy.typing import ArrayLike
    from ..qc import QCFilter

logger = logging.getLogger(__name__)


def plot_profile(
        df: pd.DataFrame | None = None,
        *,
        param_col: str = "VAL",
        depth_col: str = "LEV_M",
        station_col: str = "ID",
        profile_col: str = "PROFILE_NUMBER",
        profiles: list | None = None,
        param_label: str | None = None,
        depth_label: str = "Depth [m]",
        depth_markers: pd.DataFrame | None = None,
        depth_marker_col: str = "MLD_m",
        show_points: bool = True,
        conn: sqlite3.Connection | None = None,
        parameter: str | None = None,
        quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        save_as: str | None = None,
        dpi: int = 300,
) -> matplotlib.axes.Axes | None:
    """Plot one or more vertical profiles (parameter vs. depth).

    Accepts either a *df* or a database *conn* along with a *parameter* name.

    Args:
        df: Parameter DataFrame.
        param_col: Column with measured values.
        depth_col: Column with depth in metres.
        station_col: Column identifying the station. ``PROFILE_NUMBER``
            only identifies a profile *within* a station, so it is combined
            with *station_col* to separate profiles from different stations.
        profile_col: Column identifying individual profiles.
        profiles: Profile IDs to include. ``None`` plots all.
        param_label: x-axis label. Defaults to *param_col*.
        depth_label: y-axis label.
        depth_markers: Optional per-profile depth annotations, e.g. the
            output of :func:`~comfort.profile_analysis.mixed_layer_depth` or
            :func:`~comfort.profile_analysis.pycnocline_depth`. Must share
            *group_cols* (station/profile columns) with *df*. Matched
            profiles get a dashed horizontal line at the marker depth, in
            the same colour as the profile line.
        depth_marker_col: Column in *depth_markers* holding the depth value,
            e.g. ``"MLD_m"`` or ``"pycnocline_depth_m"``.
        show_points: Mark each actual measurement on the profile line.
            Default ``True``.
        conn: Database connection (alternative to *df*).
        parameter: Parameter name without ``P_`` prefix.
        quality_flags: QC filters forwarded to the reader.
        ax: Axes to draw into. Creates a new figure when ``None``.
        save_as: File path to save the figure.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib Axes, or ``None`` when the data is empty.
    """
    # Apply profile filter in SQL when loading from DB
    # Filter in-memory for DataFrame input
    user_df = df
    df = _resolve_data(df, conn, parameter, quality_flags, profiles=profiles)
    if profiles is not None and user_df is not None:
        df = df[df[profile_col].isin(profiles)]
    if df.empty:
        logger.warning("plot_profile: no data to plot")
        return None

    ax, standalone = _get_or_create_ax(ax)

    # Group by (station, profile)
    group_cols = [c for c in (station_col, profile_col) if c in df.columns]
    if not group_cols:
        group_cols = [profile_col]

    # Look up depth markers (e.g. MLD, pycnocline) per profile
    marker_lookup = None
    if depth_markers is not None:
        missing = [c for c in group_cols if c not in depth_markers.columns]
        if missing:
            logger.warning(
                "plot_profile: depth_markers missing columns %s; skipping markers", missing
            )
        else:
            marker_lookup = depth_markers.set_index(group_cols)[depth_marker_col]

    # Draw each profile as a separate line, with an optional depth marker
    for key, group in df.groupby(group_cols):
        group = group.sort_values(depth_col)
        label = "-".join(str(k) for k in key) if isinstance(key, tuple) else str(key)
        marker_kw = dict(marker="o", markersize=3, markeredgewidth=0) if show_points else {}
        line, = ax.plot(group[param_col], group[depth_col], alpha=0.7, label=label, **marker_kw)

        if marker_lookup is not None and key in marker_lookup.index:
            depth_val = marker_lookup.loc[key]
            if pd.notna(depth_val):
                ax.hlines(
                    depth_val, group[param_col].min(), group[param_col].max(),
                    color=line.get_color(), linestyle="--", alpha=0.9, linewidth=1,
                )

    # Depth increases downward
    ax.invert_yaxis()
    ax.set_xlabel(param_label or param_col)
    ax.set_ylabel(depth_label)

    n_groups = df.groupby(group_cols).ngroups
    if n_groups <= 10:
        title = " - ".join(_pretty_label(c) for c in group_cols)
        ax.legend(title=title, bbox_to_anchor=(1.05, 1), loc="upper left")

    return _finish(ax, standalone, save_as, dpi)


def plot_section(
        df: pd.DataFrame | None = None,
        *,
        param_col: str = "VAL",
        depth_col: str = "LEV_M",
        along_col: str | None = None,
        lat_col: str = "LATITUDE",
        lon_col: str = "LONGITUDE",
        along_label: str | None = None,
        depth_label: str = "Depth [m]",
        param_label: str | None = None,
        cmap: str = "viridis",
        clim: tuple[float, float] | None = None,
        binned: bool = False,
        along_bins: int | ArrayLike = 60,
        depth_bins: int | ArrayLike = 40,
        conn: sqlite3.Connection | None = None,
        parameter: str | None = None,
        quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        save_as: str | None = None,
        dpi: int = 300,
) -> matplotlib.axes.Axes | None:
    """Vertical section plot (depth vs. along-track position).

    Accepts either a *df* or a database *conn* along with a *parameter* name.

    The horizontal axis defaults to whichever of latitude or longitude spans
    a larger range or can be set explicitly via *along_col*. For an arbitrary
    cruise track, precompute cumulative along-track distance with
    :func:`~comfort.geo.along_track_distance` and pass its column name as
    *along_col*.

    Args:
        df: Parameter DataFrame with latitude/longitude columns.
        param_col: Column with measured values.
        depth_col: Column with depth in metres.
        along_col: Column to use as horizontal axis. Auto-selected from
            *lat_col*/*lon_col* when ``None``. Pass the output column of
            :func:`~comfort.geo.along_track_distance` to plot along an
            arbitrary transect instead.
        lat_col: Latitude column name.
        lon_col: Longitude column name.
        along_label: Horizontal axis label.
        depth_label: Vertical axis label.
        param_label: Colour-bar label. Defaults to *param_col*.
        cmap: Matplotlib colormap name.
        clim: Colour limits ``(vmin, vmax)``.
        binned: If ``True``, bin *along_col* and *depth_col* into a
            (along_bins x depth_bins) grid via :func:`~comfort.sections.bin_section`,
            average *param_col* within each cell and draw a ``pcolormesh``
            instead of a raw scatter. Default is False.
        along_bins: Number of along-axis bins, or explicit bin edges, when
            *binned* is ``True``.
        depth_bins: Number of depth bins, or explicit bin edges, when
            *binned* is ``True``.
        conn: Database connection (alternative to *df*).
        parameter: Parameter name without ``P_`` prefix.
        quality_flags: QC filters when reading parameters.
        ax: Axes to draw into.
        save_as: File path to save the figure.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib Axes, or ``None`` when the data is empty.

    Raises:
        ValueError: If no position column can be found and *along_col* is not set.
    """
    df = _resolve_data(df, conn, parameter, quality_flags)

    # Auto-select the coordinate axis with the larger spatial range
    if along_col is None:
        if lat_col in df.columns and lon_col in df.columns:
            lat_span = df[lat_col].max() - df[lat_col].min()
            lon_span = df[lon_col].max() - df[lon_col].min()
            along_col = lat_col if lat_span >= lon_span else lon_col
        elif lat_col in df.columns:
            along_col = lat_col
        elif lon_col in df.columns:
            along_col = lon_col
        else:
            raise ValueError("No position column found; pass along_col explicitly")

    # Drop NaNs
    plot_df = df[[param_col, depth_col, along_col]].dropna()
    if plot_df.empty:
        logger.warning("plot_section: no valid data after dropping NaNs")
        return None

    ax, standalone = _get_or_create_ax(ax)

    if binned:
        # Average onto an (along-bin, depth-bin) grid
        section = bin_section(plot_df, param_col=param_col, depth_col=depth_col, along_col=along_col,
                              along_bins=along_bins, depth_bins=depth_bins)
        pivot = section.pivot(index=along_col, columns=depth_col, values=param_col)
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)
        along_centers = pivot.index.values
        depth_vals = pivot.columns.values
        mesh = ax.pcolormesh(along_centers, depth_vals, pivot.T.values, cmap=cmap, shading="nearest")
        if clim:
            mesh.set_clim(*clim)
        ax.figure.colorbar(mesh, ax=ax, label=param_label or param_col)
    else:
        # Colour-coded scatter, depth on the y-axis (inverted)
        sc = ax.scatter(
            plot_df[along_col], plot_df[depth_col],
            c=plot_df[param_col], cmap=cmap, s=5, alpha=0.8,
        )
        if clim:
            sc.set_clim(*clim)
        ax.figure.colorbar(sc, ax=ax, label=param_label or param_col)
    ax.invert_yaxis()
    ax.set_xlabel(along_label or _pretty_label(along_col))
    ax.set_ylabel(depth_label)

    return _finish(ax, standalone, save_as, dpi)
