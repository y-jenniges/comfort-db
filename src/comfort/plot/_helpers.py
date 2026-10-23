"""Shared helpers for the comfort.plot subpackage."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import sqlite3
    import pandas as pd
    from ..qc import QCFilter


# Human-readable labels for raw column names shown in legends/colorbars
_PRETTY_LABELS = {
    "ID": "ID",
    "PROFILE_NUMBER": "Profile number",
    "LEV_M": "Depth [m]",
    "LEV_DBAR": "Depth [dbar]",
    "LATITUDE": "Latitude",
    "LONGITUDE": "Longitude",
    "DATEANDTIME": "Date",
}


def _pretty_label(col: str) -> str:
    """Human-readable label for a raw column name, falling back to *col* itself."""
    return _PRETTY_LABELS.get(col, col)


def _get_or_create_ax(ax=None, figsize=None, **subplot_kw):
    """Return *(ax, standalone)*.

    If *ax* is ``None`` a new figure is created and *standalone* is ``True``.
    """
    # Re-use passed axes when available
    if ax is not None:
        return ax, False
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=subplot_kw)
    return ax, True


def _save_fig(save_as: str | None, dpi: int = 300) -> None:
    """Save the current figure to *save_as*, creating parent dirs as needed."""
    if save_as is None:
        return
    # Ensure target directory exists
    os.makedirs(os.path.dirname(os.path.abspath(save_as)), exist_ok=True)
    # Save
    plt.savefig(save_as, bbox_inches="tight", dpi=dpi)


def _finish(ax, standalone: bool, save_as: str | None, dpi: int = 300):
    """Tight-layout (if standalone) and save."""
    # Only adjust layout when this function created the figure
    if standalone:
        plt.tight_layout()
    _save_fig(save_as, dpi)
    return ax


def _require_cartopy():
    """Raise ``ImportError`` with install instructions if cartopy is missing."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        raise ImportError(
            "cartopy is required for map plots. "
            'Install it with: pip install "comfort-db[geo]"'
        )
    return ccrs, cfeature


def _resolve_data(
    df: pd.DataFrame | None,
    conn: sqlite3.Connection | None,
    parameter: str | None,
    quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
    profiles: list[int] | None = None,
) -> pd.DataFrame:
    """Return *df* if provided, otherwise load from the database.

    Args:
        df: User-supplied data. Returned as-is if not ``None``.
        conn: Database connection (required when *df* is ``None``).
        parameter: Parameter name without the ``P_`` prefix (required when *df* is ``None``).
        quality_flags: QC filter tuples forwarded to the reader.
        profiles: Profile numbers to load. Pushed to SQL when loading from DB.
    """
    # Pass-through when the caller already has a DataFrame
    if df is not None:
        return df
    if conn is None or parameter is None:
        raise ValueError(
            "Provide either a DataFrame (df) or a database connection (conn) "
            "together with a parameter name (parameter)."
        )
    from ..io import read_parameter

    return read_parameter(conn, parameter, quality_flags, profiles=profiles)
