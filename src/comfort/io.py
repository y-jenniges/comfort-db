"""High-level I/O helpers for reading and subsetting COMFORT parameter data."""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import xarray as xr
    import geopandas as gpd

from .profile_analysis import interpolate_depth_levels
from .database.information import get_names_of_all_parameter_tables
from .geo import region_bbox, water_mass_masks
from .qc import QCFilter, build_where_clause, qc_conditions
from .util.sqlite_utils import validate_identifier

logger = logging.getLogger(__name__)


def _refine_to_region(df, region, region_name_col):
    """Drop rows outside the exact region polygon(s), e.g. after SQL-level bounding box filtering."""
    # Check if region and df have data
    if region is None or df.empty:
        return df

    # Check if latitude and longitude columns are given
    if "LATITUDE" not in df.columns or "LONGITUDE" not in df.columns:
        logger.warning("Region filter requested but LATITUDE/LONGITUDE columns not found - filter skipped.")
        return df

    # Filter for exact regions
    return df[water_mass_masks(df, region, region_name_col=region_name_col)["region"].notna()]


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a connection to the COMFORT SQLite database and close it on exit.

    Args:
        db_path (str or Path): Path to the COMFORT SQLite database.
    Yields:
        sqlite3.Connection
    """
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


# Depth levels (similar to Jenniges et al. 2025 (https://doi.org/10.1016/j.ecoinf.2025.103390)
DEPTH_INTERVALS = np.array([0, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000])

# Columns of the info()/describe_variables() summaries (used for empty results)
_INFO_COLUMNS = ["parameter", "n_samples", "depth_min_m", "depth_max_m", "date_min", "date_max"]
_DESCRIBE_COLUMNS = ["parameter", "n_samples", "n_stations", "n_profiles", "n_instruments",
                     "depth_min_m", "depth_max_m", "val_min", "val_max",
                     "units", "instruments", "platforms"]


def info(db_path_or_conn: str | Path | sqlite3.Connection) -> pd.DataFrame:
    """Print a quick summary of the COMFORT database without loading data.

    Shows parameter names, row counts, depth range and date range.

    Args:
        db_path_or_conn (str, Path or sqlite3.Connection): Path to the
            COMFORT SQLite database or an existing open connection.
    Returns:
        pandas.DataFrame: Summary table (one row per parameter).
    """
    # Open connection if path given
    own_conn = not isinstance(db_path_or_conn, sqlite3.Connection)
    conn = sqlite3.connect(db_path_or_conn) if own_conn else db_path_or_conn
    try:
        params = list_parameters(conn)
        if not params:
            logger.warning("info: no parameter tables found")
            return pd.DataFrame(columns=_INFO_COLUMNS)

        # Collect sample count and depth/date range per parameter
        rows = []
        for param in params:
            validate_identifier(param)
            table = f"P_{param}"
            cur = conn.cursor()

            # Get depth range
            cur.execute(
                f"SELECT COUNT(*), MIN(LEV_M), MAX(LEV_M) FROM {table};"
            )
            n, dmin, dmax = cur.fetchone()

            # Get date range via station join
            cur.execute(
                f"SELECT MIN(s.DATEANDTIME), MAX(s.DATEANDTIME) "
                f"FROM {table} p JOIN station s ON p.ID = s.ID;"
            )
            date_min, date_max = cur.fetchone()

            rows.append({
                "parameter": param,
                "n_samples": n,
                "depth_min_m": dmin,
                "depth_max_m": dmax,
                "date_min": date_min,
                "date_max": date_max,
            })

        result = pd.DataFrame(rows)
        logger.info(result.to_string(index=False))
        return result
    finally:
        if own_conn:
            conn.close()


def list_parameters(conn: sqlite3.Connection) -> list[str]:
    """Return the names of all ``P_*`` parameter tables (without the ``P_`` prefix).

    Args:
        conn (sqlite3.Connection): Connection to the database.
    Returns:
        list[str]: Parameter names, e.g. ``['NITRATE', 'OXYGEN', ...]``.
    """
    return [name[2:] for name in get_names_of_all_parameter_tables(conn)]


def read_parameter(conn: sqlite3.Connection, param_name: str,
                   quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
                   profiles: list[int] | None = None,
                   lat_min: float | None = None, lat_max: float | None = None,
                   lon_min: float | None = None, lon_max: float | None = None,
                   depth_min: float | None = None, depth_max: float | None = None,
                   date_min: str | None = None, date_max: str | None = None,
                   region: dict | str | Path | gpd.GeoDataFrame | gpd.GeoSeries | None = None,
                   region_name_col: str = "name",
                   limit: int | None = None,
                   extra_station_cols: list[str] | None = None) -> pd.DataFrame:
    """Read a P_* parameter table as a DataFrame with optional QC filtering.

    ``station`` table is always joined, so ``LATITUDE``, ``LONGITUDE``
    and ``DATEANDTIME`` columns are always included.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        param_name (str): Parameter name (without ``'P_'`` prefix).
        quality_flags (list[QCFilter] or list[tuple[str, str]]): QC filters,
            e.g. ``QC_GOOD``.
        profiles (list[int]): Profile numbers to load. ``None`` loads all.
        lat_min (float): Minimum latitude [degrees N].
        lat_max (float): Maximum latitude [degrees N].
        lon_min (float): Minimum longitude [degrees E].
        lon_max (float): Maximum longitude [degrees E].
        depth_min (float): Minimum depth [m].
        depth_max (float): Maximum depth [m].
        date_min (str or datetime-like): Start date (inclusive).
        date_max (str or datetime-like): End date (inclusive).
        region: Restrict to an arbitrary polygon (e.g. an ocean basin or a
            Longhurst province) instead of or in addition to the lat/lon box above.
            Use :func:`comfort.geo.select_regions` to select one or more named regions.
        region_name_col (str): Region name column for *region*.
        limit (int): Maximum number of rows to return (the first rows in
            table order). ``None`` returns all. Useful as a safeguard for
            large tables.
        extra_station_cols (list[str]): Additional ``station`` columns to
            include (e.g. ``["CRUISE_ID", "ST_NUMBER_ORIGIN"]``).
    Returns:
        pandas.DataFrame
    """
    # Bounding box of the region (ANDed with the explicit bounds in the SQL query)
    region_box = region_bbox(region, region_name_col) if region is not None else None

    # Get the data from the database
    df = _read_with_geo(conn, param_name, quality_flags,
                        lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
                        depth_min=depth_min, depth_max=depth_max, date_min=date_min, date_max=date_max,
                        limit=limit, region_box=region_box,
                        profiles=profiles, extra_station_cols=extra_station_cols)

    # Refine SQL's bbox to exact region polygon
    return _refine_to_region(df, region, region_name_col)


def read_station(conn: sqlite3.Connection) -> pd.DataFrame:
    """Read the station metadata table as a DataFrame.

    Args:
        conn (sqlite3.Connection): Connection to the database.
    Returns:
        pandas.DataFrame
    """
    df = pd.read_sql_query("SELECT * FROM STATION;", conn)
    if "DATEANDTIME" in df.columns:
        df["DATEANDTIME"] = df["DATEANDTIME"].str.slice(0, 19)
    return df


def describe_variables(conn: sqlite3.Connection,
                       parameters: list[str] | None = None,
                       quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Summarise available parameters: sample counts, depth range, value range,
    units, instruments and platforms.

    Lookup tables (``UNITS``, ``INSTRUMENT``, ``PLATFORM``) resolve IDs to
    human-readable names when present; otherwise IDs are shown as strings.
    Parameters with multiple distinct values show all names separated by ``/``.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        parameters (list[str]): Parameter names (without ``P_`` prefix).
            ``None`` describes all available parameters.
        quality_flags (list[list[str, str]]): Optional QC filters applied
            when counting samples and computing ranges.
    Returns:
        pandas.DataFrame: One row per parameter with columns
            ``[parameter, n_samples, n_stations, n_profiles, n_instruments,
            depth_min_m, depth_max_m, val_min, val_max,
            units, instruments, platforms]``.
    """
    # Get a list of all parameters
    all_params = list_parameters(conn)

    # Iterate over given parameters
    if parameters is not None:
        # Warn if parameters are unknown
        unknown = set(parameters) - set(all_params)
        if unknown:
            raise ValueError(f"Unknown parameters: {sorted(unknown)}")
        params = parameters
    else:
        params = all_params

    # Warn if no parameter left
    if not params:
        logger.warning("describe_variables: no parameter tables found")
        return pd.DataFrame(columns=_DESCRIBE_COLUMNS)

    # Compute aggregate statistics per parameter
    where = build_where_clause(quality_flags)
    qc_where_aliased = build_where_clause(quality_flags, "p")
    has_platform_id = _station_has_column(conn, "PLATFORM_ID")

    rows = []
    for param in params:
        # Validate identifier
        validate_identifier(param)

        # Define table name
        table = f"P_{param}"

        # Connect to db and fetch parameter statistics
        cur = conn.cursor()
        cur.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT ID), "
            f"COUNT(DISTINCT INSTRUMENT_ID), "
            f"MIN(LEV_M), MAX(LEV_M), MIN(VAL), MAX(VAL) "
            f"FROM {table} {where};"
        )
        n, n_sta, n_inst, dmin, dmax, vmin, vmax = cur.fetchone()

        # Count distinct (ID, PROFILE_NUMBER) pairs - PROFILE_NUMBER is only unique within a station
        cur.execute(
            f"SELECT COUNT(*) FROM "
            f"(SELECT DISTINCT ID, PROFILE_NUMBER FROM {table} {where});"
        )
        n_prof = cur.fetchone()[0]

        # Get IDs of units and instruments
        cur.execute(f"SELECT DISTINCT UNITS_ID FROM {table} {where};")
        unit_ids = [row[0] for row in cur.fetchall() if row[0] is not None]
        cur.execute(f"SELECT DISTINCT INSTRUMENT_ID FROM {table} {where};")
        inst_ids = [row[0] for row in cur.fetchall() if row[0] is not None]

        # Get platform IDs via station join
        platform_ids: list[int] = []
        if has_platform_id:
            cur.execute(
                f"SELECT DISTINCT s.PLATFORM_ID "
                f"FROM {table} p JOIN station s ON p.ID = s.ID "
                f"{qc_where_aliased};"
            )
            platform_ids = [row[0] for row in cur.fetchall() if row[0] is not None]

        rows.append({
            "parameter": param,
            "n_samples": n,
            "n_stations": n_sta,
            "n_profiles": n_prof,
            "n_instruments": n_inst,
            "depth_min_m": dmin,
            "depth_max_m": dmax,
            "val_min": vmin,
            "val_max": vmax,
            "units_ids": unit_ids,
            "instrument_ids": inst_ids,
            "platform_ids": platform_ids,
        })
    result = pd.DataFrame(rows)

    # Resolve unit IDs to names
    id_to_unit = _load_lookup_map(conn, "UNITS", "ID", "NAME_SHORT")
    result["units"] = result["units_ids"].apply(lambda ids: _join_names(ids, id_to_unit))
    # Warn about mixed units
    for _, row in result.iterrows():
        if len(row["units_ids"]) > 1:
            logger.warning(
                "%s contains %d different units (%s) - values may not be comparable",
                row["parameter"], len(row["units_ids"]), row["units"],
            )
    result = result.drop(columns="units_ids")

    # Resolve instrument IDs to names
    id_to_instrument = _load_lookup_map(conn, "INSTRUMENT")
    result["instruments"] = result["instrument_ids"].apply(lambda ids: _join_names(ids, id_to_instrument))
    result = result.drop(columns="instrument_ids")

    # Resolve platform IDs to names
    id_to_platform = _load_lookup_map(conn, "PLATFORM")
    result["platforms"] = result["platform_ids"].apply(lambda ids: _join_names(ids, id_to_platform))
    result = result.drop(columns="platform_ids")

    return result


def _load_lookup_map(conn, table: str, id_col: str = "ID", name_col: str = "NAME") -> dict:
    """Return a dict mapping id_col -> name_col, or {} when the table is absent."""
    from .database.information import does_table_exist

    # Check if table exists
    if not does_table_exist(conn, table):
        return {}
    cur = conn.cursor()

    # Get table info
    cur.execute(f"PRAGMA table_info({table});")
    columns = {row[1].upper() for row in cur.fetchall()}

    # Check for missing columns
    if id_col.upper() not in columns or name_col.upper() not in columns:
        logger.debug("_load_lookup_map: %s lacks %s/%s", table, id_col, name_col)
        return {}

    cur.execute(f"SELECT {id_col}, {name_col} FROM {table};")
    return {row[0]: row[1] for row in cur.fetchall()}


def _station_has_column(conn, column: str) -> bool:
    """Check whether the station table has a given column."""
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(station);")
    return any(row[1].upper() == column.upper() for row in cur.fetchall())


class _Lookups(NamedTuple):
    """Name maps for the xarray coordinates and unit attributes of load_comfort."""
    instrument: dict
    platform: dict
    units: dict
    station_platform: dict


def _load_lookups(conn) -> _Lookups:
    """Read the instrument, platform and unit ID->name maps and the station->platform_id map."""
    # Station -> platform_id mapping (if column exists)
    station_platform: dict[int, int] = {}
    if _station_has_column(conn, "PLATFORM_ID"):
        cur = conn.cursor()
        cur.execute("SELECT ID, PLATFORM_ID FROM station;")
        station_platform = {row[0]: row[1] for row in cur.fetchall() if row[1] is not None}

    return _Lookups(
        instrument=_load_lookup_map(conn, "INSTRUMENT"),
        platform=_load_lookup_map(conn, "PLATFORM"),
        units=_load_lookup_map(conn, "UNITS", "ID", "NAME_SHORT"),
        station_platform=station_platform,
    )


def subset_region(df: pd.DataFrame,
                  lat_min: float | None = None, lat_max: float | None = None,
                  lon_min: float | None = None, lon_max: float | None = None,
                  depth_min: float | None = None, depth_max: float | None = None,
                  date_min: str | None = None, date_max: str | None = None,
                  region: dict | str | Path | gpd.GeoDataFrame | gpd.GeoSeries | None = None,
                  region_name_col: str = "name") -> pd.DataFrame:
    """Filter a DataFrame to a geographic, depth and/or time region.

    Logs a warning when a requested bound cannot be applied because the
    corresponding column is absent. Recognised columns: LATITUDE, LONGITUDE,
    LEV_M, DATEANDTIME.

    Args:
        df (pandas.DataFrame): Extended parameter or station DataFrame.
        lat_min (float): Minimum latitude in degrees N.
        lat_max (float): Maximum latitude in degrees N.
        lon_min (float): Minimum longitude in degrees E.
        lon_max (float): Maximum longitude in degrees E.
        depth_min (float): Minimum depth in metres.
        depth_max (float): Maximum depth in metres.
        date_min (str or datetime-like): Start date (inclusive).
        date_max (str or datetime-like): End date (inclusive).
        region: Restrict to an arbitrary polygon (e.g. an ocean basin or a
            Longhurst province) instead of or in addition to the lat/lon box above.
            Use :func:`comfort.geo.select_regions` to select one or more named regions.
        region_name_col (str): Region name column for *region*.
    Returns:
        pandas.DataFrame: Filtered copy.
    """
    # Apply a single bound, skip with warning if column absent
    def _apply(mask, col, bound_name, bound_val, op):
        if bound_val is None:
            return mask
        if col not in df.columns:
            logger.warning(
                "subset_region: %r requested but column %r not found - filter skipped",
                bound_name, col,
            )
            return mask
        return mask & op(df[col], bound_val)

    import operator as op

    # Spatial bounds
    mask = pd.Series(True, index=df.index)
    mask = _apply(mask, "LATITUDE", "lat_min", lat_min, op.ge)
    mask = _apply(mask, "LATITUDE", "lat_max", lat_max, op.le)
    mask = _apply(mask, "LONGITUDE", "lon_min", lon_min, op.ge)
    mask = _apply(mask, "LONGITUDE", "lon_max", lon_max, op.le)
    mask = _apply(mask, "LEV_M", "depth_min", depth_min, op.ge)
    mask = _apply(mask, "LEV_M", "depth_max", depth_max, op.le)

    # Date range filter
    if date_min is not None or date_max is not None:
        if "DATEANDTIME" not in df.columns:
            logger.warning(
                "subset_region: date filter requested but column 'DATEANDTIME' not found - filter skipped"
            )
        else:
            dates = pd.to_datetime(df["DATEANDTIME"])
            if date_min is not None:
                mask &= dates >= pd.Timestamp(date_min)
            if date_max is not None:
                mask &= dates <= pd.Timestamp(date_max)

    # Exact polygon refine (in addition to the explicit bounds above)
    return _refine_to_region(df[mask].copy(), region, region_name_col)


def load_comfort(
        db_path_or_conn: str | Path | sqlite3.Connection,
        parameters: list[str] | None = None,
        quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
        lat_min: float | None = None, lat_max: float | None = None,
        lon_min: float | None = None, lon_max: float | None = None,
        depth_min: float | None = None, depth_max: float | None = None,
        date_min: str | None = None, date_max: str | None = None,
        region: dict | str | Path | gpd.GeoDataFrame | gpd.GeoSeries | None = None,
        region_name_col: str = "name",
        target_depths: np.ndarray | list[float] | None = None,
        max_gap: float | None = None,
        interp_method: str = "pchip",
        as_xarray: bool = True,
        convert_units: bool = False,
        use_density: bool = False,
        on_missing: str = "drop",
        density_fallback: str = "constant",
        limit: int | None = None,
) -> xr.Dataset | dict[str, pd.DataFrame]:
    """Load COMFORT data into a filtered xarray Dataset or dict of DataFrames.

    All filtering (QC flags, spatial bounds and date range) is pushed to SQL
    so only the requested rows are loaded into memory (``P_* JOIN station``).

    If *target_depths* is specified, the profiles are interpolated onto
    those depths using the specified interpolation method *interp_method*
    and a maximum gap between depth *max_gap*.

    Args:
        db_path_or_conn (str, Path or sqlite3.Connection): Path to the COMFORT
            SQLite database or an existing open connection. When a path is
            given, the connection is opened and closed automatically. When a
            connection is passed, it is left open.
        parameters (list[str]): Parameter names (without prefix).
            ``None`` loads all available parameters.
        quality_flags: QC filter preset, e.g. :data:`comfort.qc.QC_GOOD`.
            ``None`` means no filtering.
        lat_min (float): Minimum latitude [degrees N].
        lat_max (float): Maximum latitude [degrees N].
        lon_min (float): Minimum longitude [degrees E].
        lon_max (float): Maximum longitude [degrees E].
        depth_min (float): Minimum depth [m].
        depth_max (float): Maximum depth [m].
        date_min (str or datetime-like): Start date (inclusive).
        date_max (str or datetime-like): End date (inclusive).
        region: Restrict to an arbitrary polygon (e.g. an ocean basin or a
            Longhurst province) instead of or in addition to the lat/lon box above.
            Use :func:`comfort.geo.select_regions` to select one or more named regions.
        region_name_col (str): Region name column for *region*. See
            :func:`comfort.geo.water_mass_masks`.
        target_depths (array-like): Depth levels [m] to interpolate onto.
            ``None`` (default) skips interpolation.
        max_gap (float): Maximum allowed distance [m] between the two
            measured depths bracketing a target depth, forwarded to
            :func:`~comfort.profile_analysis.interpolate_depth_levels`.
            ``None`` (default) disables the check. Ignored when
            *target_depths* is ``None``.
        interp_method (str): Depth-interpolation method forwarded to
            :func:`~comfort.profile_analysis.interpolate_depth_levels`:
            ``"pchip"`` (default) or ``"linear"``. Ignored when
            *target_depths* is ``None``.
        as_xarray (bool): ``True`` returns ``xr.Dataset``, ``False`` returns
            ``dict[param_name, pandas.DataFrame]``.
        convert_units (bool): When ``True``, convert each parameter's values
            to its default unit using :class:`comfort.units.UnitsConverter`.
        use_density (bool): When ``True`` and ``convert_units=True``, use
            computed density for volumetric conversions instead of the
            constant 1.025 kg/L (COMFORT report Appendix C conventions:
            lab density at 22 °C for most parameters, in-situ-temperature
            density for chlorophyll and oxygen mL/L, both at atmospheric
            pressure. Needs temperature/salinity).
        on_missing (str): Policy for rows that cannot be converted when
            ``convert_units=True``: ``'drop'`` (default),
            ``'keep'`` (retain original value and UNITS_ID) or ``'raise'``.
        density_fallback (str): With ``use_density=True``, policy for rows whose
            density cannot be computed (missing/NaN salinity, temperature or
            position): ``'constant'`` (default, use 1.025 kg/L), ``'drop'`` or
            ``'raise'``. Counts are logged.
        limit (int): Maximum number of rows to return (the first rows in
            table order). ``None`` returns all. Useful as a safeguard for
            large tables.
    Returns:
        xarray.Dataset or dict[str, pandas.DataFrame].
        For the dict path each DataFrame has the measurement column named
        after the parameter (e.g. ``"NITRATE"``), not ``"VAL"``.
        For the scattered xarray path (``as_xarray=True,
        target_depths=None``), each parameter's coordinates are prefixed
        with its name (e.g. ``NITRATE_depth``, ``NITRATE_latitude``) since
        different parameters have independent observation dimensions.
    """
    # Collect geo/temporal filter arguments (the region bounding box is ANDed with the explicit bounds in SQL)
    geo_kwargs = dict(
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
        depth_min=depth_min, depth_max=depth_max,
        date_min=date_min, date_max=date_max,
        region_box=region_bbox(region, region_name_col) if region is not None else None,
    )

    # Open connection if path given; leave caller-owned connections open
    own_conn = not isinstance(db_path_or_conn, sqlite3.Connection)
    conn = sqlite3.connect(db_path_or_conn) if own_conn else db_path_or_conn
    try:
        # Default to all available parameters
        available = list_parameters(conn)
        if parameters is None:
            parameters = available
        else:
            # Fail (before loading anything) if a requested parameter table is missing
            unknown = {p for p in parameters if p.upper() not in {a.upper() for a in available}}
            if unknown:
                raise ValueError(f"Unknown parameters: {sorted(unknown)}")

        # Resolve lookup tables for xarray coordinates
        lookups = _load_lookups(conn)

        # Build unit converter when requested
        _converter = None
        if convert_units:
            from .units import UnitsConverter
            _converter = UnitsConverter.from_connection(conn)

        # Track per-parameter unit IDs for xarray attrs and warnings
        _param_unit_ids: dict[str, set] = {}

        # Load each parameter with SQL-level filtering
        dfs = {}
        for param in parameters:
            logger.info("load_comfort: Loading %s...", param)
            # Check if temperature and/or salinity are required for unit conversions
            with_temperature, with_salinity = (
                _converter.required_inputs(f"P_{param.upper()}", use_density) if _converter else (False, False))

            # Load param with space, time and optionally T and S columns
            df = _read_with_geo(conn, param, quality_flags, limit=limit,
                                with_temperature=with_temperature,
                                with_salinity=with_salinity, **geo_kwargs)

            # Refine to exact polygon refine (SQL only filters for region bbox)
            df = _refine_to_region(df, region, region_name_col)

            if not df.empty:
                # Track distinct units
                if "UNITS_ID" in df.columns:
                    uid_set = set(df["UNITS_ID"].dropna().unique())
                    _param_unit_ids[param] = uid_set
                    if len(uid_set) > 1 and _converter is None:
                        names = _join_names(uid_set, lookups.units, sep=", ")
                        logger.warning(
                            "%s contains %d different units (%s) - "
                            "values may not be comparable; consider convert_units=True",
                            param, len(uid_set), names,
                        )

                # Convert to default unit before renaming VAL
                if _converter is not None and "VAL" in df.columns:
                    df = _converter.convert_dataframe(df, f"P_{param}",
                                                      use_density=use_density,
                                                      on_missing=on_missing,
                                                      density_fallback=density_fallback)
                    # Empty df warning
                    if df.empty:
                        logger.warning("load_comfort: no %s rows left after unit conversion", param)
                        continue

                    # Update unit IDs
                    if "UNITS_ID" in df.columns:
                        _param_unit_ids[param] = set(df["UNITS_ID"].dropna().unique())

                # Rename generic VAL column to parameter name
                if "VAL" in df.columns:
                    df = df.rename(columns={"VAL": param})
                dfs[param] = df
    finally:
        if own_conn:
            conn.close()

    if not dfs:
        logger.warning("load_comfort: no data found for the given filters")
        if as_xarray:
            import xarray as xr
            return xr.Dataset()
        return {}

    # target_depths=None: Return each profile at its native measured depths
    if target_depths is None:
        if not as_xarray:
            return dfs
        return _build_scattered_dataset(dfs, lookups=lookups, param_unit_ids=_param_unit_ids)

    # target_depths given: Project every parameter onto it
    depths = np.asarray(target_depths, dtype=float)
    interp_dfs = {
        param: interpolate_depth_levels(df, depths, param_col=param, depth_col="LEV_M",
                                        max_gap=max_gap, method=interp_method)
        for param, df in dfs.items()
    }
    if not as_xarray:
        return interp_dfs

    return _build_profile_dataset(dfs, interp_dfs=interp_dfs, depths=depths, lookups=lookups, param_unit_ids=_param_unit_ids)


def _build_profile_dataset(dfs, interp_dfs, depths, lookups, param_unit_ids):
    """Build an xr.Dataset on a shared (profile, depth) grid from depth-interpolated parameters.

    Profiles are identified by (station ID, profile number). Their coordinates (position, time,
    instrument, platform) come from the first row of each profile in the un-interpolated *dfs*.
    """
    # Build xarray Dataset with a shared (profile, depth) grid
    import xarray as xr

    # Collect unique (station_id, profile_number) pairs as canonical profile identities
    all_profiles = sorted(
        set().union(*[
            set(zip(df["ID"], df["PROFILE_NUMBER"]))
            if "ID" in df.columns and "PROFILE_NUMBER" in df.columns
            else {(None, pn) for pn in df["PROFILE_NUMBER"].unique()}
            for df in dfs.values()
        ])
    )
    n_profiles = len(all_profiles)
    n_depths = len(depths)

    # Extract station metadata from first row of each (ID, PROFILE_NUMBER) pair
    meta: dict = {}
    for df in dfs.values():
        group_keys = ["ID", "PROFILE_NUMBER"] if "ID" in df.columns else ["PROFILE_NUMBER"]
        for pid, group in df.groupby(group_keys):
            key = pid if isinstance(pid, tuple) else (None, pid)
            if key not in meta:
                row = group.iloc[0]
                sid = int(row["ID"]) if "ID" in row.index and pd.notna(row["ID"]) else np.nan
                meta[key] = {
                    "latitude": row.get("LATITUDE", np.nan),
                    "longitude": row.get("LONGITUDE", np.nan),
                    "time": row.get("DATEANDTIME", None),
                    "station_id": sid,
                    "profile_number": int(row["PROFILE_NUMBER"]),
                    "instrument_id": row.get("INSTRUMENT_ID", np.nan),
                    "platform_id": lookups.station_platform.get(sid, np.nan),
                }

    # Pivot each parameter's interpolated values onto the shared grid
    data_vars = {}
    for param, interp in interp_dfs.items():
        if interp.empty:
            arr = np.full((n_profiles, n_depths), np.nan)
        else:
            # Build composite key column for pivot; auto-detected profile key matches all_profiles tuples
            if "ID" in interp.columns and "PROFILE_NUMBER" in interp.columns:
                interp = interp.assign(
                    _profile_key=list(zip(interp["ID"], interp["PROFILE_NUMBER"]))
                )
            else:
                interp = interp.assign(
                    _profile_key=[(None, pn) for pn in interp["PROFILE_NUMBER"]]
                )
            pivot = interp.pivot_table(
                index="_profile_key", columns="LEV_M",
                values=param, aggfunc="first",
            )
            pivot = pivot.reindex(index=all_profiles, columns=depths, fill_value=np.nan)
            arr = pivot.to_numpy(dtype=float)
        data_vars[param] = (["profile", "depth"], arr)

    # Build coordinate arrays from profile metadata
    lats = np.array([meta.get(p, {}).get("latitude", np.nan) for p in all_profiles])
    lons = np.array([meta.get(p, {}).get("longitude", np.nan) for p in all_profiles])
    station_ids = np.array([meta.get(p, {}).get("station_id", np.nan) for p in all_profiles])
    profile_numbers = np.array([meta[p]["profile_number"] for p in all_profiles])
    instrument_ids = [meta.get(p, {}).get("instrument_id", np.nan) for p in all_profiles]
    platform_ids = [meta.get(p, {}).get("platform_id", np.nan) for p in all_profiles]
    raw_times = [meta.get(p, {}).get("time") for p in all_profiles]
    try:
        times = pd.to_datetime(raw_times)
    except (ValueError, TypeError):
        times = np.array(raw_times, dtype=object)

    # Resolve instrument and platform IDs to names
    instruments = _id_names(instrument_ids, lookups.instrument)
    platforms = _id_names(platform_ids, lookups.platform)

    # Assemble xarray Dataset
    ds = xr.Dataset(
        data_vars,
        coords={
            "depth": depths,
            "profile_number": ("profile", profile_numbers),
            "station_id": ("profile", station_ids),
            "instrument": ("profile", instruments),
            "platform": ("profile", platforms),
            "latitude": ("profile", lats),
            "longitude": ("profile", lons),
            "time": ("profile", times),
        },
    )

    # Attach resolved unit names as variable attributes
    _attach_unit_attrs(ds, params=data_vars, param_unit_ids=param_unit_ids, units_map=lookups.units)

    # Climate and Forecast (CF) compliant metadata
    ds["depth"].attrs.update({"units": "m", "positive": "down"})
    ds["latitude"].attrs["units"] = "degrees_north"
    ds["longitude"].attrs["units"] = "degrees_east"
    return ds


def _join_names(ids, lookup: dict, sep: str = " / ") -> str | None:
    """Join the names of *ids* (sorted, falling back to its stringified ID). ``None`` without IDs."""
    if len(ids) == 0:
        return None
    return sep.join(lookup.get(int(i), str(int(i))) for i in sorted(ids))


def _attach_unit_attrs(ds, params, param_unit_ids, units_map) -> None:
    """Set each parameter's ``units`` attribute to the resolved name(s) of the units found in its data."""
    # Iterate over params
    for param in params:
        uid_set = param_unit_ids.get(param, set())  # get unit ids of the parameter
        if uid_set:
            # Resolve unit ID names
            ds[param].attrs["units"] = _join_names(ids=uid_set, lookup=units_map)


def _id_names(ids, lookup: dict) -> np.ndarray:
    """Resolve IDs to names: ``"unknown"`` for a missing ID, the stringified ID if not in *lookup*."""
    return np.array([
        "unknown" if pd.isna(i) else lookup.get(int(i), str(int(i))) for i in ids
    ])


def _build_scattered_dataset(dfs, lookups, param_unit_ids):
    """Build an xr.Dataset with each parameter on its own observation
    dimension, at native measured depths (no interpolation, no shared grid).

    Different parameters have independent numbers of observations at
    independent depths, so they need independent dimensions. Therefore,
    each parameter's coordinates are namespaced with its own name (e.g. ``NITRATE_depth``).
    """
    import xarray as xr

    data_vars = {}
    coords = {}
    for param, df in dfs.items():
        dim = f"{param}_obs"
        n = len(df)
        data_vars[param] = (dim, df[param].to_numpy(dtype=float))

        # Define coordinates
        coords[f"{param}_depth"] = (dim, df["LEV_M"].to_numpy(dtype=float))
        coords[f"{param}_latitude"] = (dim, df["LATITUDE"].to_numpy(dtype=float))
        coords[f"{param}_longitude"] = (dim, df["LONGITUDE"].to_numpy(dtype=float))
        try:
            times = pd.to_datetime(df["DATEANDTIME"]).to_numpy()
        except (ValueError, TypeError):
            times = df["DATEANDTIME"].to_numpy(dtype=object)
        coords[f"{param}_time"] = (dim, times)

        # Get station IDs and profile numbers
        station_ids = df["ID"].to_numpy(dtype=int)
        coords[f"{param}_station_id"] = (dim, station_ids)
        coords[f"{param}_profile_number"] = (dim, df["PROFILE_NUMBER"].to_numpy(dtype=int))

        # Get instrument IDs
        instrument_ids = df["INSTRUMENT_ID"] if "INSTRUMENT_ID" in df.columns else np.full(n, np.nan)
        coords[f"{param}_instrument"] = (dim, _id_names(instrument_ids, lookups.instrument))

        # Get platform IDs
        platform_ids = [lookups.station_platform.get(int(sid), np.nan) for sid in station_ids]
        coords[f"{param}_platform"] = (dim, _id_names(platform_ids, lookups.platform))

    ds = xr.Dataset(data_vars, coords=coords)

    # Attach resolved unit names as variable attributes
    _attach_unit_attrs(ds, dfs, param_unit_ids, lookups.units)

    # Metadata per parameter
    for param in dfs:
        ds[f"{param}_depth"].attrs.update({"units": "m", "positive": "down"})
        ds[f"{param}_latitude"].attrs["units"] = "degrees_north"
        ds[f"{param}_longitude"].attrs["units"] = "degrees_east"

    return ds


def _read_with_geo(conn, param_name, quality_flags,
                   lat_min=None, lat_max=None, lon_min=None, lon_max=None,
                   depth_min=None, depth_max=None, date_min=None, date_max=None,
                   limit=None, with_temperature=False, with_salinity=False, region_box=None,
                   profiles=None, extra_station_cols=None):
    """Read a parameter table with lat/lon/time, pushing all filters to SQL.

    Joins ``P_*`` with ``station``. *with_temperature*/*with_salinity* additionally
    attach the co-located temperature/salinity (needed for unit conversions).
    *region_box* is the bounding box of a region (see :func:`_append_geo_conditions`).
    *profiles* restricts to profile numbers, *extra_station_cols* adds ``station`` columns.
    This is the query behind both :func:`read_parameter` and :func:`load_comfort`.
    """
    from .database.information import does_table_exist

    # Validate identifiers
    validate_identifier(param_name)
    for col in extra_station_cols or []:
        validate_identifier(col)

    # Join P_* with station to get geo columns
    params = []
    conditions = qc_conditions(quality_flags, "p")
    if profiles is not None:
        # Filter for profile numbers
        placeholders = ",".join("?" * len(profiles))
        conditions.append(f"p.PROFILE_NUMBER IN ({placeholders})")
        params.extend(profiles)
    _append_geo_conditions(
        conditions, params, "s.", "s.", "p.", "s.",
        lat_min, lat_max, lon_min, lon_max, depth_min, depth_max, date_min, date_max,
        region_box=region_box,
    )
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    # Optionally attach T/S for unit conversion (QC + space + time filtered)
    ts_select, ts_join = "", ""
    ts_params: list = []
    if with_temperature and does_table_exist(conn, "P_TEMPERATURE"):
        t_conditions = qc_conditions(quality_flags, "t")
        t_params: list = []
        _append_geo_conditions(
            t_conditions, t_params, "s2.", "s2.", "t.", "s2.",
            lat_min, lat_max, lon_min, lon_max, depth_min, depth_max, date_min, date_max,
            region_box=region_box,
        )
        t_where = ("WHERE " + " AND ".join(t_conditions)) if t_conditions else ""
        ts_select += ", tmp.temperature"
        ts_join += (f"LEFT JOIN (SELECT t.ID, t.LEV_M, AVG(t.VAL) AS temperature "
                    f"FROM P_TEMPERATURE t JOIN station s2 ON t.ID = s2.ID {t_where} "
                    f"GROUP BY t.ID, t.LEV_M) tmp "
                    f"ON tmp.ID = p.ID AND tmp.LEV_M = p.LEV_M ")
        ts_params.extend(t_params)

    if with_salinity and does_table_exist(conn, "P_SALINITY"):
        sal_conditions = qc_conditions(quality_flags, "t")
        sal_params: list = []
        _append_geo_conditions(
            sal_conditions, sal_params, "s2.", "s2.", "t.", "s2.",
            lat_min, lat_max, lon_min, lon_max, depth_min, depth_max, date_min, date_max,
            region_box=region_box,
        )
        sal_where = ("WHERE " + " AND ".join(sal_conditions)) if sal_conditions else ""
        ts_select += ", sal.salinity"
        ts_join += (f"LEFT JOIN (SELECT t.ID, t.LEV_M, AVG(t.VAL) AS salinity "
                    f"FROM P_SALINITY t JOIN station s2 ON t.ID = s2.ID {sal_where} "
                    f"GROUP BY t.ID, t.LEV_M) sal "
                    f"ON sal.ID = p.ID AND sal.LEV_M = p.LEV_M ")
        ts_params.extend(sal_params)

    # Select additional columns from the station table
    extra_select = "".join(f", s.{col}" for col in (extra_station_cols or []))

    # Assemble the query
    query = (
        f"SELECT p.*, s.LATITUDE, s.LONGITUDE{extra_select}, "
        f"strftime('%Y-%m-%d %H:%M:%S', s.DATEANDTIME) AS DATEANDTIME{ts_select} "
        f"FROM P_{param_name.upper()} p JOIN station s ON p.ID = s.ID {ts_join}{where}"
    )
    # T/S sub-query values come first in the SQL text, so they come first in the parameters
    params = ts_params + params

    # Add optional limit, ORDER BY rowid takes the first rows in table order (reproducible)
    if limit is not None:
        query += f" ORDER BY p.rowid LIMIT {int(limit)}"

    # Execute query with parametrised geo/date values
    return pd.read_sql_query(query, conn, params=params)


def _append_box_conditions(conditions, params, lat_pfx, lon_pfx,
                           lat_min, lat_max, lon_min, lon_max):
    """Append lat/lon bounding-box SQL conditions and ? parameter values in-place."""
    if lat_min is not None:
        conditions.append(f"{lat_pfx}LATITUDE >= ?")
        params.append(float(lat_min))
    if lat_max is not None:
        conditions.append(f"{lat_pfx}LATITUDE <= ?")
        params.append(float(lat_max))
    if lon_min is not None and lon_max is not None and lon_min > lon_max:
        # Region contains the antimeridian (see comfort.geo.region_bbox) -> Use OR
        conditions.append(f"({lon_pfx}LONGITUDE >= ? OR {lon_pfx}LONGITUDE <= ?)")
        params.append(float(lon_min))
        params.append(float(lon_max))
    else:
        # Region does not contain antimeridian -> Use AND
        if lon_min is not None:
            conditions.append(f"{lon_pfx}LONGITUDE >= ?")
            params.append(float(lon_min))
        if lon_max is not None:
            conditions.append(f"{lon_pfx}LONGITUDE <= ?")
            params.append(float(lon_max))


def _append_geo_conditions(conditions, params, lat_pfx, lon_pfx, depth_pfx, date_pfx,
                           lat_min, lat_max, lon_min, lon_max,
                           depth_min, depth_max, date_min, date_max, region_box=None):
    """Append spatial/temporal SQL conditions and ? parameter values in-place."""
    # Append lat/lon conditions
    _append_box_conditions(conditions, params, lat_pfx, lon_pfx, lat_min, lat_max, lon_min, lon_max)

    # Append region bbox conditions
    if region_box is not None:
        _append_box_conditions(conditions, params, lat_pfx, lon_pfx, *region_box)

    # Append depth conditions
    if depth_min is not None:
        conditions.append(f"{depth_pfx}LEV_M >= ?")
        params.append(float(depth_min))
    if depth_max is not None:
        conditions.append(f"{depth_pfx}LEV_M <= ?")
        params.append(float(depth_max))

    # Append time conditions
    if date_min is not None:
        conditions.append(f"{date_pfx}DATEANDTIME >= ?")
        params.append(pd.Timestamp(date_min).isoformat(sep=" ", timespec="seconds"))
    if date_max is not None:
        conditions.append(f"{date_pfx}DATEANDTIME <= ?")
        params.append(pd.Timestamp(date_max).isoformat(sep=" ", timespec="seconds"))
