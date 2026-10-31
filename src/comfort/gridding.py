"""Classes and functions for spatiotemporal gridding of parameter values.

Adapted from Jenniges (2025), doi:10.5281/zenodo.15827777
"""
from __future__ import annotations

__all__ = [
    "Grid",
    "average_duplicate_locations",
    "get_missing_value_info",
    "seasonal_mean",
]

import logging
import os
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from .util.columns import find_col, resolve_param_columns
from .util.optional import import_optional
from .util.types import Aggregation, GridMode, check_choice, check_timestamp
from .database.information import param_label, param_table_name

logger = logging.getLogger(__name__)


def _is_multiple(span: float, step: float) -> bool:
    """Return True if *span* is a whole number of *step*s, within floating-point tolerance.

    A plain ``span % step == 0`` fails for floats, e.g. ``1 % 0.1`` is ``0.0999...``.
    """
    n = span / step
    return abs(n - round(n)) < 1e-9


def _axis_cells(lo: float, hi: float, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the cell edges and cell centres of an axis from *lo* to *hi*. """
    n = round((hi - lo) / step)  # number of bins
    return lo + step * np.arange(n + 1), lo + step * (np.arange(n) + 0.5)


def _inclusive_last_edge(edges: np.ndarray) -> np.ndarray:
    """Return a float copy of *edges* whose last edge is adapted so that the last bin includes the maximum."""
    edges = np.asarray(edges, dtype=float).copy()  # copy the edges
    edges[-1] = np.nextafter(edges[-1], np.inf)  # increase last edge by the next floating point number
    return edges


def _cut_cells(values: pd.Series, edges: np.ndarray, labels: np.ndarray) -> pd.Series:
    """Bin *values* into left-closed cells ``[a, b)`` between *edges*, the maximum edge included and label them."""
    return pd.cut(values, bins=_inclusive_last_edge(edges), right=False, labels=labels)


class Grid:
    """A grid based on a bathymetry file, with or without a time axis.
    Lat/lon are cell-centred, time and depth are node-centred.

    The grid has a time axis if the time parameters (``time_min``, ``time_max``,
    ``mode``, ``dtime``) are given. Leave all of them as None for a purely spatial grid.

    ``mode`` selects how the time axis is built, in steps of ``dtime``:

    - ``'Y'``, ``'YM'``, ``'YD'``: Consecutive years, months or days starting at ``time_min``.
    - ``'M'``: Month of the year (01-12), pooling all years.
    - ``'D'``: Day of the month (01-31), pooling all months and years.
    - ``'MD'``: Month and day (MM-DD) of the leap year 2000, pooling all years.
    """

    def __init__(self, *, bathymetry_grid_path, lon_min=-180, lon_max=180, dlon=1,
                 lat_min=-90, lat_max=90, dlat=1,
                 z_min=None, z_max=None, dz=None, z_array=None,
                 time_min=None, time_max=None, mode: GridMode | None = None, dtime=None, selection=None,
                 lon_variable="lon", lat_variable="lat", depth_variable="elevation"):
        """
        Args:
            lon_min, lon_max, dlon: Longitude range and step [°].
            lat_min, lat_max, dlat: Latitude range and step [°].
            bathymetry_grid_path (str): Path to a bathymetry NetCDF file. Required.
            z_min, z_max, dz: Depth range and step [m]. Provide either dz or z_array.
            z_array (array-like): Explicit depth nodes [m]. Mutually exclusive with dz.
            time_min, time_max (str): Inclusive bounds of the time dimension as 'YYYY-MM-DD HH:MM:SS'.
            mode (str): Time grouping mode. One of 'Y', 'M', 'D', 'YM', 'YD', 'MD' (see class docstring).
            dtime (int): Number of time units per cell.
                time_min, time_max, mode and dtime must be given together, or all be
                None for a grid without time axis.
            selection (array-like): Boolean mask to sub-select the time array.
                Requires a time axis.
            lon_variable, lat_variable, depth_variable: Variable names in the NetCDF file.
        """
        # Init grid params
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.dlon = dlon
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.dlat = dlat
        self.z_array = np.sort(np.array(z_array)) if z_array is not None else None
        self.z_min = z_min if z_array is None else self.z_array.min()
        self.z_max = z_max if z_array is None else self.z_array.max()
        self.dz = dz
        self.time_min = time_min
        self.time_max = time_max
        self.mode = mode
        self.dtime = dtime
        self.selection = np.array(selection) if selection is not None else None
        self.bathymetry_grid_path = bathymetry_grid_path

        self._check_spatial_input()
        self._check_time_input()

        # Create grid
        self.time_array = self._create_time_array() if self.has_time else None
        self.grid = self._create_grid(lon_variable, lat_variable, depth_variable)

    @property
    def has_time(self) -> bool:
        """Whether this grid has a time axis (derived from *mode*)."""
        return self.mode is not None

    def _check_time_input(self) -> None:
        """Consistency checks for time input."""
        # Time parameters are all-or-nothing
        time_params = {"time_min": self.time_min, "time_max": self.time_max, "mode": self.mode, "dtime": self.dtime}
        missing = [param for param, val in time_params.items() if val is None]
        if len(missing) == len(time_params):
            # No time axis, so a time selection has nothing to select from
            if self.selection is not None:
                raise ValueError("selection requires a time axis (time_min, time_max, mode, dtime)")
            return
        if missing:
            raise ValueError(
                f"Grid parameters {missing} must not be None when a time axis is used. "
                "Set all of time_min, time_max, mode and dtime or none of them."
            )

        # Bounds must be in correct format
        check_timestamp(self.time_min, "time_min")
        check_timestamp(self.time_max, "time_max")

        # Time step >=1
        if self.dtime < 1:
            raise ValueError("dtime must be >= 1")

        # Valid modes
        check_choice(self.mode, GridMode, "mode")

        # Type of selection object
        if self.selection is not None and not isinstance(self.selection, (np.ndarray, list)):
            raise ValueError("selection must be a list or numpy array")

    @staticmethod
    def _iter_range(start, end, step):
        """Yield datetimes from start to end (exclusive) by relativedelta step."""
        from dateutil.parser import parse as _parse_dt
        current = _parse_dt(start) if isinstance(start, str) else start
        end_dt = _parse_dt(end) if isinstance(end, str) else end
        while current < end_dt:
            yield current
            current += step

    def _time_nodes(self) -> np.ndarray:
        """Create the time steps of the grid, before ``selection`` is applied."""
        time_array = []

        # Determine all valid dates for the desired time range
        if self.mode == "Y":
            time_array = [v.strftime("%Y-%m-%d %H:%M:%S")
                          for v in self._iter_range(self.time_min, self.time_max, relativedelta(years=+self.dtime))]
        elif self.mode == "YM":
            time_array = [v.strftime("%Y-%m-%d %H:%M:%S")
                          for v in self._iter_range(self.time_min, self.time_max, relativedelta(months=+self.dtime))]
        elif self.mode == "YD":
            time_array = [v.strftime("%Y-%m-%d %H:%M:%S")
                          for v in self._iter_range(self.time_min, self.time_max, relativedelta(days=+int(self.dtime)))]
        elif self.mode == "M":
            time_array = np.char.zfill(np.arange(1, 13, self.dtime).astype(str), 2)
        elif self.mode == "MD":
            time_array = [v.strftime("%m-%d %H:%M:%S")
                          for v in self._iter_range("2000-01-01", "2001-01-01", relativedelta(days=+int(self.dtime)))]
        elif self.mode == "D":
            time_array = np.char.zfill(np.arange(1, 32, self.dtime).astype(str), 2)

        return np.array(time_array)

    def _create_time_array(self) -> np.ndarray:
        """Create the time array containing the (selected) time steps for the grid."""
        time_array = self._time_nodes()  # get the nodes

        # Perform selection
        if self.selection is not None:
            if time_array.shape != self.selection.shape:
                raise ValueError("selection shape must match time_array shape")
            time_array = time_array[self.selection.astype(bool)]

        return time_array

    def _open_bathymetry(self, bathymetry_grid_path: str, lon_variable: str,
                         lat_variable: str, depth_variable: str):
        """Open, normalise and load a bathymetry NetCDF file into memory, then close it."""
        xr = import_optional("xarray", "grid", "gridding")

        # Check if filepath is valid
        if bathymetry_grid_path is None or not os.path.isfile(bathymetry_grid_path):
            raise FileNotFoundError(f"Bathymetry file not found: {bathymetry_grid_path}")

        # Load bathymetry dataset
        with xr.open_dataset(bathymetry_grid_path) as ds:
            ds = ds.rename({lon_variable: "LONGITUDE", lat_variable: "LATITUDE", depth_variable: "LEV_M"})
            ds["LEV_M"] = ds["LEV_M"] * -1  # Positive z points downward (COMFORT convention)
            ds.load()
        return ds

    def _check_spatial_input(self) -> None:
        """Validate spatial grid parameters; raise ValueError on invalid input."""
        required = {
            "lon_min": self.lon_min, "lon_max": self.lon_max, "dlon": self.dlon,
            "lat_min": self.lat_min, "lat_max": self.lat_max, "dlat": self.dlat,
            "z_min": self.z_min, "z_max": self.z_max,
        }

        # Check if all params are specified
        for param, val in required.items():
            if val is None:
                raise ValueError(f"Grid parameter {param!r} must not be None")

        # Check min/max values for longitude and latitude
        if not (-180 <= self.lon_min <= 180) or not (-180 <= self.lon_max <= 180):
            raise ValueError("lon_min and lon_max must be in [-180, 180]")
        if not (-90 <= self.lat_min <= 90) or not (-90 <= self.lat_max <= 90):
            raise ValueError("lat_min and lat_max must be in [-90, 90]")

        # Check min/max values for depth
        if not (0 <= self.z_min <= 12000) or not (0 <= self.z_max <= 12000):
            raise ValueError("z_min and z_max must be in [0, 12000]")

        # The ranges must be non-empty (np.arange would silently return an empty grid otherwise)
        if not self.lon_min < self.lon_max:
            raise ValueError("lon_min must be smaller than lon_max")
        if not self.lat_min < self.lat_max:
            raise ValueError("lat_min must be smaller than lat_max")
        if not self.z_min < self.z_max:
            raise ValueError("z_min must be smaller than z_max")

        # Steps must be positive
        if self.dlon <= 0 or self.dlat <= 0:
            raise ValueError("dlon and dlat must be > 0")

        # Check depth (specified as z_array)
        if self.z_array is None:
            if self.dz is None:
                raise ValueError("dz must be provided when z_array is None")
            if self.dz <= 0:
                raise ValueError("dz must be > 0")

            # For a regular grid, we need complete grid cells
            if not _is_multiple(self.lon_max - self.lon_min, self.dlon):
                raise ValueError("Longitude range must be evenly divisible by dlon")
            if not _is_multiple(self.lat_max - self.lat_min, self.dlat):
                raise ValueError("Latitude range must be evenly divisible by dlat")
            if not _is_multiple(self.z_max - self.z_min, self.dz):
                raise ValueError("Depth range must be evenly divisible by dz")
        else:
            if self.dz is not None:
                raise ValueError("dz must be None when z_array is provided")

    def _make_z_arr(self) -> np.ndarray:
        """Return the depth array to use for the water mask (excluding the last boundary node)."""
        if self.z_array is None:
            return _axis_cells(self.z_min, self.z_max, self.dz)[0][:-1]
        return self.z_array[:-1]

    def _water_mask(self, ds, lon, lat):
        """Return a xarray DataArray that is True where there is water (depth > 0)."""
        xr = import_optional("xarray", "grid", "gridding")

        # Wrap depth levels into xarray
        z_arr = self._make_z_arr()
        z = xr.DataArray(z_arr, dims="LEV_M", coords={"LEV_M": z_arr}, name="LEV_M")

        # Assign
        z_at_grid = ds.LEV_M.sel(LONGITUDE=lon, LATITUDE=lat, method="nearest")
        z_at_grid.coords["LONGITUDE"] = lon
        z_at_grid.coords["LATITUDE"] = lat
        return (z <= z_at_grid).rename("water")

    def _map_lon_lat_depth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bin LONGITUDE, LATITUDE and LEV_M columns in df to grid cell centres."""
        # Map longitude values into grid cells (labels = cell centres, maximum included)
        lon_edges, lon_centres = _axis_cells(self.lon_min, self.lon_max, self.dlon)
        df["LONGITUDE"] = _cut_cells(df["LONGITUDE"], lon_edges, lon_centres)

        # Map latitude values
        lat_edges, lat_centres = _axis_cells(self.lat_min, self.lat_max, self.dlat)
        df["LATITUDE"] = _cut_cells(df["LATITUDE"], lat_edges, lat_centres)

        # Map depth values (labels = the node at the top of each depth bin, maximum included)
        if self.z_array is not None:
            z_edges = self.z_array.astype(float)
        else:
            z_edges = _axis_cells(self.z_min, self.z_max, self.dz)[0]
        df["LEV_M"] = _cut_cells(df["LEV_M"], z_edges, z_edges[:-1])

        return df

    @staticmethod
    def _flatten_bin_agg(grouped: pd.DataFrame, value_col: str, agg) -> pd.DataFrame:
        """Flatten the (column, statistic) column names pandas produces."""
        # Single aggregation (e.g. agg="mean") already has flat columns
        if not isinstance(agg, (list, tuple)):
            return grouped

        # Rename aggregated columns to the name of their statistic
        grouped = grouped.copy()
        grouped.columns = [
            c[0] if c[0] != value_col else c[1]
            for c in grouped.columns.to_flat_index()
        ]

        # Rename the first requested stat's column back to value_col
        first_stat = agg[0] if isinstance(agg[0], str) else agg[0].__name__
        return grouped.rename(columns={first_stat: value_col})

    def bin_space(self, df: pd.DataFrame, value_col: str | None = None, agg: Aggregation | None = None) -> pd.DataFrame:
        """Bin LONGITUDE/LATITUDE/LEV_M in *df* onto this grid's cell centres.

        Works on data already in memory (e.g. after ``load_comfort``).

        Args:
            df (pandas.DataFrame): Must contain LATITUDE, LONGITUDE, LEV_M.
            value_col (str): Column to aggregate. Required if *agg* is given;
                omit both to get back every row of *df* with its coordinates
                snapped to the grid (e.g. to run a custom aggregation like
                ``seasonal_mean`` before further binning).
            agg: Aggregation passed to ``DataFrame.agg``. A string ("mean"),
                a list of strings/callables, or a callable.
        Returns:
            pandas.DataFrame: One row per occupied cell if *agg* is given,
                otherwise one row per input row.
        """
        # Resolve column names (insensitive to casing)
        value_col = find_col(df, value_col)

        # Snap LATITUDE/LONGITUDE/LEV_M onto this grid's cell centres
        df = self._map_lon_lat_depth(df.copy())

        # Drop rows outside the grid's extent
        df = df.dropna(subset=["LATITUDE", "LONGITUDE", "LEV_M"])
        df = df.astype({"LATITUDE": float, "LONGITUDE": float, "LEV_M": float})
        if agg is None:
            # No aggregation requested, return every row with snapped coordinates
            return df

        # Group rows that snapped onto the same cell and aggregate value_col
        grouped = df.groupby(["LATITUDE", "LONGITUDE", "LEV_M"], as_index=False).agg({value_col: agg})
        grouped = self._flatten_bin_agg(grouped, value_col, agg)
        grouped[value_col] = grouped[value_col].astype(float)
        return grouped

    def map_dataframes(self, dfs: dict[str, pd.DataFrame], agg: Aggregation = "mean",
                       bin_fn=None, dropping_land_cells: bool = True) -> pd.DataFrame:
        """Turn a dict of per-parameter DataFrames into one wide grid table.

        The full workflow is two calls: create the grid, then map data onto it::

            grid = Grid(...)
            df_wide = grid.map_dataframes(raw)

        Args:
            dfs (dict[str, pandas.DataFrame]): Parameter name -> DataFrame,
                Each DataFrame must contain LATITUDE, LONGITUDE, LEV_M
                (+ DATEANDTIME for a grid with time axis) and a value column named
                after its parameter key. Insensitive to casing and ``P_`` prefix.
            agg: Aggregation passed to ``self.bin(df, value_col, agg=agg)`` for
                every parameter. A string ("mean"), a list, a callable, or
                (on a grid with time axis) ``"seasonal_mean"`` for monthly-first annual
                averaging before binning onto the grid's time steps. Ignored
                when *bin_fn* is given.
            bin_fn (callable): ``(df, value_col) -> DataFrame`` used instead of
                the default ``self.bin(df, value_col, agg=agg)``, for any
                per-parameter aggregation not expressible via *agg*.
            dropping_land_cells (bool): Drop grid cells that are on land and
                never carry a value for any parameter. Default True.
        Returns:
            pandas.DataFrame: This grid's template merged with one ``P_<name>``
                column per input parameter.
        """
        # Custom aggregation function
        if bin_fn is None:
            def bin_fn(df, value_col):
                return self.bin(df, value_col, agg=agg)

        # Start from the full grid template
        df_wide = self.grid.copy()
        merge_cols = [c for c in ("LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME")
                      if c in df_wide.columns]

        # Successively merge each binned parameter onto the grid
        for parameter, df in dfs.items():
            # Rename the value column to P_<parameter> (casing and P_ prefix of the column do not matter)
            col = param_table_name(parameter)
            df = df.rename(columns={resolve_param_columns(df, parameter)[0]: col})
            # Binning
            binned = bin_fn(df, col)
            # Left-merge on grid
            df_wide = df_wide.merge(binned, on=merge_cols, how="left")

        if dropping_land_cells:
            # Drop cells that are on land and have no data for any parameter
            n_before = len(df_wide)
            df_wide = _drop_land_cells(df_wide).reset_index(drop=True)
            logger.info("map_dataframes: dropped %d empty land cells (%d -> %d rows)",
                        n_before - len(df_wide), n_before, len(df_wide))
        return df_wide

    def _create_grid(self, lon_variable: str, lat_variable: str,
                      depth_variable: str) -> pd.DataFrame:
        """Creating a grid based on:
        https://github.com/willirath/geomar-open-hacky-hour-2021-04/blob/main/2022-06-23/etopo05_to_grid.ipynb
        grid must be nc file, format: latitude, longitude, depth
        """
        xr = import_optional("xarray", "grid", "gridding")

        # Load bathymetry
        ds = self._open_bathymetry(self.bathymetry_grid_path, lon_variable, lat_variable, depth_variable)

        # Define cell-centered lon/lat values
        lon = xr.DataArray(_axis_cells(self.lon_min, self.lon_max, self.dlon)[1], dims="LONGITUDE")
        lat = xr.DataArray(_axis_cells(self.lat_min, self.lat_max, self.dlat)[1], dims="LATITUDE")

        # Add water mask (and time dimension, if the grid has one)
        water_filled = self._water_mask(ds, lon, lat)
        if self.has_time:
            water_filled = water_filled.expand_dims(DATEANDTIME=self.time_array)

        # Convert to df
        df = water_filled.to_dataframe().reset_index()

        dtypes = {"LEV_M": float, "LONGITUDE": float, "LATITUDE": float, "water": bool}
        if self.has_time:
            dtypes["DATEANDTIME"] = str
        return df.astype(dtypes)

    def _filter_time_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter *df* for ``[time_min, time_max]``."""
        dates = pd.to_datetime(df["DATEANDTIME"])
        in_window = (dates >= pd.Timestamp(self.time_min)) & (dates <= pd.Timestamp(self.time_max))
        return df[in_window].copy()

    def _map_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bin the DATEANDTIME column, according to self.mode."""
        if self.mode in ("Y", "YM", "YD"):
            # Bins from all nodes (also deselected ones), the last edge just after time_max so that it is included
            last_edge = pd.Timestamp(self.time_max) + pd.Timedelta(nanoseconds=1)
            time_bins = pd.DatetimeIndex(self._time_nodes()).append(pd.DatetimeIndex([last_edge]))
            df["DATEANDTIME"] = pd.cut(
                pd.to_datetime(df["DATEANDTIME"]),
                bins=time_bins, labels=time_bins[:-1], right=False,
            )
            df["DATEANDTIME"] = df["DATEANDTIME"].astype("datetime64[ns]").dt.strftime("%Y-%m-%d %H:%M:%S")
        elif self.mode == "M":
            # Snap the month to the first month of its bin
            dates = df["DATEANDTIME"].astype("datetime64[ns]")
            bin_index = (dates.dt.month - 1) // self.dtime
            node = bin_index * self.dtime + 1  # get the grid node, i.e. the month
            df["DATEANDTIME"] = node.astype("Int64").astype(str).str.zfill(2)
        elif self.mode == "MD":
            # Snap the day-month bin
            dates = df["DATEANDTIME"].astype("datetime64[ns]")
            # Days index of the year (1 Jan = 0)
            day_index = dates.dt.day_of_year - 1
            # Check if the day is from a non-leap year after February
            after_feb_non_leap = (~dates.dt.is_leap_year) & (dates.dt.month > 2)
            # Add one day for non-leap years after February (to align with leap reference year)
            day_index = day_index + after_feb_non_leap.astype(int)
            # Start of the dtime-day bin the day falls into
            bin_index = day_index // self.dtime
            node_day = bin_index * self.dtime
            # Turn the day number back into a date in 2000 (midnight, so the time of day is dropped)
            node = pd.Timestamp("2000-01-01") + pd.to_timedelta(node_day, unit="D")
            df["DATEANDTIME"] = node.dt.strftime("%m-%d %H:%M:%S")
        elif self.mode == "D":
            # Snap the day to the first day of its bin
            dates = df["DATEANDTIME"].astype("datetime64[ns]")
            bin_idx = (dates.dt.day - 1) // self.dtime
            node = bin_idx * self.dtime + 1
            df["DATEANDTIME"] = node.astype("Int64").astype(str).str.zfill(2)

        return df

    def bin_time(self, df: pd.DataFrame, value_col: str, agg: Aggregation = "mean") -> pd.DataFrame:
        """Bin DATEANDTIME in *df* onto this grid's time steps and aggregate *value_col*.

        LATITUDE/LONGITUDE/LEV_M must already be snapped to grid cell centres (e.g. by calling ``bin_space`` first).
        Rows are then grouped by cell + time bin together.

        Args:
            df (pandas.DataFrame): Must contain LATITUDE, LONGITUDE, LEV_M, DATEANDTIME and *value_col*.
            value_col (str): Column to aggregate.
            agg: Aggregation passed to ``DataFrame.agg``.
        Returns:
            pandas.DataFrame: One row per occupied cell and time step.
        """
        # Resolve column names (insensitive to casing)
        value_col = find_col(df, value_col)

        if not self.has_time:
            raise ValueError("bin_time requires a grid with time axis (time_min, time_max, mode, dtime)")

        # Drop rows outside the time window, then grid DATEANDTIME
        df = self._map_time(self._filter_time_window(df))

        # Drop rows that fell into a deselected time step
        if self.selection is not None:
            df = df[df["DATEANDTIME"].isin(self.time_array)]

        # Group rows in the same cell and time bin and aggregate value_col
        grouped = df.groupby(
            ["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"], as_index=False,
        ).agg({value_col: agg})
        grouped = self._flatten_bin_agg(grouped, value_col, agg)
        grouped[value_col] = grouped[value_col].astype(float)
        grouped["DATEANDTIME"] = grouped["DATEANDTIME"].astype(str)
        return grouped

    def bin(self, df: pd.DataFrame, value_col: str, agg: Aggregation = "mean") -> pd.DataFrame:
        """Bin *df* onto this grid's spatial cells and time steps, aggregating
        *value_col* per cell. Usable for data already in memory (e.g. after ``load_comfort``).

        Args:
            df (pandas.DataFrame): Must contain LATITUDE, LONGITUDE, LEV_M, *value_col* (+ DATEANDTIME for a grid with time axis).
            value_col (str): Column to aggregate.
            agg: Aggregation passed to ``DataFrame.agg``. The special
                value ``"seasonal_mean"`` (grid with time axis only) runs
                monthly-first annual averaging before binning onto this
                grid's time steps (common for yearly gridding).
        Returns:
            pandas.DataFrame: One row per occupied cell (and time step).
        """
        # Resolve column names (insensitive to casing)
        value_col = find_col(df, value_col)

        if not self.has_time:
            # No-time grid
            if agg == "seasonal_mean":
                raise ValueError("agg='seasonal_mean' requires a grid with time axis")
            return self.bin_space(df, value_col, agg)
        if agg == "seasonal_mean":
            # Apply seasonal-mean to time grid
            return self._bin_seasonal_mean(df, value_col)
        return self.bin_time(self.bin_space(df), value_col, agg)

    def _bin_seasonal_mean(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Average value_col per grid cell per year, monthly-first, then bin onto this grid's time steps.
        Requires ``mode="Y"`` and a ``time_min`` on 1 January.

        Implements ``agg="seasonal_mean"`` on ``bin``. A plain annual mean lets
        months with more observations dominate. Averaging month-by-month first,
        then averaging those (at most 12) monthly means, corrects for that.
        """
        # Check if mode="Y" and time_min is on 1 January
        start = pd.Timestamp(self.time_min)
        if self.mode != "Y" or start != pd.Timestamp(year=start.year, month=1, day=1):
            raise ValueError("agg='seasonal_mean' requires mode='Y' and a time_min on 1 January")

        # Grid space (inside the time window), then compute the monthly-first annual mean per cell
        annual = seasonal_mean(
            self.bin_space(self._filter_time_window(df)), value_col=value_col,
            group_cols=["LATITUDE", "LONGITUDE", "LEV_M"],
        )

        # Convert YEAR returned by seasonal_mean into a time stamp
        annual["DATEANDTIME"] = annual["YEAR"].astype(int).astype(str) + "-01-01 00:00:00"
        annual = annual.drop(columns=["YEAR"])
        annual = annual[pd.to_datetime(annual["DATEANDTIME"]) < pd.Timestamp(self.time_max)]  # filter for max timestamp
        return self.bin_time(annual, value_col, agg="mean")


def average_duplicate_locations(
    raw: dict[str, pd.DataFrame],
    other_params: list[str] | None = None,
    temperature_col: str = "TEMPERATURE",
    loc_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Average duplicate location records since a COMFORT station
    can carry more than one measurement of the same parameter
    at an identical (lat, lon, depth, time).

    Locations are matched by exact equality of the *loc_cols* values (no tolerance, time is matched to the second).

    Args:
        raw (dict[str, pandas.DataFrame]): Per-parameter DataFrames.
        other_params (list[str]): Parameters to average over identical locations
            (simple mean). Defaults to all keys in *raw* other than *temperature_col*.
        temperature_col (str): Column for temperature. Default ``"TEMPERATURE"``.
        loc_cols (list[str]): Columns identifying a unique location. Default
            ``["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"]``.
    Returns:
        dict[str, pandas.DataFrame]: One DataFrame per parameter (plus
            ``LEV_DBAR`` for *temperature_col*, kept for downstream pressure
            use e.g. TEOS-10 conversions).
    """
    # Identify location and parameter columns (other than temperature)
    if loc_cols is None:
        loc_cols = ["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"]
    if other_params is None:
        other_params = [p for p in raw if p != temperature_col]

    # Average temperature and keep pressure column
    averaged = {}
    if temperature_col in raw:
        t_avg = raw[temperature_col].groupby(loc_cols, as_index=False).agg(
            {temperature_col: "mean", "LEV_DBAR": "mean"})
        logger.info("average_duplicate_locations: %s %d -> %d",
                     temperature_col, len(raw[temperature_col]), len(t_avg))
        averaged[temperature_col] = t_avg

    # Average other parameters
    for param in other_params:
        if param not in raw:
            continue
        avg = raw[param].groupby(loc_cols, as_index=False).agg({param: "mean"})
        logger.info("average_duplicate_locations: %s %d -> %d", param, len(raw[param]), len(avg))
        averaged[param] = avg

    return averaged


def seasonal_mean(
    df: pd.DataFrame,
    value_col: str = "VAL",
    date_col: str = "DATEANDTIME",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Annual mean with monthly averaging to correct seasonal sampling bias.

    Args:
        df (pandas.DataFrame): with value, date and spatial/group columns.
        value_col (str): Column containing measured values.
        date_col (str): Column containing datetime values.
        group_cols (list<str>): Columns that identify spatial cells. Defaults to
            ``["LATITUDE", "LONGITUDE", "LEV_M"]``.

    Returns:
        pd.DataFrame with columns ``group_cols + ["YEAR", value_col]``,
        one row per group per year.
    """
    # Resolve column names (insensitive to casing)
    value_col = find_col(df, value_col)
    date_col = find_col(df, date_col)

    if group_cols is None:
        group_cols = ["LATITUDE", "LONGITUDE", "LEV_M"]

    # Extract year and month data
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col])
    temp["_YEAR"] = temp[date_col].dt.year
    temp["_MONTH"] = temp[date_col].dt.month

    # Monthly mean per cell per year
    monthly = temp.groupby(
        group_cols + ["_YEAR", "_MONTH"], as_index=False,
    ).agg({value_col: "mean"})

    # Annual mean as average of monthly means
    annual = monthly.groupby(
        group_cols + ["_YEAR"], as_index=False,
    ).agg({value_col: "mean"})

    return annual.rename(columns={"_YEAR": "YEAR"})


def _drop_land_cells(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Remove grid cells that are on land (water=False) and never have a parameter value.

    A (LATITUDE, LONGITUDE, LEV_M) cell is dropped only if it is on land AND all
    P_* columns are NaN for every time step. This keeps the grid identical across
    time steps at a given depth: a cell that has data in at least one time step
    is kept in all time steps.

    Args:
        df_wide (pandas.DataFrame): Wide grid table with a 'water' column.
    Returns:
        pandas.DataFrame
    """
    temp = df_wide.copy()

    # Get parameter columns
    param_tables = [x for x in temp.columns if x.startswith("P_")]
    if not param_tables:
        return temp

    # A cell ever has data if any parameter is non-NaN in any time step
    has_data = temp[param_tables].notna().any(axis=1)
    group_cols = ["LATITUDE", "LONGITUDE", "LEV_M"]
    ever_has_data = has_data.groupby([temp[c] for c in group_cols]).transform("any")

    # Drop cells that are on land and never have data, at any depth or time
    return temp[temp["water"] | ever_has_data]


def get_missing_value_info(df_wide: pd.DataFrame, parameters: list[str] | None = None) -> pd.DataFrame:
    """Count missing values per parameter in a wide grid table.

    Every row counts as a grid cell, so land cells without data raise the
    percentages unless they were dropped (see ``dropping_land_cells`` in
    :meth:`Grid.map_dataframes`).

    Args:
        df_wide (pandas.DataFrame): Wide grid table with ``P_*`` columns.
        parameters (list[str]): Parameters to report, insensitive to casing and ``P_`` prefix.
            Default ``None`` reports every ``P_*`` column.
    Returns:
        pandas.DataFrame: Columns ``'parameter'`` (without ``P_``),
        ``'absolute'`` (number of missing cells) and ``'relative'`` (percent),
        sorted by ``'relative'``.
    """
    # Get parameter columns
    if parameters is None:
        param_tables = [x for x in df_wide.columns if x.startswith("P_")]
    else:
        param_tables = resolve_param_columns(df_wide, parameters)

    # Compute absolute missingness
    rows = [{"parameter": param_label(p), "absolute": df_wide[p].isna().sum()} for p in param_tables]
    num_nulls = pd.DataFrame(rows, columns=["parameter", "absolute"])

    # Compute relative missingness
    num_grid_cells = len(df_wide)
    if num_grid_cells == 0:
        num_nulls["relative"] = 0.0
    else:
        num_nulls["relative"] = num_nulls["absolute"] / num_grid_cells * 100
        num_nulls = num_nulls.sort_values("relative")
    return num_nulls
