"""Tests for comfort.gridding."""
import itertools
import os
import sqlite3
import numpy as np
import pandas as pd
import pytest

_has_xarray = True
try:
    import xarray
except ImportError:
    _has_xarray = False

from comfort.database.information import does_table_exist, get_table_as_df
from comfort.gridding import (
    Grid,
    GridManager,
    generate_grid_id,
    average_duplicate_locations,
    create_wide_table_online,
    drop_land_cells,
    get_missing_value_info_offline,
    get_missing_value_info_offline_per_param,
    get_missing_value_info_per_param,
    load_wide_table,
    seasonal_mean,
)


class TestSeasonalMean:
    def _make_df(self):
        """One cell, two years, one observation per month."""
        rows = []
        for year in [2000, 2001]:
            for month in range(1, 13):
                rows.append({
                    "LATITUDE": 10.0, "LONGITUDE": 20.0, "LEV_M": 0.0,
                    "DATEANDTIME": f"{year}-{month:02d}-15",
                    "VAL": float(month),
                })
        return pd.DataFrame(rows)

    # Output carries YEAR and the value column, no leftover helper columns
    def test_basic_output_columns(self):
        df = self._make_df()
        result = seasonal_mean(df)
        assert "YEAR" in result.columns
        assert "VAL" in result.columns
        assert "_YEAR" not in result.columns
        assert "_MONTH" not in result.columns

    # Baseline case: One obs/month, so annual mean is just the mean of monthly means
    def test_uniform_coverage_baseline(self):
        df = self._make_df()
        result = seasonal_mean(df)
        expected = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) / 12.0
        for _, row in result.iterrows():
            assert np.isclose(row["VAL"], expected)

    # Summer-heavy sampling should not bias the annual mean
    def test_seasonal_bias_correction(self):
        rows = []
        for month in range(1, 13):
            n_obs = 100 if month in (6, 7, 8) else 1
            for _ in range(n_obs):
                rows.append({
                    "LATITUDE": 10.0, "LONGITUDE": 20.0, "LEV_M": 0.0,
                    "DATEANDTIME": f"2000-{month:02d}-15",
                    "VAL": 20.0 if month in (6, 7, 8) else 5.0,
                })
        df = pd.DataFrame(rows)
        result = seasonal_mean(df)

        # Monthly-first: Each month gets equal weight
        expected = (9 * 5.0 + 3 * 20.0) / 12.0
        assert np.isclose(result["VAL"].iloc[0], expected)

        # Naive mean would be biased towards summer (300 x 20 vs 9 x 5)
        naive = df["VAL"].mean()
        assert not np.isclose(naive, expected)

    # group_cols lets the caller group by columns other than LATITUDE/LONGITUDE/LEV_M
    def test_custom_group_cols(self):
        df = pd.DataFrame({
            "LAT": [10.0] * 12,
            "LON": [20.0] * 12,
            "DATEANDTIME": [f"2000-{m:02d}-15" for m in range(1, 13)],
            "VAL": [float(m) for m in range(1, 13)],
        })
        result = seasonal_mean(df, group_cols=["LAT", "LON"])
        assert len(result) == 1
        assert "YEAR" in result.columns

    # Each year gets its own averaged row
    def test_multiple_years(self):
        df = self._make_df()
        result = seasonal_mean(df)
        assert result["YEAR"].nunique() == 2
        assert set(result["YEAR"]) == {2000, 2001}

    # value_col lets the caller average a differently-named column
    def test_custom_value_col(self):
        df = self._make_df().rename(columns={"VAL": "TEMP"})
        result = seasonal_mean(df, value_col="TEMP")
        assert "TEMP" in result.columns


class TestAverageDuplicateLocations:
    def _loc(self, **overrides):
        base = {"LATITUDE": 10.0, "LONGITUDE": 20.0, "LEV_M": 1000.0, "LEV_DBAR": 1009.0, "DATEANDTIME": "2000-01-15"}
        base.update(overrides)
        return base

    # Two temperature readings at the same location are averaged into one
    def test_averages_duplicate_temperature(self):
        raw = {"TEMPERATURE": pd.DataFrame([
            {**self._loc(), "TEMPERATURE": 4.0},
            {**self._loc(), "TEMPERATURE": 6.0},
        ])}
        result = average_duplicate_locations(raw)
        assert len(result["TEMPERATURE"]) == 1
        assert result["TEMPERATURE"]["TEMPERATURE"].iloc[0] == pytest.approx(5.0)

    # Salinity is averaged the same way as any other parameter
    def test_averages_duplicate_salinity(self):
        raw = {"SALINITY": pd.DataFrame([
            {**self._loc(), "SALINITY": 34.0},
            {**self._loc(), "SALINITY": 36.0},
        ])}
        result = average_duplicate_locations(raw)
        assert len(result["SALINITY"]) == 1
        assert result["SALINITY"]["SALINITY"].iloc[0] == pytest.approx(35.0)

    # other_params=None defaults to every key in raw besides TEMPERATURE
    def test_other_params_defaults_to_remaining_keys(self):
        raw = {
            "TEMPERATURE": pd.DataFrame([{**self._loc(), "TEMPERATURE": 4.0}]),
            "SALINITY": pd.DataFrame([{**self._loc(), "SALINITY": 35.0}]),
            "OXYGEN": pd.DataFrame([
                {**self._loc(), "OXYGEN": 200.0},
                {**self._loc(), "OXYGEN": 220.0},
            ]),
        }
        result = average_duplicate_locations(raw)
        assert "OXYGEN" in result
        assert result["OXYGEN"]["OXYGEN"].iloc[0] == pytest.approx(210.0)

    # An explicit other_params list drops any key not named in it
    def test_other_params_explicit_list_ignores_unlisted_keys(self):
        raw = {
            "TEMPERATURE": pd.DataFrame([{**self._loc(), "TEMPERATURE": 4.0}]),
            "OXYGEN": pd.DataFrame([{**self._loc(), "OXYGEN": 200.0}]),
            "NITRATE": pd.DataFrame([{**self._loc(), "NITRATE": 5.0}]),
        }
        result = average_duplicate_locations(raw, other_params=["OXYGEN"])
        assert "OXYGEN" in result
        assert "NITRATE" not in result


class TestDropLandCells:
    # --- Grid behaviour (with time) -------------------------------------------- #
    def _make_wide(self, water, val):
        """Each row is a distinct grid cell (distinct LATITUDE), single time step."""
        return pd.DataFrame({
            "LATITUDE": [float(i) for i in range(len(water))],
            "LONGITUDE": [0.0] * len(water),
            "LEV_M": [0.0] * len(water),
            "DATEANDTIME": ["2000-01-01"] * len(water),
            "water": water,
            "P_NITRATE": val,
        })

    # A land cell with no parameter value at all is dropped
    def test_land_with_no_value_is_dropped(self):
        df = self._make_wide([False, True], [None, 1.0])
        result = drop_land_cells(df)
        assert len(result) == 1
        assert result["water"].iloc[0]

    # A land cell that does have a value is kept despite water=False
    def test_land_with_value_is_kept(self):
        df = self._make_wide([False, True], [5.0, 1.0])
        result = drop_land_cells(df)
        assert len(result) == 2

    # A grid with no land cells at all passes through unchanged
    def test_all_water_unchanged(self):
        df = self._make_wide([True, True], [1.0, 2.0])
        result = drop_land_cells(df)
        assert len(result) == 2

    # A land cell with data in at least one time step keeps the cell in every time step
    # (grid stays identical across time)
    def test_land_cell_with_value_in_any_time_step_kept_in_all(self):
        df = pd.DataFrame({
            "LATITUDE":    [0.0, 1.0, 2.0, 2.0, 0.0, 1.0],
            "LONGITUDE":   [0.0] * 6,
            "LEV_M":       [0.0] * 6,
            "DATEANDTIME": ["2000", "2000", "2000", "2001", "2001", "2001"],
            "water":       [False] * 6,
            "P_NITRATE":   [None, 5.0, None, 7.0, None, None],
        })
        result = drop_land_cells(df)

        # Cell lat=0.0 never has data -> dropped in both years
        assert not ((result["LATITUDE"] == 0.0)).any()

        # Cells lat=1.0 and lat=2.0 have data in some year -> kept in both years
        for lat in (1.0, 2.0):
            assert set(result.loc[result["LATITUDE"] == lat, "DATEANDTIME"]) == {"2000", "2001"}

        # Grid cell count must be identical across years
        counts = result.groupby("DATEANDTIME").size()
        assert counts["2000"] == counts["2001"]

    # --- Grid without time axis ------------------------------------------------ #
    def _make_wide_no_time(self, water, val):
        """Each row is a distinct grid cell (distinct LATITUDE)."""
        return pd.DataFrame({
            "LATITUDE": [float(i) for i in range(len(water))],
            "LONGITUDE": [0.0] * len(water),
            "LEV_M": [0.0] * len(water),
            "water": water,
            "P_NITRATE": val,
        })

    # A land cell with no parameter value at all is dropped
    def test_no_time_land_without_value_dropped(self):
        df = self._make_wide_no_time([False, True], [None, 1.0])
        result = drop_land_cells(df)
        assert len(result) == 1
        assert result["water"].iloc[0]

    # A land cell that does have a value is kept despite water=False
    def test_no_time_land_with_value_kept(self):
        df = self._make_wide_no_time([False, True], [5.0, 1.0])
        result = drop_land_cells(df)
        assert len(result) == 2

    # A grid with no land cells at all passes through unchanged
    def test_no_time_all_water_unchanged(self):
        df = self._make_wide_no_time([True, True], [1.0, 2.0])
        result = drop_land_cells(df)
        assert len(result) == 2

    # "Ever has data" is evaluated per (lat, lon, depth) cell, across depths too
    def test_no_time_multiple_depths(self):
        df = pd.DataFrame({
            "LATITUDE": [0.5, 0.5, 1.5, 1.5],
            "LONGITUDE": [0.5, 0.5, 0.5, 0.5],
            "LEV_M": [0.0, 100.0, 0.0, 100.0],
            "water": [False, False, True, True],
            "P_NITRATE": [3.0, None, 1.0, 2.0],
        })
        result = drop_land_cells(df)
        assert len(result) == 3
        dropped = set(df.index) - set(result.index)
        assert dropped == {1}


class TestGetMissingValueInfoOffline:
    def _make_wide(self, values):
        return pd.DataFrame({
            "P_NITRATE": values,
            "LATITUDE": range(len(values)),
        })

    # No missing values gives absolute=0
    def test_no_nulls(self):
        df = self._make_wide([1.0, 2.0, 3.0])
        result = get_missing_value_info_offline(df)
        assert result.loc[result["parameter"] == "P_NITRATE", "absolute"].iloc[0] == 0

    # Every value missing gives relative=100%
    def test_all_nulls(self):
        df = self._make_wide([None, None])
        result = get_missing_value_info_offline(df)
        assert result.loc[result["parameter"] == "P_NITRATE", "relative"].iloc[0] == 100.0

    # An empty DataFrame does not divide by zero
    def test_empty_dataframe(self):
        df = pd.DataFrame({"P_NITRATE": pd.Series([], dtype=float)})
        result = get_missing_value_info_offline(df)
        assert result.loc[result["parameter"] == "P_NITRATE", "relative"].iloc[0] == 0.0


@pytest.mark.skipif(not _has_xarray, reason="xarray not installed ([grid] extra)")
class TestGridValidation:
    """Grid constructor must raise an error for invalid parameters."""

    _COMMON_PARAMS = dict(
        lat_min=-90, lat_max=90, dlat=1,
        lon_min=-180, lon_max=180, dlon=1,
        z_min=0, z_max=1000, dz=100,
        time_min="2000-01-01 00:00:00", time_max="2001-01-01 00:00:00",
        mode="Y", dtime=1,
        bathymetry_grid_path="/nonexistent/path.nc",
    )

    # A missing bathymetry file is a FileNotFoundError, not a silent failure
    def test_missing_bathymetry_raises(self):
        with pytest.raises(FileNotFoundError):
            Grid(**self._COMMON_PARAMS)

    # An unrecognised time mode is rejected up front
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            Grid(**{**self._COMMON_PARAMS, "mode": "INVALID", "bathymetry_grid_path": "/nonexistent/path.nc"})

    # The time parameters are all-or-nothing
    @pytest.mark.parametrize("missing", ["time_min", "time_max", "mode", "dtime"])
    def test_partial_time_parameters_raise(self, missing):
        with pytest.raises(ValueError, match=missing):
            Grid(**{**self._COMMON_PARAMS, missing: None})

    # Grid is accepted without time parameters (then fails on the missing bathymetry file)
    def test_no_time_parameters_accepted(self):
        params = {**self._COMMON_PARAMS, "time_min": None, "time_max": None, "mode": None, "dtime": None}
        with pytest.raises(FileNotFoundError):
            Grid(**params)

    # A time selection needs a time axis
    def test_selection_without_time_raises(self):
        params = {**self._COMMON_PARAMS, "time_min": None, "time_max": None, "mode": None, "dtime": None, "selection": [True]}
        with pytest.raises(ValueError, match="selection"):
            Grid(**params)

    # Latitude bounds outside [-90, 90] are rejected
    def test_lat_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Grid(**{**self._COMMON_PARAMS, "lat_min": -100, "bathymetry_grid_path": "/nonexistent/path.nc"})

    # Longitude bounds outside [-180, 180] are rejected
    def test_lon_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Grid(**{**self._COMMON_PARAMS, "lon_min": -200, "bathymetry_grid_path": "/nonexistent/path.nc"})

    # dz and z_array are mutually exclusive; neither given is an error
    def test_dz_none_without_z_array_raises(self):
        with pytest.raises(ValueError):
            Grid(**{**self._COMMON_PARAMS, "dz": None, "bathymetry_grid_path": "/nonexistent/path.nc"})

    # A range that divides evenly is accepted (the grid then fails on missing bathymetry file)
    def test_positive_bounds_evenly_divisible_accepted(self):
        with pytest.raises(FileNotFoundError):
            Grid(**{**self._COMMON_PARAMS, "lat_min": 2, "lat_max": 12, "dlat": 5})

    # A range that does not divide evenly is rejected
    def test_positive_bounds_not_divisible_raises(self):
        with pytest.raises(ValueError, match="Latitude range"):
            Grid(**{**self._COMMON_PARAMS, "lat_min": 2, "lat_max": 8, "dlat": 5})

    # Longitude and depth ranges are checked the same way
    def test_lon_and_depth_not_divisible_raise(self):
        with pytest.raises(ValueError, match="Longitude range"):
            Grid(**{**self._COMMON_PARAMS, "lon_min": 2, "lon_max": 8, "dlon": 5})
        with pytest.raises(ValueError, match="Depth range"):
            Grid(**{**self._COMMON_PARAMS, "z_min": 20, "z_max": 80, "dz": 50})

    # Float steps are compared with a tolerance (1 % 0.1 is not 0 in floating point)
    def test_float_step_accepted(self):
        with pytest.raises(FileNotFoundError):
            Grid(**{**self._COMMON_PARAMS, "lat_min": 0, "lat_max": 1, "dlat": 0.1})

    # min must be smaller than max for latitude, longitude and depth
    @pytest.mark.parametrize("params", [
        dict(lat_min=30, lat_max=10),
        dict(lat_min=10, lat_max=10),
        dict(lon_min=50, lon_max=-50),
        dict(z_min=500, z_max=100),
    ])
    def test_min_not_smaller_than_max_raises(self, params):
        with pytest.raises(ValueError, match="smaller than"):
            Grid(**{**self._COMMON_PARAMS, **params})

    # Steps must be positive
    @pytest.mark.parametrize("params", [dict(dlat=0), dict(dlon=-1), dict(dz=0), dict(dz=-100)])
    def test_non_positive_step_raises(self, params):
        with pytest.raises(ValueError, match="must be > 0"):
            Grid(**{**self._COMMON_PARAMS, **params})


def _full_grid_df(with_time=False, time_array=("2000-01-01 00:00:00",)):
    """Full (lat, lon, depth[, time]) template matching the stub grids below."""
    lat, lon, lev = [2.5, 7.5], [2.5, 7.5], [0.0, 50.0]
    if with_time:
        rows = list(itertools.product(lat, lon, lev, time_array))
        df = pd.DataFrame(rows, columns=["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"])
    else:
        rows = list(itertools.product(lat, lon, lev))
        df = pd.DataFrame(rows, columns=["LATITUDE", "LONGITUDE", "LEV_M"])
    df["water"] = True
    return df


def _make_grid_stub(mode="Y", time_array=("2000-01-01 00:00:00",), time_max="2001-01-01"):
    """Build a dummy Grid with time (no bathymetry/xarray needed)."""
    grid = object.__new__(Grid)
    for k, v in dict(
        lat_min=0, lat_max=10, dlat=5, lon_min=0, lon_max=10, dlon=5,
        z_min=0, z_max=100, dz=50, z_array=None,
        mode=mode, time_array=np.array(time_array), time_max=time_max,
    ).items():
        setattr(grid, k, v)
    grid.grid = _full_grid_df(with_time=True, time_array=time_array)
    return grid


def _make_no_time_grid_stub():
    """Build a dummy Grid without time axis (no bathymetry/xarray needed)."""
    grid = object.__new__(Grid)
    for k, v in dict(
        lat_min=0, lat_max=10, dlat=5, lon_min=0, lon_max=10, dlon=5,
        z_min=0, z_max=100, dz=50, z_array=None, mode=None,
    ).items():
        setattr(grid, k, v)
    grid.grid = _full_grid_df(with_time=False)
    return grid


class TestBinSpace:
    """Test bin_space, which performs spatial gridding on in-memory data."""

    # Rows within the grid snap onto cell centres, out-of-range rows are dropped
    def test_bins_and_drops_out_of_range_rows(self):
        grid = _make_grid_stub()
        df = pd.DataFrame({
            "LATITUDE": [1.0, 6.0, 20.0],
            "LONGITUDE": [1.0, 6.0, 1.0],
            "LEV_M": [10.0, 60.0, 10.0],
            "VAL": [1.0, 2.0, 3.0],
        })
        result = grid.bin_space(df)
        assert len(result) == 2
        assert set(result["LATITUDE"]) == {2.5, 7.5}

    # Omitting value_col/agg returns every row with snapped coordinates, unaggregated
    def test_no_agg_returns_every_row(self):
        grid = _make_grid_stub()
        df = pd.DataFrame({
            "LATITUDE": [1.0, 2.0],
            "LONGITUDE": [1.0, 2.0],
            "LEV_M": [10.0, 20.0],
            "VAL": [1.0, 3.0],
        })
        result = grid.bin_space(df)
        assert len(result) == 2

    # value_col + agg groups rows landing in the same cell and aggregates them
    def test_aggregates_when_value_col_and_agg_given(self):
        grid = _make_grid_stub()
        df = pd.DataFrame({
            "LATITUDE": [1.0, 2.0],
            "LONGITUDE": [1.0, 2.0],
            "LEV_M": [10.0, 20.0],
            "VAL": [1.0, 3.0],
        })
        result = grid.bin_space(df, "VAL", agg="mean")
        assert len(result) == 1
        assert result["VAL"].iloc[0] == 2.0

    # A list of aggregations keeps each stat as its own column, renaming the first back to value_col
    def test_list_agg_flattens_and_renames_first_stat_to_value_col(self):
        grid = _make_grid_stub()
        df = pd.DataFrame({
            "LATITUDE": [1.0, 1.0],
            "LONGITUDE": [1.0, 1.0],
            "LEV_M": [10.0, 10.0],
            "VAL": [1.0, 3.0],
        })
        result = grid.bin_space(df, "VAL", agg=["mean", "std", "count"])
        assert result["VAL"].iloc[0] == 2.0
        assert result["count"].iloc[0] == 2


class TestBinTime:
    # DATEANDTIME is binned onto the grid's time steps and value_col aggregated within each
    def test_bins_time_and_aggregates(self):
        grid = _make_grid_stub(time_array=("2000-01-01 00:00:00", "2001-01-01 00:00:00"), time_max="2002-01-01")
        df = pd.DataFrame({
            "LATITUDE": [1.0, 1.0], "LONGITUDE": [1.0, 1.0], "LEV_M": [10.0, 10.0],
            "DATEANDTIME": ["2000-03-01", "2000-06-01"],
            "VAL": [1.0, 3.0],
        })
        result = grid.bin_time(df, "VAL", agg="mean")
        assert len(result) == 1
        assert result["VAL"].iloc[0] == 2.0
        assert result["DATEANDTIME"].iloc[0] == "2000-01-01 00:00:00"


class TestGridBinComposesSpaceAndTime:
    # bin() is bin_space() followed by bin_time() in one call
    def test_bin_combines_space_and_time_binning(self):
        grid = _make_grid_stub()
        df = pd.DataFrame({
            "LATITUDE": [1.0, 6.0], "LONGITUDE": [1.0, 6.0], "LEV_M": [10.0, 60.0],
            "DATEANDTIME": ["2000-03-01", "2000-06-01"],
            "VAL": [1.0, 5.0],
        })
        result = grid.bin(df, "VAL", agg="mean")
        assert len(result) == 2
        assert set(result["LATITUDE"]) == {2.5, 7.5}
        assert set(result["DATEANDTIME"]) == {"2000-01-01 00:00:00"}


class TestGridBinSeasonalMean:
    """agg="seasonal_mean" runs monthly-first annual averaging before time-binning."""

    # A single lopsided month (Jul) does not dominate the annual mean over three January samples
    def test_weights_months_equally_regardless_of_sample_count(self):
        grid = _make_grid_stub()
        df = pd.DataFrame({
            "LATITUDE": [1.0, 1.0, 1.0, 1.0], "LONGITUDE": [1.0, 1.0, 1.0, 1.0],
            "LEV_M": [10.0, 10.0, 10.0, 10.0],
            "DATEANDTIME": ["2000-01-01", "2000-01-05", "2000-01-20", "2000-07-01"],
            "VAL": [0.0, 0.0, 0.0, 30.0],
        })
        result = grid.bin(df, "VAL", agg="seasonal_mean")
        assert len(result) == 1
        # (Jan mean=0, Jul mean=30) -> 15, not the naive row mean of 7.5
        assert result["VAL"].iloc[0] == 15.0
        assert result["DATEANDTIME"].iloc[0] == "2000-01-01 00:00:00"

    # map_dataframes forwards agg="seasonal_mean" through to Grid.bin
    def test_map_dataframes_accepts_seasonal_mean_as_agg(self):
        grid = _make_grid_stub()
        dataframes = {"TEMPERATURE": pd.DataFrame({
            "LATITUDE": [1.0, 1.0, 1.0], "LONGITUDE": [1.0, 1.0, 1.0], "LEV_M": [10.0, 10.0, 10.0],
            "DATEANDTIME": ["2000-01-01", "2000-01-15", "2000-07-01"],
            "TEMPERATURE": [0.0, 0.0, 20.0],
        })}
        result = grid.map_dataframes(dataframes, agg="seasonal_mean")
        cell = result[result["P_TEMPERATURE"].notna()]
        assert len(cell) == 1
        assert cell["P_TEMPERATURE"].iloc[0] == 10.0


class TestNoTimeGridBin:
    # A grid without time axis delegates bin() straight to bin_space()
    def test_bin_delegates_to_bin_space(self):
        grid = _make_no_time_grid_stub()
        df = pd.DataFrame({
            "LATITUDE": [1.0, 2.0],
            "LONGITUDE": [1.0, 2.0],
            "LEV_M": [10.0, 20.0],
            "VAL": [1.0, 3.0],
        })
        result = grid.bin(df, "VAL", agg="mean")
        assert len(result) == 1
        assert result["VAL"].iloc[0] == 2.0

    # Time operations are rejected instead of silently doing nothing
    def test_time_operations_raise(self):
        grid = _make_no_time_grid_stub()
        df = pd.DataFrame({"LATITUDE": [1.0], "LONGITUDE": [1.0], "LEV_M": [10.0], "VAL": [1.0]})
        with pytest.raises(ValueError, match="time axis"):
            grid.bin_time(df, "VAL")
        with pytest.raises(ValueError, match="time axis"):
            grid.bin(df, "VAL", agg="seasonal_mean")

    # has_time follows from mode
    def test_has_time(self):
        assert not _make_no_time_grid_stub().has_time
        assert _make_grid_stub().has_time


class TestNoTimeMapTables:
    """Test Grid.map_tables for a grid without time axis against an in-memory database."""

    @staticmethod
    def _conn():
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE station (ID INTEGER, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
        conn.execute("CREATE TABLE P_NITRATE (ID INTEGER, LEV_M REAL, VAL REAL)")
        conn.executemany("INSERT INTO station VALUES (?, ?, ?, ?)", [
            (1, 1.0, 1.0, "2000-01-01 00:00:00"),
            (2, 2.0, 2.0, "2000-01-02 00:00:00"),
            (3, 20.0, 1.0, "2000-01-03 00:00:00"),  # outside the grid extent
        ])
        conn.executemany("INSERT INTO P_NITRATE VALUES (?, ?, ?)",
                         [(1, 10.0, 1.0), (2, 20.0, 3.0), (3, 10.0, 99.0)])
        return conn

    # Coordinates come from station, observations in one cell are aggregated, bare names work
    def test_joins_station_and_aggregates_per_cell(self):
        grid = _make_no_time_grid_stub()
        grid.grid_id = 998
        conn = self._conn()
        mapped = grid.map_tables(conn, param_tables=["NITRATE"])

        assert mapped == ["P_NITRATE_998"]
        out = get_table_as_df(conn, "P_NITRATE_998")
        assert "DATEANDTIME" not in out.columns
        # The out-of-extent observation is dropped, the two others share one cell
        assert len(out) == 1
        assert out["P_NITRATE"].iloc[0] == 2.0
        assert out["count"].iloc[0] == 2
        assert out["median"].iloc[0] == 2.0
        assert out["idx"].notna().all()


class TestMapDataframes:
    """Test map_dataframes, which first creates the grid and then maps the data to it."""

    # Multiple parameters merge onto the same full grid template, each as its own P_<name> column
    def test_merges_multiple_parameters_onto_full_template(self):
        grid = _make_no_time_grid_stub()
        dataframes = {
            "NITRATE": pd.DataFrame({
                "LATITUDE": [1.0, 2.0], "LONGITUDE": [1.0, 2.0], "LEV_M": [10.0, 10.0],
                "NITRATE": [1.0, 3.0],
            }),
            "OXYGEN": pd.DataFrame({
                "LATITUDE": [6.0], "LONGITUDE": [6.0], "LEV_M": [60.0],
                "OXYGEN": [5.0],
            }),
        }
        result = grid.map_dataframes(dataframes)
        assert len(result) == 8  # full grid template preserved (2x2x2 cells)
        assert {"P_NITRATE", "P_OXYGEN"}.issubset(result.columns)

        cell = result[(result["LATITUDE"] == 2.5) & (result["LONGITUDE"] == 2.5) & (result["LEV_M"] == 0.0)]
        assert cell["P_NITRATE"].iloc[0] == 2.0

        cell = result[(result["LATITUDE"] == 7.5) & (result["LONGITUDE"] == 7.5) & (result["LEV_M"] == 50.0)]
        assert cell["P_OXYGEN"].iloc[0] == 5.0

    # bin_fn overrides the default mean aggregation with a custom per-parameter function
    def test_custom_bin_fn_overrides_default_agg(self):
        grid = _make_no_time_grid_stub()
        calls = []

        def bin_fn(df, value_col):
            calls.append(value_col)
            return grid.bin_space(df, value_col, agg="max")

        dataframes = {"NITRATE": pd.DataFrame({
            "LATITUDE": [1.0, 1.0], "LONGITUDE": [1.0, 1.0], "LEV_M": [10.0, 10.0],
            "NITRATE": [1.0, 9.0],
        })}
        result = grid.map_dataframes(dataframes, bin_fn=bin_fn)
        assert calls == ["P_NITRATE"]

        cell = result[(result["LATITUDE"] == 2.5) & (result["LONGITUDE"] == 2.5) & (result["LEV_M"] == 0.0)]
        assert cell["P_NITRATE"].iloc[0] == 9.0

    # On a time-aware Grid, merging also matches on the binned DATEANDTIME
    def test_grid_map_dataframes_merges_on_time_too(self):
        grid = _make_grid_stub()
        dataframes = {"TEMPERATURE": pd.DataFrame({
            "LATITUDE": [1.0], "LONGITUDE": [1.0], "LEV_M": [10.0],
            "DATEANDTIME": ["2000-03-01"],
            "TEMPERATURE": [4.0],
        })}
        result = grid.map_dataframes(dataframes)
        assert "DATEANDTIME" in result.columns
        filled = result[result["P_TEMPERATURE"].notna()]
        assert len(filled) == 1
        assert filled["P_TEMPERATURE"].iloc[0] == 4.0
        assert filled["DATEANDTIME"].iloc[0] == "2000-01-01 00:00:00"

    # dropping_land_cells=True (default) removes a land cell with no data for any parameter
    def test_drops_empty_land_cells_by_default(self):
        grid = _make_no_time_grid_stub()
        # Mark one cell as land with no data anywhere for any parameter
        grid.grid.loc[
            (grid.grid["LATITUDE"] == 2.5) & (grid.grid["LONGITUDE"] == 2.5)
            & (grid.grid["LEV_M"] == 0.0), "water",
        ] = False
        dataframes = {"NITRATE": pd.DataFrame({
            "LATITUDE": [6.0], "LONGITUDE": [6.0], "LEV_M": [60.0], "NITRATE": [1.0],
        })}
        result = grid.map_dataframes(dataframes)
        assert len(result) == 7  # the empty land cell was dropped
        assert not ((result["LATITUDE"] == 2.5) & (result["LONGITUDE"] == 2.5)
                    & (result["LEV_M"] == 0.0)).any()

    # dropping_land_cells=False keeps that same empty land cell
    def test_dropping_land_cells_false_keeps_empty_land_cells(self):
        grid = _make_no_time_grid_stub()
        grid.grid.loc[
            (grid.grid["LATITUDE"] == 2.5) & (grid.grid["LONGITUDE"] == 2.5)
            & (grid.grid["LEV_M"] == 0.0), "water",
        ] = False
        dataframes = {"NITRATE": pd.DataFrame({
            "LATITUDE": [6.0], "LONGITUDE": [6.0], "LEV_M": [60.0], "NITRATE": [1.0],
        })}
        result = grid.map_dataframes(dataframes, dropping_land_cells=False)
        assert len(result) == 8


class TestOnlineWideTableFunctions:
    """create_wide_table_online / load_wide_table / get_missing_value_info_per_param(_offline)
    operate on tables already mapped into the database by Grid.map_tables."""

    @staticmethod
    def _mapped_conn():
        conn = sqlite3.connect(":memory:")
        # Create dummy station and nitrate tables
        conn.execute("CREATE TABLE station (ID INTEGER, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
        conn.execute("CREATE TABLE P_NITRATE (ID INTEGER, LEV_M REAL, VAL REAL)")
        conn.executemany("INSERT INTO station VALUES (?,?,?,?)", [
            (1, 1.0, 1.0, "2000-03-01 00:00:00"),
            (2, 6.0, 6.0, "2000-06-01 00:00:00"),
        ])
        conn.executemany("INSERT INTO P_NITRATE VALUES (?,?,?)", [
            (1, 10.0, 5.0),
            (2, 60.0, 9.0),
        ])
        conn.commit()

        grid = _make_grid_stub()
        grid.grid_id = 999
        grid.grid_name = "grid_999"
        grid.time_min = "2000-01-01 00:00:00"
        grid.map_tables(conn, param_tables=["P_NITRATE"])
        return conn

    # map_tables writes both the grid template and a per-parameter mapped table
    def test_map_tables_creates_grid_and_mapped_table(self):
        conn = self._mapped_conn()
        assert does_table_exist(conn, "grid_999", "table")
        assert does_table_exist(conn, "P_NITRATE_999", "table")

    # Bare parameter names ("NITRATE") are accepted everywhere "P_NITRATE" is
    def test_bare_parameter_name_accepted_everywhere(self):
        # Create dummy station and nitrate tables
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE station (ID INTEGER, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)"
        )
        conn.execute("CREATE TABLE P_NITRATE (ID INTEGER, LEV_M REAL, VAL REAL)")
        conn.executemany("INSERT INTO station VALUES (?,?,?,?)", [(1, 1.0, 1.0, "2000-03-01 00:00:00")])
        conn.executemany("INSERT INTO P_NITRATE VALUES (?,?,?)", [(1, 10.0, 5.0)])
        conn.commit()

        # Gridding and map_tables
        grid = _make_grid_stub()
        grid.grid_id = 997
        grid.grid_name = "grid_997"
        grid.time_min = "2000-01-01 00:00:00"
        mapped = grid.map_tables(conn, param_tables=["NITRATE"])  # bare parameter name
        assert mapped == ["P_NITRATE_997"]
        assert does_table_exist(conn, "P_NITRATE_997", "table")

        # create_wide_tale_online
        wide_name = create_wide_table_online(conn, 997, param_tables=["NITRATE"]) # bare parameter name
        df_wide = get_table_as_df(conn, wide_name)
        assert "P_NITRATE" in df_wide.columns

        # get_missing_value_info_per_param
        coverage = get_missing_value_info_per_param(conn, wide_name, ["NITRATE"]) # bare parameter name
        assert "P_NITRATE" in coverage["parameter"].values

    # map_tables casts every bind value to a native Python type
    def test_map_tables_handles_numpy_int64_bounds(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE station (ID INTEGER, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)"
        )
        conn.execute("CREATE TABLE P_NITRATE (ID INTEGER, LEV_M REAL, VAL REAL)")
        conn.executemany("INSERT INTO station VALUES (?,?,?,?)", [(1, 1.0, 1.0, "2000-03-01 00:00:00")])
        conn.executemany("INSERT INTO P_NITRATE VALUES (?,?,?)", [(1, 10.0, 5.0)])
        conn.commit()

        # Grid spec
        grid = _make_grid_stub()
        grid.grid_id = 996
        grid.grid_name = "grid_996"
        grid.time_min = "2000-01-01 00:00:00"
        grid.z_min = np.array([0, 50, 100], dtype=np.int64).min()
        grid.z_max = np.array([0, 50, 100], dtype=np.int64).max()
        assert isinstance(grid.z_min, np.int64) and isinstance(grid.z_max, np.int64)  # check if type in grid is correct

        # Map parameter table onto grid (should work with correct types)
        grid.map_tables(conn, param_tables=["P_NITRATE"])
        df = get_table_as_df(conn, "P_NITRATE_996")
        assert df["P_NITRATE"].notna().sum() == 1

    # The wide table joins the grid template with each mapped parameter table on idx
    def test_create_wide_table_online(self):
        conn = self._mapped_conn()
        wide_name = create_wide_table_online(conn, 999, param_tables=["P_NITRATE"])
        assert wide_name.startswith("wide_999_")

        df_wide = get_table_as_df(conn, wide_name)
        assert len(df_wide) == 8  # 2 lat x 2 lon x 2 depth, 1 time step
        assert "P_NITRATE" in df_wide.columns

        cell = df_wide[(df_wide["LATITUDE"] == 2.5) & (df_wide["LONGITUDE"] == 2.5)
                       & (df_wide["LEV_M"] == 0.0)]
        assert cell["P_NITRATE"].iloc[0] == pytest.approx(5.0)

        cell = df_wide[(df_wide["LATITUDE"] == 7.5) & (df_wide["LONGITUDE"] == 7.5)
                       & (df_wide["LEV_M"] == 50.0)]
        assert cell["P_NITRATE"].iloc[0] == pytest.approx(9.0)

    # dropping_land_cells removes an empty land cell that was loaded back from SQL
    def test_load_wide_table_drops_empty_land_cells(self):
        conn = self._mapped_conn()
        wide_name = create_wide_table_online(conn, 999, param_tables=["P_NITRATE"])
        # Mark a cell with no data as land
        conn.execute(
            f"UPDATE {wide_name} SET water=0 WHERE LATITUDE=2.5 AND LONGITUDE=7.5 AND LEV_M=0.0"
        )
        conn.commit()

        dropped = load_wide_table(conn, wide_name, dropping_land_cells=True)
        assert len(dropped) == 7

        kept = load_wide_table(conn, wide_name, dropping_land_cells=False)
        assert len(kept) == 8

    # The online per-param missingness matches the manually computed fraction
    def test_get_missing_value_info_per_param(self):
        conn = self._mapped_conn()
        wide_name = create_wide_table_online(conn, 999, param_tables=["P_NITRATE"])
        result = get_missing_value_info_per_param(conn, wide_name, ["P_NITRATE"])
        row = result[result["parameter"] == "P_NITRATE"].iloc[0]
        assert row["total"] == 6
        assert row["relative"] == pytest.approx(75.0)

    # The offline (list-of-DataFrames) missingness-info variant agrees with the online (SQL) variant
    def test_get_missing_value_info_offline_per_param_matches_online(self):
        conn = self._mapped_conn()
        wide_name = create_wide_table_online(conn, 999, param_tables=["P_NITRATE"])
        df_wide = get_table_as_df(conn, wide_name)

        online = get_missing_value_info_per_param(conn, wide_name, ["P_NITRATE"])
        offline = get_missing_value_info_offline_per_param([df_wide])

        online_row = online[online["parameter"] == "P_NITRATE"].iloc[0]
        offline_row = offline[offline["parameter"] == "P_NITRATE"].iloc[0]

        assert offline_row["absolute"] == online_row["total"]
        assert offline_row["relative"] == pytest.approx(online_row["relative"])

    # An empty input list returns an empty, correctly-shaped DataFrame
    def test_get_missing_value_info_offline_per_param_empty_list(self):
        result = get_missing_value_info_offline_per_param([])
        assert result.empty
        assert list(result.columns) == ["parameter", "absolute", "relative"]


@pytest.mark.needs_bathymetry
@pytest.mark.skipif(not _has_xarray, reason="xarray not installed ([grid] extra)")
class TestGridManagerWithBathymetry:
    # Full round-trip against a real bathymetry file: Create, look up, load, remove a grid
    def test_create_and_retrieve_grid(self, tmp_path):
        bathymetry = os.environ.get("BATHYMETRY_PATH")
        db_copy = str(tmp_path / "test.sqlite")

        gm = GridManager(db_copy, "grid_info")
        try:
            grid = gm.create_grid(
                lat_min=0, lat_max=10, dlat=1,
                lon_min=0, lon_max=10, dlon=1,
                z_min=0, z_max=1000, dz=500,
                time_min="2000-01-01 00:00:00",
                time_max="2001-01-01 00:00:00",
                mode="Y", dtime=1,
                bathymetry_grid_path=bathymetry,
            )
            assert gm.does_grid_exist(grid.grid_id)

            loaded = gm.load_grid(grid.grid_id, bathymetry_grid_path=bathymetry)
            assert loaded.grid_id == grid.grid_id
            assert loaded.lat_min == grid.lat_min

            gm.remove_grid(grid.grid_id)
            assert not gm.does_grid_exist(grid.grid_id)
        finally:
            gm.close()


class TestGenerateId:
    # IDs generated at same second are distinct and increasing
    def test_unique_and_increasing(self):
        ids = [generate_grid_id() for _ in range(1000)]
        assert len(set(ids)) == 1000
        assert ids == sorted(ids)

    # IDs are plain integers
    def test_is_int(self):
        assert isinstance(generate_grid_id(), int)

    # Two wide tables for the same grid created in quick succession do not collide
    def test_wide_table_names_unique(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE grid_1 (idx INTEGER, DATEANDTIME, LEV_M, LATITUDE, LONGITUDE, water)")
        conn.execute("CREATE TABLE P_X_1 (idx INTEGER, P_X)")
        a = create_wide_table_online(conn, 1, ["X"])
        b = create_wide_table_online(conn, 1, ["X"])
        assert a != b
        conn.close()


def _make_manager_grid(grid_id):
    """Build a dummy Grid adn insert it into a dummy grid_info table."""
    # Dummy grid
    grid = _make_grid_stub()

    # Insert dummy grid into dummy grid_info tables
    for k, v in dict(
        grid_id=grid_id, grid_name=f"grid_{grid_id}", selection=None,
        time_min="2000-01-01 00:00:00", dtime=1,
    ).items():
        setattr(grid, k, v)
    return grid


class TestAddGrid:
    @pytest.fixture
    def gm(self, tmp_path):
        # Define a grid manager
        manager = GridManager(str(tmp_path / "test.sqlite"))
        yield manager
        manager.close()

    # Grid table and grid_info row are both written
    def test_adds_table_and_row(self, gm):
        grid = _make_manager_grid(11)
        gm.add_grid(grid)
        assert gm.does_grid_exist(11)
        assert does_table_exist(gm.connection, "grid_11", "table")

    # A duplicate grid_id raises and leaves the first grid intact
    def test_duplicate_id_raises_and_keeps_first(self, gm):
        gm.add_grid(_make_manager_grid(11))
        with pytest.raises(Exception):
            gm.add_grid(_make_manager_grid(11))
        assert gm.does_grid_exist(11)
        assert len(get_table_as_df(gm.connection, "grid_11")) == len(_make_manager_grid(11).grid)

    # A failing grid_info insert drops the grid table
    def test_failed_insert_leaves_no_table(self, gm):
        # A grid_info row with this ID but no grid table makes the insert fail
        gm.connection.execute("INSERT INTO grid_info (grid_id) VALUES (12)")
        gm.connection.commit()
        with pytest.raises(sqlite3.Error):
            gm.add_grid(_make_manager_grid(12))
        assert not does_table_exist(gm.connection, "grid_12", "table")

    # A grid without time axis (dtime is NULL) is found again by its parameters
    def test_no_time_grid_found_by_params(self, gm):
        grid = _make_no_time_grid_stub()
        for k, v in dict(grid_id=21, grid_name="grid_21", selection=None,
                         time_min=None, time_max=None, dtime=None).items():
            setattr(grid, k, v)
        gm.add_grid(grid)
        found = gm.get_grid_id_from_params(0, 10, 5, 0, 10, 5, z_min=0, z_max=100, dz=50)
        assert found == 21

    # A table with the grid name that already exists is not overwritten
    def test_existing_table_not_overwritten(self, gm):
        gm.connection.execute("CREATE TABLE grid_13 (x INTEGER)")
        gm.connection.execute("INSERT INTO grid_13 VALUES (1)")
        gm.connection.commit()
        with pytest.raises(ValueError):
            gm.add_grid(_make_manager_grid(13))
        assert gm.connection.execute("SELECT x FROM grid_13").fetchall() == [(1,)]
        assert not gm.does_grid_exist(13)


def _add_mapped_tables(gm, grid_id):
    """Add a mapped parameter table and a wide table for a grid, as map_tables/create_wide_table_online do."""
    gm.connection.execute(f"CREATE TABLE P_NITRATE_{grid_id} (idx INTEGER)")
    gm.connection.execute(f"CREATE TABLE wide_{grid_id}_555 (idx INTEGER)")
    gm.connection.commit()


class TestRemoveGrid:
    @pytest.fixture
    def gm(self, tmp_path):
        # Dummy grid manager
        manager = GridManager(str(tmp_path / "test.sqlite"))
        yield manager
        manager.close()

    @staticmethod
    def _table_names(gm):
        # Get table names
        return {r[0] for r in gm.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    # The grid_info row, grid table, mapped tables and wide tables are removed, other grids and tables stay
    def test_removes_only_this_grid(self, gm):
        gm.add_grid(_make_manager_grid(31))
        gm.add_grid(_make_manager_grid(32))
        _add_mapped_tables(gm, 31)
        _add_mapped_tables(gm, 32)
        gm.connection.execute("CREATE TABLE P_NITRATE (ID INTEGER)")

        gm.remove_grid(31)

        assert not gm.does_grid_exist(31)
        assert gm.does_grid_exist(32)
        assert self._table_names(gm) == {"grid_info", "grid_32", "P_NITRATE_32", "wide_32_555", "P_NITRATE"}

    # grid_info keeps its primary key and column types after removing a grid
    def test_keeps_grid_info_schema(self, gm):
        gm.add_grid(_make_manager_grid(31))
        gm.add_grid(_make_manager_grid(32))
        before = gm.connection.execute("PRAGMA table_info(grid_info)").fetchall()
        gm.remove_grid(31)
        assert gm.connection.execute("PRAGMA table_info(grid_info)").fetchall() == before
        with pytest.raises(sqlite3.IntegrityError):
            gm.connection.execute("INSERT INTO grid_info (grid_id) VALUES (32)")

    # A custom grid_info table name is used, the default one is not touched
    def test_custom_grid_info_table(self, tmp_path):
        with GridManager(str(tmp_path / "test.sqlite"), "my_info") as gm:
            gm.add_grid(_make_manager_grid(33))
            gm.connection.execute("CREATE TABLE grid_info (grid_id)")
            gm.connection.execute("INSERT INTO grid_info VALUES (99)")
            gm.remove_grid(33)
            assert not gm.does_grid_exist(33)
            assert gm.connection.execute("SELECT grid_id FROM grid_info").fetchall() == [(99,)]

    # A row in grid_info whose grid table is already gone is still removed
    def test_removes_row_without_table(self, gm):
        gm.add_grid(_make_manager_grid(34))
        gm.connection.execute("DROP TABLE grid_34")
        gm.connection.commit()
        gm.remove_grid(34)
        assert not gm.does_grid_exist(34)

    # remove_all_grids empties grid_info and drops every grid's tables
    def test_remove_all_grids(self, gm):
        gm.add_grid(_make_manager_grid(35))
        gm.add_grid(_make_manager_grid(36))
        _add_mapped_tables(gm, 35)
        gm.remove_all_grids()
        assert gm.connection.execute("SELECT COUNT(*) FROM grid_info").fetchone() == (0,)
        assert self._table_names(gm) == {"grid_info"}

    # Removing an unknown grid does nothing
    def test_unknown_grid_is_noop(self, gm):
        gm.add_grid(_make_manager_grid(37))
        gm.remove_grid(12345)
        assert gm.does_grid_exist(37)

    # A failure while dropping rolls back the row deletion and the earlier drops
    def test_failure_rolls_back(self, gm):
        gm.add_grid(_make_manager_grid(38))
        _add_mapped_tables(gm, 38)
        # An invalid table name makes validate_identifier raise part-way through the drops
        gm.connection.execute('CREATE TABLE "bad name_38" (x)')
        gm.connection.commit()
        with pytest.raises(ValueError):
            gm.remove_grid(38)
        assert gm.does_grid_exist(38)
        assert {"grid_38", "P_NITRATE_38", "wide_38_555"} <= self._table_names(gm)


class TestGridLookup:
    """get_grid_id_from_params finds a stored grid independent of how its parameters are written."""

    @pytest.fixture
    def gm(self, tmp_path):
        # Dummy grid manager
        manager = GridManager(str(tmp_path / "test.sqlite"))
        yield manager
        manager.close()

    @staticmethod
    def _stored(gm, grid_id, **overrides):
        # Build test grid, optionally override parameters
        grid = _make_manager_grid(grid_id)
        for k, v in overrides.items():
            setattr(grid, k, v)
        gm.add_grid(grid)

    # Define a base row in grid_info table
    _BASE = dict(lat_min=0, lat_max=10, dlat=5, lon_min=0, lon_max=10, dlon=5,
                 z_min=0, z_max=100, dz=50, time_min="2000-01-01 00:00:00",
                 time_max="2001-01-01", mode="Y", dtime=1)

    # int and float depth parameters describe the same grid
    def test_int_and_float_depths_match(self, gm):
        self._stored(gm, 41)
        assert gm.get_grid_id_from_params(**{**self._BASE, "z_min": 0.0, "z_max": 100.0, "dz": 50.0}) == 41

    # Date strings with or without time of day describe the same grid
    def test_equivalent_time_strings_match(self, gm):
        self._stored(gm, 42)
        assert gm.get_grid_id_from_params(**{**self._BASE, "time_min": "2000-01-01", "time_max": "2001-01-01 00:00:00"}) == 42

    # z_array: order and int/float do not matter (Grid sorts it)
    def test_z_array_order_and_type_do_not_matter(self, gm):
        self._stored(gm, 43, z_array=np.array([0.0, 50.0, 100.0]))
        params = {k: v for k, v in self._BASE.items() if k not in ("z_min", "z_max", "dz")}
        assert gm.get_grid_id_from_params(**params, z_array=[100, 0, 50]) == 43

    # Different grids are told apart
    def test_different_grid_not_found(self, gm):
        self._stored(gm, 44)
        assert gm.get_grid_id_from_params(**{**self._BASE, "dz": 25}) is None
        assert gm.get_grid_id_from_params(**{**self._BASE, "dtime": 2}) is None


def _make_time_grid_stub(mode, dtime):
    """Build a dummy Grid with only the attributes the time array and _map_time need."""
    grid = object.__new__(Grid)
    for k, v in dict(mode=mode, dtime=dtime, selection=None,
                     time_min="2000-01-01", time_max="2001-01-01").items():
        setattr(grid, k, v)
    grid.time_array = grid._create_time_array()
    return grid


def _snap(grid, dates):
    """Return the time bin each date is mapped to by _map_time."""
    df = pd.DataFrame({"DATEANDTIME": pd.to_datetime(dates)})
    return grid._map_time(df)["DATEANDTIME"].tolist()


class TestTimeModes:
    # M: dtime-sized month bins, every snapped value is a node of the time array
    def test_month_dtime_3(self):
        grid = _make_time_grid_stub("M", 3)
        assert grid.time_array.tolist() == ["01", "04", "07", "10"]
        dates = ["2005-01-15", "2005-02-10", "2005-03-31", "2005-05-01", "2005-12-31"]
        assert _snap(grid, dates) == ["01", "01", "01", "04", "10"]

    # M with dtime=1 keeps the plain two-digit month
    def test_month_dtime_1(self):
        grid = _make_time_grid_stub("M", 1)
        assert _snap(grid, ["2005-02-10", "2005-11-30"]) == ["02", "11"]

    # D: dtime-sized day bins
    def test_day_dtime_10(self):
        grid = _make_time_grid_stub("D", 10)
        assert grid.time_array.tolist() == ["01", "11", "21", "31"]
        assert _snap(grid, ["2005-03-01", "2005-03-10", "2005-03-11", "2005-03-31"]) == ["01", "01", "11", "31"]

    # MD nodes cover the whole leap reference year, including 29 Feb and 31 Dec
    def test_month_day_nodes_cover_year(self):
        grid = _make_time_grid_stub("MD", 1)
        assert len(grid.time_array) == 366
        assert "02-29 00:00:00" in grid.time_array
        assert grid.time_array[-1] == "12-31 00:00:00"

    # MD drops the time of day so that observations match a node
    def test_month_day_drops_time_of_day(self):
        grid = _make_time_grid_stub("MD", 1)
        assert _snap(grid, ["2005-03-15 12:30:00", "2005-12-31 23:59:00"]) == ["03-15 00:00:00", "12-31 00:00:00"]

    # MD aligns the days of non-leap and leap years to the same calendar date
    def test_month_day_leap_alignment(self):
        grid = _make_time_grid_stub("MD", 1)
        assert _snap(grid, ["2005-03-01", "2004-03-01", "2005-12-31", "2004-12-31"]) == [
            "03-01 00:00:00", "03-01 00:00:00", "12-31 00:00:00", "12-31 00:00:00"]

    # MD with dtime>1 snaps to the start of the bin, which is a node
    def test_month_day_dtime_30(self):
        grid = _make_time_grid_stub("MD", 30)
        nodes = set(grid.time_array.tolist())
        snapped = _snap(grid, ["2005-01-15", "2005-03-15", "2005-12-31"])
        assert snapped[0] == "01-01 00:00:00"
        assert set(snapped) <= nodes

    # Every date of the year maps onto a time array node
    @pytest.mark.parametrize("mode,dtime", [("M", 1), ("M", 5),
                                            ("D", 1), ("D", 7),
                                            ("MD", 1), ("MD", 45)])
    def test_every_date_maps_to_a_node(self, mode, dtime):
        grid = _make_time_grid_stub(mode, dtime)
        dates = pd.date_range("2004-01-01", "2005-12-31", freq="D")
        # Equal sets: No date lands outside the nodes, and no node is left without a date
        assert set(_snap(grid, dates)) == set(grid.time_array.tolist())


class TestEmptyMissingValueInfo:
    """Missing-value summaries keep their documented columns for degenerate input."""

    # No P_* columns at all: Empty summary
    def test_offline_without_parameter_columns(self):
        for df in (pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": pd.Series(dtype=float)})):
            result = get_missing_value_info_offline(df)
            assert result.empty
            assert list(result.columns) == ["parameter", "absolute", "relative"]

    # The per-parameter variant returns exactly the documented columns
    def test_offline_per_param_columns(self):
        df = pd.DataFrame({
            "LATITUDE": [1.0, 2.0], "LONGITUDE": [1.0, 2.0], "LEV_M": [0.0, 0.0],
            "DATEANDTIME": ["2000", "2000"], "water": [True, True], "P_X": [1.0, np.nan],
        })
        result = get_missing_value_info_offline_per_param([df])
        assert list(result.columns) == ["parameter", "absolute", "relative"]

    # A grid without any water cell does not divide by zero
    def test_offline_per_param_zero_cells(self):
        df = pd.DataFrame({
            "LATITUDE": pd.Series(dtype=float), "LONGITUDE": pd.Series(dtype=float),
            "LEV_M": pd.Series(dtype=float), "DATEANDTIME": pd.Series(dtype=object),
            "water": pd.Series(dtype=bool), "P_X": pd.Series(dtype=float),
        })
        result = get_missing_value_info_offline_per_param([df])
        assert list(result.columns) == ["parameter", "absolute", "relative"]
        assert result["relative"].iloc[0] == 0.0
