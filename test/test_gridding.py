"""Tests for comfort.gridding."""
import itertools
import os
import numpy as np
import pandas as pd
import pytest

_has_xarray = True
try:
    import xarray
except ImportError:
    _has_xarray = False

from comfort.gridding import (
    Grid,
    _axis_cells,
    average_duplicate_locations,
    _drop_land_cells,
    get_missing_value_info,
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
        result = _drop_land_cells(df)
        assert len(result) == 1
        assert result["water"].iloc[0]

    # A land cell that does have a value is kept despite water=False
    def test_land_with_value_is_kept(self):
        df = self._make_wide([False, True], [5.0, 1.0])
        result = _drop_land_cells(df)
        assert len(result) == 2

    # A grid with no land cells at all passes through unchanged
    def test_all_water_unchanged(self):
        df = self._make_wide([True, True], [1.0, 2.0])
        result = _drop_land_cells(df)
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
        result = _drop_land_cells(df)

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
        result = _drop_land_cells(df)
        assert len(result) == 1
        assert result["water"].iloc[0]

    # A land cell that does have a value is kept despite water=False
    def test_no_time_land_with_value_kept(self):
        df = self._make_wide_no_time([False, True], [5.0, 1.0])
        result = _drop_land_cells(df)
        assert len(result) == 2

    # A grid with no land cells at all passes through unchanged
    def test_no_time_all_water_unchanged(self):
        df = self._make_wide_no_time([True, True], [1.0, 2.0])
        result = _drop_land_cells(df)
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
        result = _drop_land_cells(df)
        assert len(result) == 3
        dropped = set(df.index) - set(result.index)
        assert dropped == {1}


class TestGetMissingValueInfo:
    def _make_wide(self, values):
        return pd.DataFrame({
            "P_NITRATE": values,
            "LATITUDE": range(len(values)),
        })

    # No missing values gives absolute=0
    def test_no_nulls(self):
        df = self._make_wide([1.0, 2.0, 3.0])
        result = get_missing_value_info(df)
        assert result.loc[result["parameter"] == "NITRATE", "absolute"].iloc[0] == 0

    # Every value missing gives relative=100%
    def test_all_nulls(self):
        df = self._make_wide([None, None])
        result = get_missing_value_info(df)
        assert result.loc[result["parameter"] == "NITRATE", "relative"].iloc[0] == 100.0

    # Parameters restrict the output and are matched ignoring casing and the P_ prefix
    def test_parameters_select_columns(self):
        df = self._make_wide([1.0, None]).assign(P_DIC=[1.0, 2.0])
        result = get_missing_value_info(df, parameters=["nitrate"])
        assert list(result["parameter"]) == ["NITRATE"]

    # A parameter without a column raises
    def test_unknown_parameter_raises(self):
        with pytest.raises(KeyError):
            get_missing_value_info(self._make_wide([1.0]), parameters=["DIC"])

    # An empty DataFrame does not divide by zero
    def test_empty_dataframe(self):
        df = pd.DataFrame({"P_NITRATE": pd.Series([], dtype=float)})
        result = get_missing_value_info(df)
        assert result.loc[result["parameter"] == "NITRATE", "relative"].iloc[0] == 0.0


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

    # The bathymetry path is a required argument
    def test_bathymetry_path_required(self):
        params = {k: v for k, v in self._COMMON_PARAMS.items() if k != "bathymetry_grid_path"}
        with pytest.raises(TypeError, match="bathymetry_grid_path"):
            Grid(**params)

    # An explicit None bathymetry path gives the same FileNotFoundError as a missing file
    def test_none_bathymetry_path_raises(self):
        with pytest.raises(FileNotFoundError):
            Grid(**{**self._COMMON_PARAMS, "bathymetry_grid_path": None})

    # An unrecognised time mode is rejected up front
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            Grid(**{**self._COMMON_PARAMS, "mode": "INVALID", "bathymetry_grid_path": "/nonexistent/path.nc"})

    # The time parameters are all-or-nothing
    @pytest.mark.parametrize("missing", ["time_min", "time_max", "mode", "dtime"])
    def test_partial_time_parameters_raise(self, missing):
        with pytest.raises(ValueError, match=missing):
            Grid(**{**self._COMMON_PARAMS, missing: None})

    # Time bounds must be given in the long format (date-only is rejected)
    @pytest.mark.parametrize("param", ["time_min", "time_max"])
    def test_date_only_time_bound_raises(self, param):
        with pytest.raises(ValueError, match=param):
            Grid(**{**self._COMMON_PARAMS, param: "2000-01-01"})

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


def _make_grid_stub(mode="Y", time_array=("2000-01-01 00:00:00",), time_min="2000-01-01 00:00:00",
                    time_max="2000-12-31 23:59:59"):
    """Build a dummy Grid with time (no bathymetry/xarray needed)."""
    grid = object.__new__(Grid)
    for k, v in dict(
        lat_min=0, lat_max=10, dlat=5, lon_min=0, lon_max=10, dlon=5,
        z_min=0, z_max=100, dz=50, z_array=None,
        mode=mode, time_array=np.array(time_array), time_min=time_min, time_max=time_max,
        dtime=1, selection=None,
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
        assert result["VAL_count"].iloc[0] == 2
        assert "count" not in result.columns


class TestCellAxes:
    """Template cell centres and the bins of bin_space come from one construction."""

    @staticmethod
    def _axes_stub(lon, lat):
        """Dummy Grid with the given (min, max, step) per axis and two depth layers."""
        grid = object.__new__(Grid)
        for k, v in dict(
            lon_min=lon[0], lon_max=lon[1], dlon=lon[2],
            lat_min=lat[0], lat_max=lat[1], dlat=lat[2],
            z_min=0, z_max=100, dz=50, z_array=None,
        ).items():
            setattr(grid, k, v)
        return grid

    @staticmethod
    def _bathymetry(tmp_path):
        """Write a flat 500 m deep bathymetry NetCDF."""
        lon = np.arange(0.0, 4.0, 0.5)
        lat = np.arange(0.0, 4.0, 0.5)
        ds = xarray.Dataset(
            {"elevation": (("lat", "lon"), np.full((len(lat), len(lon)), -500.0))},
            coords={"lat": lat, "lon": lon},
        )
        path = str(tmp_path / "bathymetry.nc")
        ds.to_netcdf(path)
        return path

    # The centres assigned by bin_space equal the template cell centres
    @pytest.mark.skipif(not _has_xarray, reason="xarray not installed ([grid] extra)")
    @pytest.mark.parametrize("step", [0.1, 0.2, 0.3])
    def test_binned_centres_equal_template_centres(self, tmp_path, step):
        grid = Grid(
            bathymetry_grid_path=self._bathymetry(tmp_path),
            lon_min=0, lon_max=3, dlon=step, lat_min=0, lat_max=3, dlat=step,
            z_min=0, z_max=100, dz=50,
        )
        lon, lat = np.meshgrid(grid.grid["LONGITUDE"].unique(), grid.grid["LATITUDE"].unique())
        df = pd.DataFrame({"LONGITUDE": lon.ravel(), "LATITUDE": lat.ravel(), "LEV_M": 10.0, "VAL": 1.0})

        binned = grid.bin_space(df, "VAL", "mean")

        # The binned centres are exactly the template centres
        for col in ("LONGITUDE", "LATITUDE"):
            assert np.array_equal(np.sort(binned[col].unique()), np.sort(grid.grid[col].unique()))

        # Merging the binned values onto the template fills every cell of the first depth layer
        merged = grid.grid.merge(binned, on=["LATITUDE", "LONGITUDE", "LEV_M"], how="left")
        assert merged.loc[merged["LEV_M"] == 0, "VAL"].notna().all()

    # An axis has (max - min) / step cells and one more edge
    @pytest.mark.parametrize("lo, hi, step", [(-90, 90, 0.3), (30, 70, 0.7), (0, 1, 0.1)])
    def test_cell_count(self, lo, hi, step):
        edges, centres = _axis_cells(lo, hi, step)
        assert len(centres) == round((hi - lo) / step)
        assert len(edges) == len(centres) + 1

    # A point beyond the maximum belongs to no cell and is dropped
    def test_point_beyond_the_maximum_is_dropped(self):
        grid = self._axes_stub(lon=(0, 10, 1), lat=(-90, 90, 0.3))
        df = pd.DataFrame({"LONGITUDE": [5.5, 5.5], "LATITUDE": [89.95, 90.1], "LEV_M": 10.0, "VAL": 1.0})
        result = grid.bin_space(df, "VAL", "mean")
        assert len(result) == 1
        assert result["LATITUDE"].iloc[0] == pytest.approx(89.85)

    # The maximum itself is inclusive, also for a negative maximum
    @pytest.mark.parametrize("hi", [30, 0, -10])
    def test_maximum_is_inclusive(self, hi):
        grid = self._axes_stub(lon=(hi - 10, hi, 5), lat=(hi - 10, hi, 5))
        df = pd.DataFrame({"LONGITUDE": [hi], "LATITUDE": [hi], "LEV_M": 10.0, "VAL": 1.0})
        result = grid.bin_space(df, "VAL", "mean")
        assert len(result) == 1
        assert result["LONGITUDE"].iloc[0] == hi - 2.5
        assert result["LATITUDE"].iloc[0] == hi - 2.5

    # Each depth bin is labelled by its upper boundary (the top of the layer), the maximum depth belongs to the last bin
    def test_depth_maximum_is_inclusive(self):
        grid = self._axes_stub(lon=(0, 10, 5), lat=(0, 10, 5))
        df = pd.DataFrame({"LONGITUDE": 1.0, "LATITUDE": 1.0, "LEV_M": [0.0, 49.9, 50.0, 100.0], "VAL": 1.0})
        result = grid.bin_space(df, "VAL", "count")
        assert dict(zip(result["LEV_M"], result["VAL"])) == {0.0: 2, 50.0: 2}


class TestBinTime:
    # DATEANDTIME is binned onto the grid's time steps and value_col aggregated within each
    def test_bins_time_and_aggregates(self):
        grid = _make_grid_stub(time_array=("2000-01-01 00:00:00", "2001-01-01 00:00:00"),
                               time_max="2001-12-31 23:59:59")
        df = pd.DataFrame({
            "LATITUDE": [1.0, 1.0], "LONGITUDE": [1.0, 1.0], "LEV_M": [10.0, 10.0],
            "DATEANDTIME": ["2000-03-01", "2000-06-01"],
            "VAL": [1.0, 3.0],
        })
        result = grid.bin_time(df, "VAL", agg="mean")
        assert len(result) == 1
        assert result["VAL"].iloc[0] == 2.0
        assert result["DATEANDTIME"].iloc[0] == "2000-01-01 00:00:00"


class TestTimeWindow:
    """bin_time drops rows outside [time_min, time_max] in every mode."""

    @staticmethod
    def _df(dates):
        # Simple df (onw row)
        return pd.DataFrame({"LATITUDE": 1.0, "LONGITUDE": 1.0, "LEV_M": 10.0,
                             "DATEANDTIME": dates, "VAL": np.arange(len(dates), dtype=float)})

    # Both ends are included (Y mode)
    def test_y_mode_window_is_inclusive(self):
        grid = _make_grid_stub(time_array=("2000-01-01 00:00:00", "2001-01-01 00:00:00"), time_max="2001-12-31 23:59:59")
        dates = ["1999-12-31 23:59:59", "2000-01-01 00:00:00", "2001-12-31 23:59:59", "2002-01-01 00:00:00"]
        result = grid.bin_time(self._df(dates), "VAL", agg="count")
        assert dict(zip(result["DATEANDTIME"], result["VAL"])) == {"2000-01-01 00:00:00": 1, "2001-01-01 00:00:00": 1}

    # Data from outside the time window is not included (M, D, MD modes)
    @pytest.mark.parametrize("mode", ["M", "D", "MD"])
    def test_pooling_modes_drop_out_of_window_rows(self, mode):
        grid = _make_grid_stub(mode=mode, time_array=("01",))
        dates = ["1990-01-01 00:00:00", "2000-01-01 00:00:00", "2000-12-31 23:59:59", "2001-01-01 00:00:00"]
        result = grid.bin_time(self._df(dates), "VAL", agg="count")
        assert result["VAL"].sum() == 2

    # A deselected time step is dropped (not counted into previous bin)
    def test_deselected_step_does_not_leak_into_previous_bin(self):
        grid = _make_grid_stub(time_max="2002-12-31 23:59:59")
        # Nodes are 2000, 2001, 2002 (2001 is deselected)
        grid.selection = np.array([True, False, True])
        grid.time_array = grid._create_time_array()
        result = grid.bin_time(self._df(["2000-06-01", "2001-06-01", "2002-06-01"]), "VAL", agg="count")
        assert dict(zip(result["DATEANDTIME"], result["VAL"])) == {"2000-01-01 00:00:00": 1, "2002-01-01 00:00:00": 1}


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

    # The annual stamp of a year at or after time_max does not contaminate the last bin (2.3)
    def test_year_at_time_max_is_excluded(self):
        grid = _make_grid_stub(time_array=("2000-01-01 00:00:00", "2001-01-01 00:00:00"),
                               time_max="2002-01-01 00:00:00")
        df = pd.DataFrame({
            "LATITUDE": 1.0, "LONGITUDE": 1.0, "LEV_M": 10.0,
            "DATEANDTIME": ["2001-03-01 00:00:00", "2002-01-01 00:00:00"], "VAL": [1.0, 100.0],
        })
        result = grid.bin(df, "VAL", agg="seasonal_mean")
        assert dict(zip(result["DATEANDTIME"], result["VAL"])) == {"2001-01-01 00:00:00": 1.0}

    # Seasonal means need annual stamps to match the bins
    @pytest.mark.parametrize("kwargs", [{"mode": "YM"}, {"time_min": "2000-07-01 00:00:00"}])
    def test_unsupported_grid_raises(self, kwargs):
        grid = _make_grid_stub(**kwargs)
        df = pd.DataFrame({"LATITUDE": 1.0, "LONGITUDE": 1.0, "LEV_M": 10.0,
                           "DATEANDTIME": ["2000-08-01"], "VAL": [1.0]})
        with pytest.raises(ValueError, match="seasonal_mean"):
            grid.bin(df, "VAL", agg="seasonal_mean")

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

    # A list for agg names the extra statistics per parameter, so two parameters do not collide on the merge
    def test_list_agg_with_two_parameters_has_no_suffix_collisions(self):
        grid = _make_no_time_grid_stub()
        dataframes = {
            name: pd.DataFrame({"LATITUDE": [1.0, 1.0], "LONGITUDE": [1.0, 1.0], "LEV_M": [10.0, 10.0], name: values})
            for name, values in {"NITRATE": [1.0, 3.0], "OXYGEN": [10.0, 20.0]}.items()
        }
        result = grid.map_dataframes(dataframes, agg=["mean", "median", "std", "count"])
        assert not [c for c in result.columns if c.endswith(("_x", "_y"))]  # no auto suffixes
        cell = result[(result["LATITUDE"] == 2.5) & (result["LONGITUDE"] == 2.5) & (result["LEV_M"] == 0.0)]
        assert cell["P_NITRATE"].iloc[0] == 2.0
        assert cell["P_NITRATE_count"].iloc[0] == 2
        assert cell["P_OXYGEN"].iloc[0] == 15.0
        assert cell["P_OXYGEN_median"].iloc[0] == 15.0
        assert cell["P_OXYGEN_std"].iloc[0] == pytest.approx(7.0710678)

    # Keys and value columns in any casing / with P_ prefix are matched
    @pytest.mark.parametrize("key, column", [("nitrate", "NITRATE"), ("NITRATE", "p_nitrate"), ("P_NITRATE", "nitrate")])
    def test_key_and_column_spellings(self, key, column):
        grid = _make_no_time_grid_stub()
        df = pd.DataFrame({"LATITUDE": [1.0, 2.0], "LONGITUDE": [1.0, 2.0], "LEV_M": [10.0, 10.0], column: [1.0, 3.0]})
        result = grid.map_dataframes({key: df})
        cell = result[(result["LATITUDE"] == 2.5) & (result["LONGITUDE"] == 2.5) & (result["LEV_M"] == 0.0)]
        assert cell["P_NITRATE"].iloc[0] == 2.0

    # A key without a matching value column is an error
    def test_missing_value_column_raises(self):
        grid = _make_no_time_grid_stub()
        df = pd.DataFrame({"LATITUDE": [1.0], "LONGITUDE": [1.0], "LEV_M": [10.0], "OXYGEN": [1.0]})
        with pytest.raises(KeyError, match="NITRATE"):
            grid.map_dataframes({"NITRATE": df})

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


@pytest.mark.needs_bathymetry
@pytest.mark.skipif(not _has_xarray, reason="xarray not installed ([grid] extra)")
class TestGridWithBathymetry:
    # The grid template is built from a real bathymetry file
    def test_builds_template(self):
        grid = Grid(
            lon_min=0, lon_max=10, dlon=1,
            lat_min=0, lat_max=10, dlat=1,
            z_min=0, z_max=1000, dz=500,
            time_min="2000-01-01 00:00:00", time_max="2001-01-01 00:00:00",
            mode="Y", dtime=1,
            bathymetry_grid_path=os.environ.get("BATHYMETRY_PATH"),
        )
        assert {"LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME", "water"} <= set(grid.grid.columns)
        # 10 x 10 cells, 2 depth levels (z_max is the upper edge), one time step
        assert len(grid.grid) == 10 * 10 * 2 * len(grid.time_array)



def _make_time_grid_stub(mode, dtime):
    """Build a dummy Grid with only the attributes the time array and _map_time need."""
    grid = object.__new__(Grid)
    for k, v in dict(mode=mode, dtime=dtime, selection=None,
                     time_min="2000-01-01 00:00:00", time_max="2001-01-01 00:00:00").items():
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
    def test_without_parameter_columns(self):
        for df in (pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": pd.Series(dtype=float)})):
            result = get_missing_value_info(df)
            assert result.empty
            assert list(result.columns) == ["parameter", "absolute", "relative"]


class TestBinColumnCasing:
    # value_col in another casing bins the column under its original spelling
    def test_bin_space_other_casing(self):
        grid = _make_grid_stub()
        df = pd.DataFrame({"LATITUDE": [1.0, 2.0], "LONGITUDE": [1.0, 2.0], "LEV_M": [10.0, 20.0], "VAL": [1.0, 3.0]})
        result = grid.bin_space(df, value_col="val", agg="mean")
        assert result["VAL"].iloc[0] == 2.0
