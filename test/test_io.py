"""Tests for comfort.io, i.e. load_comfort, describe_variables, subset_region, read_parameter, info."""
import logging
import sqlite3
import numpy as np
import pandas as pd
import pytest

from comfort.io import (
    _read_with_geo, describe_variables, info, load_comfort, read_parameter, subset_region,
)

_has_shapely = True
try:
    import shapely
except ImportError:
    _has_shapely = False

needs_shapely = pytest.mark.skipif(not _has_shapely, reason="shapely not installed ([geo] extra)")


@pytest.fixture
def comfort_db(tmp_path):
    """Create a minimal COMFORT-like SQLite database for testing."""
    db_path = str(tmp_path / "test_comfort.sqlite")
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE station (
            ID INTEGER PRIMARY KEY,
            LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT
        )
    """)
    conn.executemany("INSERT INTO station VALUES (?,?,?,?)", [
        (1, 45.0, -30.0, "2000-01-15"),
        (2, 55.0, -20.0, "2005-06-10"),
    ])

    # Common column layout for parameter tables
    _param_ddl = """
        CREATE TABLE {tbl} (
            ID INTEGER, LEV_DBAR REAL, LEV_M REAL, VAL REAL,
            PQF1 INTEGER, PQF2 INTEGER, SQF INTEGER,
            BOTTLE_NUMBER INTEGER, PROFILE_NUMBER INTEGER,
            PROFILE_BEST INTEGER, UNITS_ID INTEGER, INSTRUMENT_ID INTEGER
        )
    """
    for tbl in ("P_NITRATE", "P_OXYGEN"):
        conn.execute(_param_ddl.format(tbl=tbl))

    # Station 1's first (and only) profile, PROFILE_NUMBER 1, CTD instrument 1: 3 depths
    conn.executemany("INSERT INTO P_NITRATE VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        (1, 0.0,   0.0,   5.0, 1, 3, 0, 1, 1, 1, 3, 1),
        (1, 100.0, 100.0, 4.0, 1, 3, 0, 2, 1, 1, 3, 1),
        (1, 200.0, 200.0, 3.0, 1, 3, 0, 3, 1, 1, 3, 1),
    ])
    # Station 2's first (and only) profile, also PROFILE_NUMBER 1, bottle instrument 2: 3 depths
    conn.executemany("INSERT INTO P_NITRATE VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        (2, 0.0,   0.0,   8.0, 1, 3, 0, 1, 1, 1, 3, 2),
        (2, 100.0, 100.0, 7.0, 1, 3, 0, 2, 1, 1, 3, 2),
        (2, 200.0, 200.0, 6.0, 1, 3, 0, 3, 1, 1, 3, 2),
    ])
    conn.executemany("INSERT INTO P_OXYGEN VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        (1, 0.0,   0.0,   200.0, 1, 3, 0, 1, 1, 1, 3, 1),
        (1, 100.0, 100.0, 180.0, 1, 3, 0, 2, 1, 1, 3, 1),
        (2, 0.0,   0.0,   220.0, 1, 3, 0, 1, 1, 1, 3, 2),
        (2, 100.0, 100.0, 210.0, 1, 3, 0, 2, 1, 1, 3, 2),
    ])

    conn.commit()
    conn.close()
    return db_path


class TestDescribeVariables:
    # Output is a DataFrame
    def test_returns_dataframe(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        result = describe_variables(conn)
        conn.close()
        assert isinstance(result, pd.DataFrame)

    # The summary table exposes the documented column set
    def test_has_expected_columns(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        result = describe_variables(conn)
        conn.close()
        assert {"parameter", "n_samples", "n_stations", "n_profiles",
                "depth_min_m", "depth_max_m"}.issubset(result.columns)

    # Sample counts match the number of rows per parameter table
    def test_correct_sample_counts(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        result = describe_variables(conn).set_index("parameter")
        conn.close()
        assert result.loc["NITRATE", "n_samples"] == 6
        assert result.loc["OXYGEN", "n_samples"] == 4

    # Station and profile counts are distinct from raw sample counts
    def test_station_and_profile_counts(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        result = describe_variables(conn).set_index("parameter")
        conn.close()
        assert result.loc["NITRATE", "n_stations"] == 2
        assert result.loc["NITRATE", "n_profiles"] == 2
        assert result.loc["OXYGEN", "n_stations"] == 2
        assert result.loc["OXYGEN", "n_profiles"] == 2

    # A UNITS table present resolves UNITS_ID to a human-readable name
    def test_units_resolved_when_units_table_present(self, tmp_path):
        db_path = str(tmp_path / "with_units.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
        conn.execute("INSERT INTO station VALUES (1, 45.0, -30.0, '2000-01-15')")
        conn.execute("CREATE TABLE P_NITRATE (ID INTEGER, LEV_DBAR REAL, LEV_M REAL, VAL REAL, PQF1 INTEGER, PQF2 INTEGER, SQF INTEGER, BOTTLE_NUMBER INTEGER, PROFILE_NUMBER INTEGER, PROFILE_BEST INTEGER, UNITS_ID INTEGER, INSTRUMENT_ID INTEGER)")
        conn.execute("INSERT INTO P_NITRATE VALUES (1, 0.0, 0.0, 5.0, 1, 3, 0, 1, 1, 1, 3, 1)")
        conn.execute("CREATE TABLE UNITS (ID INTEGER PRIMARY KEY, NAME_SHORT TEXT)")
        conn.execute("INSERT INTO UNITS VALUES (3, 'µmol/kg')")
        conn.commit()
        result = describe_variables(conn)
        conn.close()
        assert "units" in result.columns
        assert "units_id" not in result.columns
        assert result.loc[result["parameter"] == "NITRATE", "units"].iloc[0] == "µmol/kg"

    # Without a UNITS table, the raw UNITS_ID is shown as a fallback string
    def test_units_fallback_without_units_table(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        result = describe_variables(conn)
        conn.close()
        assert "units" in result.columns
        assert result.loc[result["parameter"] == "NITRATE", "units"].iloc[0] == "3"

    # A present INSTRUMENT table resolves INSTRUMENT_ID to names
    def test_instruments_resolved_when_instrument_table_present(self, tmp_path):
        db_path = str(tmp_path / "with_instruments.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
        conn.execute("INSERT INTO station VALUES (1, 45.0, -30.0, '2000-01-15')")
        conn.execute(
            "CREATE TABLE P_NITRATE (ID INTEGER, LEV_DBAR REAL, LEV_M REAL, VAL REAL, "
            "PQF1 INTEGER, PQF2 INTEGER, SQF INTEGER, BOTTLE_NUMBER INTEGER, "
            "PROFILE_NUMBER INTEGER, PROFILE_BEST INTEGER, UNITS_ID INTEGER, INSTRUMENT_ID INTEGER)"
        )
        conn.execute("INSERT INTO P_NITRATE VALUES (1, 0.0, 0.0, 5.0, 1, 3, 0, 1, 1, 1, 3, 1)")
        conn.execute("INSERT INTO P_NITRATE VALUES (1, 100.0, 100.0, 4.0, 1, 3, 0, 2, 1, 1, 3, 2)")
        conn.execute("CREATE TABLE INSTRUMENT (ID INTEGER PRIMARY KEY, WOD_ID INTEGER, NAME TEXT, NOTES TEXT, DATE_ADDED TEXT, DATE_UPDATED TEXT)")
        conn.execute("INSERT INTO INSTRUMENT VALUES (1, 1, 'MBT', '', '', '')")
        conn.execute("INSERT INTO INSTRUMENT VALUES (2, 2, 'XBT', '', '', '')")
        conn.commit()
        result = describe_variables(conn)
        conn.close()
        assert "instruments" in result.columns
        assert "instrument_id" not in result.columns
        instruments_val = result.loc[result["parameter"] == "NITRATE", "instruments"].iloc[0]
        assert "MBT" in instruments_val
        assert "XBT" in instruments_val

    # Without an INSTRUMENT table, raw instrument IDs are joined as a fallback string
    def test_instruments_fallback_without_instrument_table(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        result = describe_variables(conn)
        conn.close()
        assert "instruments" in result.columns
        assert result.loc[result["parameter"] == "NITRATE", "instruments"].iloc[0] == "1 / 2"

    # parameters argument restricts the summary to the requested subset
    def test_parameters_filter(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        result = describe_variables(conn, parameters=["NITRATE"])
        conn.close()
        assert list(result["parameter"]) == ["NITRATE"]
        assert len(result) == 1

    # An unknown requested parameter is a hard error, not silently ignored
    def test_parameters_filter_unknown_raises(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        with pytest.raises(ValueError, match="Unknown parameters"):
            describe_variables(conn, parameters=["NITRATE", "DOESNOTEXIST"])
        conn.close()

    # Multiple units in use for one parameter are all shown, not just the first
    def test_multiple_units_shown(self, tmp_path):
        db_path = str(tmp_path / "multi_units.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
        conn.execute("INSERT INTO station VALUES (1, 45.0, -30.0, '2000-01-15')")
        conn.execute(
            "CREATE TABLE P_NITRATE (ID INTEGER, LEV_DBAR REAL, LEV_M REAL, VAL REAL, "
            "PQF1 INTEGER, PQF2 INTEGER, SQF INTEGER, BOTTLE_NUMBER INTEGER, "
            "PROFILE_NUMBER INTEGER, PROFILE_BEST INTEGER, UNITS_ID INTEGER, INSTRUMENT_ID INTEGER)"
        )
        conn.execute("INSERT INTO P_NITRATE VALUES (1, 0.0, 0.0, 5.0, 1, 3, 0, 1, 1, 1, 3, 1)")
        conn.execute("INSERT INTO P_NITRATE VALUES (1, 100.0, 100.0, 10.0, 1, 3, 0, 2, 1, 1, 7, 1)")
        conn.execute("CREATE TABLE UNITS (ID INTEGER PRIMARY KEY, NAME_SHORT TEXT)")
        conn.execute("INSERT INTO UNITS VALUES (3, 'µmol/kg')")
        conn.execute("INSERT INTO UNITS VALUES (7, 'µmol/L')")
        conn.commit()
        result = describe_variables(conn)
        conn.close()
        units_val = result.loc[result["parameter"] == "NITRATE", "units"].iloc[0]
        assert "µmol/kg" in units_val
        assert "µmol/L" in units_val

    # A parameter reported in more than one unit also logs a warning
    def test_multiple_units_warns(self, tmp_path, caplog):
        db_path = str(tmp_path / "multi_units_warn.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
        conn.execute("INSERT INTO station VALUES (1, 45.0, -30.0, '2000-01-15')")
        conn.execute(
            "CREATE TABLE P_NITRATE (ID INTEGER, LEV_DBAR REAL, LEV_M REAL, VAL REAL, "
            "PQF1 INTEGER, PQF2 INTEGER, SQF INTEGER, BOTTLE_NUMBER INTEGER, "
            "PROFILE_NUMBER INTEGER, PROFILE_BEST INTEGER, UNITS_ID INTEGER, INSTRUMENT_ID INTEGER)"
        )
        conn.execute("INSERT INTO P_NITRATE VALUES (1, 0.0, 0.0, 5.0, 1, 3, 0, 1, 1, 1, 3, 1)")
        conn.execute("INSERT INTO P_NITRATE VALUES (1, 100.0, 100.0, 10.0, 1, 3, 0, 2, 1, 1, 7, 1)")
        conn.execute("CREATE TABLE UNITS (ID INTEGER PRIMARY KEY, NAME_SHORT TEXT)")
        conn.execute("INSERT INTO UNITS VALUES (3, 'µmol/kg')")
        conn.execute("INSERT INTO UNITS VALUES (7, 'µmol/L')")
        conn.commit()
        with caplog.at_level(logging.WARNING):
            describe_variables(conn)
        conn.close()
        assert "different units" in caplog.text
        assert "NITRATE" in caplog.text

    # A present PLATFORM table resolves PLATFORM_ID to names
    def test_platforms_resolved_when_platform_table_present(self, tmp_path):
        db_path = str(tmp_path / "with_platforms.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, "
            "LONGITUDE REAL, DATEANDTIME TEXT, PLATFORM_ID INTEGER)"
        )
        conn.execute("INSERT INTO station VALUES (1, 45.0, -30.0, '2000-01-15', 100)")
        conn.execute("INSERT INTO station VALUES (2, 55.0, -20.0, '2005-06-10', 200)")
        conn.execute(
            "CREATE TABLE P_NITRATE (ID INTEGER, LEV_DBAR REAL, LEV_M REAL, VAL REAL, "
            "PQF1 INTEGER, PQF2 INTEGER, SQF INTEGER, BOTTLE_NUMBER INTEGER, "
            "PROFILE_NUMBER INTEGER, PROFILE_BEST INTEGER, UNITS_ID INTEGER, INSTRUMENT_ID INTEGER)"
        )
        conn.execute("INSERT INTO P_NITRATE VALUES (1, 0.0, 0.0, 5.0, 1, 3, 0, 1, 1, 1, 3, 1)")
        conn.execute("INSERT INTO P_NITRATE VALUES (2, 0.0, 0.0, 8.0, 1, 3, 0, 1, 2, 1, 3, 2)")
        conn.execute(
            "CREATE TABLE PLATFORM (ID INTEGER PRIMARY KEY, NODC_CODE TEXT, WOD_ID INTEGER, "
            "IMO_ID TEXT, CALLSIGN TEXT, COUNTRY_ID INTEGER, NAME TEXT, NAME_NATIVE TEXT, "
            "NOTES_ICES TEXT, NOTES_WOD TEXT, NOTES TEXT, DATE_ADDED TEXT, DATE_UPDATED TEXT)"
        )
        conn.execute("INSERT INTO PLATFORM (ID, NAME) VALUES (100, 'ALEMANIA')")
        conn.execute("INSERT INTO PLATFORM (ID, NAME) VALUES (200, 'POLARSTERN')")
        conn.commit()
        result = describe_variables(conn)
        conn.close()
        assert "platforms" in result.columns
        platforms_val = result.loc[result["parameter"] == "NITRATE", "platforms"].iloc[0]
        assert "ALEMANIA" in platforms_val
        assert "POLARSTERN" in platforms_val

    # Without a PLATFORM table, the platforms column is present but empty
    def test_platforms_fallback_without_platform_table(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        result = describe_variables(conn)
        conn.close()
        assert "platforms" in result.columns
        assert result.loc[result["parameter"] == "NITRATE", "platforms"].iloc[0] is None


class TestSubsetRegion:
    @staticmethod
    def _df():
        return pd.DataFrame({
            "LATITUDE": [10.0, 20.0, 30.0],
            "LONGITUDE": [5.0, 10.0, 15.0],
            "LEV_M": [10.0, 100.0, 500.0],
            "DATEANDTIME": ["2000-01-01", "2005-06-15", "2010-12-31"],
            "VAL": [1.0, 2.0, 3.0],
        })

    # lat_min/lat_max filter rows outside the latitude band
    def test_lat_filter(self):
        r = subset_region(self._df(), lat_min=15.0, lat_max=25.0)
        assert list(r["LATITUDE"]) == [20.0]

    # lon_min/lon_max filter rows outside the longitude band
    def test_lon_filter(self):
        r = subset_region(self._df(), lon_min=8.0, lon_max=12.0)
        assert list(r["LONGITUDE"]) == [10.0]

    # depth_max filters out deeper rows
    def test_depth_filter(self):
        r = subset_region(self._df(), depth_max=200.0)
        assert len(r) == 2

    # date_min/date_max filter rows outside the time window
    def test_date_filter(self):
        r = subset_region(self._df(), date_min="2005-01-01", date_max="2008-01-01")
        assert list(r["LATITUDE"]) == [20.0]

    # No filters given returns every row unchanged
    def test_no_filters_returns_all(self):
        assert len(subset_region(self._df())) == 3

    # Multiple filter kinds combine with AND semantics
    def test_combined_filters(self):
        r = subset_region(self._df(), lat_min=15.0, depth_max=200.0)
        assert list(r["LATITUDE"]) == [20.0]

    # A filter on a column absent from df warns and is skipped, not a crash
    def test_missing_column_warns(self, caplog):
        df = self._df().drop(columns="LATITUDE")
        with caplog.at_level(logging.WARNING):
            r = subset_region(df, lat_min=15.0)
        assert len(r) == 3  # filter skipped, all rows returned
        assert "LATITUDE" in caplog.text

    # The returned DataFrame is independent of the input
    def test_returns_copy(self):
        df = self._df()
        r = subset_region(df, lat_min=15.0)
        r["VAL"] = 99.0
        assert df["VAL"].iloc[1] != 99.0

    # region= filters to points inside an arbitrary polygon, not just a box
    @needs_shapely
    def test_region_filter(self):
        region = {"box": [(8, 15), (12, 15), (12, 25), (8, 25)]}
        r = subset_region(self._df(), region=region)
        assert list(r["LATITUDE"]) == [20.0]

    # An explicit box and region= combine (intersect), not override each other
    @needs_shapely
    def test_region_and_box_combine(self):
        region = {"box": [(0, 0), (20, 0), (20, 40), (0, 40)]}
        r = subset_region(self._df(), lat_min=25.0, region=region)
        assert list(r["LATITUDE"]) == [30.0]

    # A region requested with LATITUDE/LONGITUDE absent warns and is skipped, not a crash
    def test_region_missing_columns_warns(self, caplog):
        region = {"box": [(8, 15), (12, 15), (12, 25), (8, 25)]}
        df = self._df().drop(columns=["LATITUDE", "LONGITUDE"])
        with caplog.at_level(logging.WARNING):
            r = subset_region(df, region=region)
        assert len(r) == 3
        assert "Region filter requested" in caplog.text

    # A region crossing the antimeridian (e.g. a Pacific basin stored as two parts
    # near +-180) keeps points on both sides of the dateline, not just one
    @needs_shapely
    def test_region_crossing_antimeridian(self):
        from shapely.geometry import MultiPolygon, Polygon
        pacific_strip = MultiPolygon([
            Polygon([(170, -10), (180, -10), (180, 10), (170, 10)]),
            Polygon([(-180, -10), (-170, -10), (-170, 10), (-180, 10)]),
        ])
        df = pd.DataFrame({
            "LATITUDE": [0.0, 0.0, 0.0],
            "LONGITUDE": [175.0, -175.0, 0.0],
            "VAL": [1.0, 2.0, 3.0],
        })
        r = subset_region(df, region={"pacific_strip": pacific_strip})
        assert set(r["LONGITUDE"]) == {175.0, -175.0}

    # Explicit lon_min/lon_max combined with an antimeridian-wrapping region are both applied
    @needs_shapely
    def test_region_crossing_antimeridian_with_explicit_lon(self):
        from shapely.geometry import MultiPolygon, Polygon
        pacific_strip = MultiPolygon([
            Polygon([(170, -10), (180, -10), (180, 10), (170, 10)]),
            Polygon([(-180, -10), (-170, -10), (-170, 10), (-180, 10)]),
        ])
        df = pd.DataFrame({
            "LATITUDE": [0.0, 0.0],
            "LONGITUDE": [175.0, -175.0],
            "VAL": [1.0, 2.0],
        })
        r = subset_region(df, lon_min=-180.0, lon_max=0.0, region={"pacific_strip": pacific_strip})
        assert list(r["LONGITUDE"]) == [-175.0]


class TestLoadComfort:
    # as_xarray=False returns a dict keyed by parameter name
    def test_as_dict_returns_dict(self, comfort_db):
        result = load_comfort(comfort_db, as_xarray=False)
        assert isinstance(result, dict)
        assert "NITRATE" in result

    # Each dict entry is a DataFrame
    def test_dict_values_are_dataframes(self, comfort_db):
        result = load_comfort(comfort_db, as_xarray=False)
        assert isinstance(result["NITRATE"], pd.DataFrame)

    # Station geo columns are joined in dict mode
    def test_dict_contains_geo_columns(self, comfort_db):
        result = load_comfort(comfort_db, as_xarray=False)
        assert {"LATITUDE", "LONGITUDE", "DATEANDTIME"}.issubset(result["NITRATE"].columns)

    # The generic VAL column is renamed to the parameter's own name
    def test_dict_val_renamed_to_param(self, comfort_db):
        result = load_comfort(comfort_db, as_xarray=False)
        assert "NITRATE" in result["NITRATE"].columns
        assert "VAL" not in result["NITRATE"].columns

    # target_depths=None (default) keeps each profile's native measured depths
    def test_dict_no_target_depths_keeps_native_depths(self, comfort_db):
        result = load_comfort(comfort_db, parameters=["NITRATE"], as_xarray=False)
        assert set(result["NITRATE"]["LEV_M"].unique()) == {0.0, 100.0, 200.0}

    # Passing target_depths maps dict output onto it instead
    def test_dict_target_depths_regularises_depths(self, comfort_db):
        result = load_comfort(comfort_db, parameters=["NITRATE"], as_xarray=False, target_depths=[0, 50, 100, 150, 200])
        assert set(result["NITRATE"]["LEV_M"].unique()) == {0.0, 50.0, 100.0, 150.0, 200.0}

    # max_gap/interp_method are forwarded to the interpolation when target_depths is given
    def test_dict_target_depths_respects_max_gap(self, comfort_db):
        result = load_comfort(comfort_db, parameters=["NITRATE"], as_xarray=False, target_depths=[50], max_gap=50)
        assert result["NITRATE"].empty

    # parameters argument restricts the dict to the requested subset
    def test_parameter_filter(self, comfort_db):
        result = load_comfort(comfort_db, parameters=["NITRATE"], as_xarray=False)
        assert list(result.keys()) == ["NITRATE"]

    # lat_min restricts the returned data to the matching station's rows
    def test_spatial_filter(self, comfort_db):
        result = load_comfort(comfort_db, lat_min=50.0, as_xarray=False)
        # Only station 2's profile (lat=55) should survive
        assert result["NITRATE"]["PROFILE_NUMBER"].nunique() == 1
        assert result["NITRATE"]["LATITUDE"].iloc[0] == 55.0

    # region restricts the returned data to an arbitrary polygon (no lat/lon box needed)
    @needs_shapely
    def test_region_filter(self, comfort_db):
        region = {"box": [(-25, 50), (-15, 50), (-15, 60), (-25, 60)]}
        result = load_comfort(comfort_db, region=region, as_xarray=False)
        # Only station 2's profile (lat=55) should survive
        assert result["NITRATE"]["PROFILE_NUMBER"].nunique() == 1
        assert result["NITRATE"]["LATITUDE"].iloc[0] == 55.0

    # depth_max restricts the returned data to rows within the depth bound
    def test_depth_filter(self, comfort_db):
        result = load_comfort(comfort_db, depth_max=50.0, as_xarray=False)
        assert (result["NITRATE"]["LEV_M"] <= 50.0).all()

    # date_min restricts the returned data to rows within the date bound
    def test_date_filter(self, comfort_db):
        result = load_comfort(comfort_db, date_min="2003-01-01", as_xarray=False)
        # Only station 2's profile (2005) survives
        assert result["NITRATE"]["PROFILE_NUMBER"].nunique() == 1

    # Default as_xarray=True returns an xarray Dataset
    def test_xarray_returns_dataset(self, comfort_db):
        xr = pytest.importorskip("xarray")
        result = load_comfort(comfort_db, target_depths=[0, 50, 100, 150, 200])
        assert isinstance(result, xr.Dataset)

    # Each requested parameter becomes its own data variable
    def test_xarray_has_parameter_vars(self, comfort_db):
        pytest.importorskip("xarray")
        result = load_comfort(comfort_db, target_depths=[0, 100, 200])
        assert "NITRATE" in result.data_vars

    # Dataset dims are (profile, depth) when target_depths is given
    def test_xarray_dims(self, comfort_db):
        pytest.importorskip("xarray")
        result = load_comfort(comfort_db, target_depths=[0, 100, 200])
        assert set(result.dims) == {"profile", "depth"}

    # The depth coordinate matches target_depths exactly
    def test_xarray_depth_coord(self, comfort_db):
        pytest.importorskip("xarray")
        result = load_comfort(comfort_db, target_depths=[0, 100, 200])
        np.testing.assert_array_equal(result["depth"].values, [0.0, 100.0, 200.0])

    # Latitude/longitude are exposed as coordinates, not data variables, when target_depths is given
    def test_xarray_geo_coords(self, comfort_db):
        pytest.importorskip("xarray")
        result = load_comfort(comfort_db, target_depths=[0, 100, 200])
        assert "latitude" in result.coords
        assert "longitude" in result.coords

    # Station/instrument/platform metadata are also exposed as coordinates when target_depths is given
    def test_xarray_station_instrument_platform_coords(self, comfort_db):
        pytest.importorskip("xarray")
        result = load_comfort(comfort_db, target_depths=[0, 100, 200])
        assert "station_id" in result.coords
        assert "profile_number" in result.coords
        assert "instrument" in result.coords
        assert "platform" in result.coords
        assert set(result["station_id"].values) == {1, 2}
        assert set(result["instrument"].values) == {"1", "2"}

    # Default (target_depths=None) exposes each parameter on its own
    # observation dimension, namespaced coordinates, native measured depths
    def test_xarray_scattered_by_default(self, comfort_db):
        pytest.importorskip("xarray")
        result = load_comfort(comfort_db, parameters=["NITRATE"])
        assert "NITRATE_obs" in result.dims
        assert "profile" not in result.dims
        np.testing.assert_array_equal(sorted(result["NITRATE_depth"].values), [0.0, 0.0, 100.0, 100.0, 200.0, 200.0])
        assert set(result["NITRATE_station_id"].values) == {1, 2}

    # Each parameter gets its own independent observation dimension
    def test_xarray_scattered_independent_dims_per_parameter(self, comfort_db):
        pytest.importorskip("xarray")
        result = load_comfort(comfort_db, parameters=["NITRATE", "OXYGEN"])
        assert result.sizes["NITRATE_obs"] == 6
        assert result.sizes["OXYGEN_obs"] == 4

    # max_gap is forwarded to interpolate_depth_levels
    def test_max_gap_drops_wide_bracket(self, comfort_db):
        pytest.importorskip("xarray")
        result = load_comfort(comfort_db, parameters=["NITRATE"], target_depths=[50], max_gap=50)
        assert np.isnan(result["NITRATE"].values).all()

    # A looser max_gap keeps the same target depth
    def test_max_gap_keeps_narrow_bracket(self, comfort_db):
        pytest.importorskip("xarray")
        result = load_comfort(comfort_db, parameters=["NITRATE"], target_depths=[50], max_gap=150)
        assert not np.isnan(result["NITRATE"].values).all()

    # max_gap/interp_method, i.e. interpolation is not applied when target_depths is None (default)
    def test_max_gap_ignored_without_target_depths(self, comfort_db):
        pytest.importorskip("xarray")
        result = load_comfort(comfort_db, parameters=["NITRATE"], max_gap=0, interp_method="bogus")
        assert not np.isnan(result["NITRATE"].values).any()

    # An unknown interp_method raises
    def test_interp_method_invalid_raises(self, comfort_db):
        pytest.importorskip("xarray")
        with pytest.raises(ValueError, match="Unknown method"):
            load_comfort(comfort_db, parameters=["NITRATE"],
                        target_depths=[50], interp_method="bogus")

    # Two stations with the same PROFILE_NUMBER=1 must produce two distinct profiles
    def test_xarray_composite_profile_key(self, tmp_path):
        pytest.importorskip("xarray")
        db_path = str(tmp_path / "shared_pnum.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
        conn.execute("INSERT INTO station VALUES (10, 45.0, -30.0, '2000-01-15')")
        conn.execute("INSERT INTO station VALUES (20, 55.0, -20.0, '2005-06-10')")
        conn.execute(
            "CREATE TABLE P_NITRATE (ID INTEGER, LEV_DBAR REAL, LEV_M REAL, VAL REAL, "
            "PQF1 INTEGER, PQF2 INTEGER, SQF INTEGER, BOTTLE_NUMBER INTEGER, "
            "PROFILE_NUMBER INTEGER, PROFILE_BEST INTEGER, UNITS_ID INTEGER, INSTRUMENT_ID INTEGER)"
        )
        # Both stations use PROFILE_NUMBER=1
        conn.executemany("INSERT INTO P_NITRATE VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
            (10, 0.0, 0.0, 5.0, 1, 3, 0, 1, 1, 1, 3, 1),
            (10, 100.0, 100.0, 4.0, 1, 3, 0, 2, 1, 1, 3, 1),
            (20, 0.0, 0.0, 8.0, 1, 3, 0, 1, 1, 1, 3, 1),
            (20, 100.0, 100.0, 7.0, 1, 3, 0, 2, 1, 1, 3, 1),
        ])
        conn.commit()
        conn.close()
        result = load_comfort(db_path, target_depths=[0, 100])
        # Must produce 2 profiles
        assert result.sizes["profile"] == 2
        assert set(result["station_id"].values) == {10, 20}

    # INSTRUMENT_ID resolves to a name when the INSTRUMENT table exists
    def test_xarray_instrument_resolved_with_table(self, tmp_path):
        pytest.importorskip("xarray")
        db_path = str(tmp_path / "with_instr.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
        conn.execute("INSERT INTO station VALUES (1, 45.0, -30.0, '2000-01-15')")
        conn.execute(
            "CREATE TABLE P_NITRATE (ID INTEGER, LEV_DBAR REAL, LEV_M REAL, VAL REAL, "
            "PQF1 INTEGER, PQF2 INTEGER, SQF INTEGER, BOTTLE_NUMBER INTEGER, "
            "PROFILE_NUMBER INTEGER, PROFILE_BEST INTEGER, UNITS_ID INTEGER, INSTRUMENT_ID INTEGER)"
        )
        conn.executemany("INSERT INTO P_NITRATE VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
            (1, 0.0, 0.0, 5.0, 1, 3, 0, 1, 1, 1, 3, 1),
            (1, 100.0, 100.0, 4.0, 1, 3, 0, 2, 1, 1, 3, 1),
        ])
        conn.execute("CREATE TABLE INSTRUMENT (ID INTEGER PRIMARY KEY, WOD_ID INTEGER, NAME TEXT, NOTES TEXT, DATE_ADDED TEXT, DATE_UPDATED TEXT)")
        conn.execute("INSERT INTO INSTRUMENT VALUES (1, 1, 'CTD', '', '', '')")
        conn.commit()
        conn.close()
        result = load_comfort(db_path, target_depths=[0, 100])
        assert "instrument" in result.coords
        assert "instrument_id" not in result.coords
        assert result["instrument"].values[0] == "CTD"

    # target_depths matching the source LEV_M values exactly requires no interpolation,
    # so each profile's values must come back unchanged and correctly assigned to its station
    def test_xarray_values_match_source_at_exact_depths(self, comfort_db):
        pytest.importorskip("xarray")
        result = load_comfort(comfort_db, target_depths=[0, 100, 200])
        station_ids = result["station_id"].values.tolist()
        nitrate = result["NITRATE"]
        np.testing.assert_allclose(
            nitrate.isel(profile=station_ids.index(1)).values, [5.0, 4.0, 3.0])
        np.testing.assert_allclose(
            nitrate.isel(profile=station_ids.index(2)).values, [8.0, 7.0, 6.0])

    # PLATFORM_ID resolves to a name when the PLATFORM table exists
    def test_xarray_platform_resolved_with_table(self, tmp_path):
        pytest.importorskip("xarray")
        db_path = str(tmp_path / "with_platform.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, "
            "LONGITUDE REAL, DATEANDTIME TEXT, PLATFORM_ID INTEGER)"
        )
        conn.execute("INSERT INTO station VALUES (1, 45.0, -30.0, '2000-01-15', 100)")
        conn.execute(
            "CREATE TABLE P_NITRATE (ID INTEGER, LEV_DBAR REAL, LEV_M REAL, VAL REAL, "
            "PQF1 INTEGER, PQF2 INTEGER, SQF INTEGER, BOTTLE_NUMBER INTEGER, "
            "PROFILE_NUMBER INTEGER, PROFILE_BEST INTEGER, UNITS_ID INTEGER, INSTRUMENT_ID INTEGER)"
        )
        conn.executemany("INSERT INTO P_NITRATE VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
            (1, 0.0, 0.0, 5.0, 1, 3, 0, 1, 1, 1, 3, 1),
            (1, 100.0, 100.0, 4.0, 1, 3, 0, 2, 1, 1, 3, 1),
        ])
        conn.execute(
            "CREATE TABLE PLATFORM (ID INTEGER PRIMARY KEY, NODC_CODE TEXT, WOD_ID INTEGER, "
            "IMO_ID TEXT, CALLSIGN TEXT, COUNTRY_ID INTEGER, NAME TEXT, NAME_NATIVE TEXT, "
            "NOTES_ICES TEXT, NOTES_WOD TEXT, NOTES TEXT, DATE_ADDED TEXT, DATE_UPDATED TEXT)"
        )
        conn.execute("INSERT INTO PLATFORM (ID, NAME) VALUES (100, 'POLARSTERN')")
        conn.commit()
        conn.close()
        result = load_comfort(db_path, target_depths=[0, 100])
        assert "platform" in result.coords
        assert result["platform"].values[0] == "POLARSTERN"

    # A single source unit is reflected verbatim in the xarray "units" attr
    def test_xarray_unit_attrs_single_unit(self, tmp_path):
        pytest.importorskip("xarray")
        db_path = str(tmp_path / "unit_attrs.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
        conn.execute("INSERT INTO station VALUES (1, 45.0, -30.0, '2000-01-15')")
        conn.execute(
            "CREATE TABLE P_NITRATE (ID INTEGER, LEV_DBAR REAL, LEV_M REAL, VAL REAL, "
            "PQF1 INTEGER, PQF2 INTEGER, SQF INTEGER, BOTTLE_NUMBER INTEGER, "
            "PROFILE_NUMBER INTEGER, PROFILE_BEST INTEGER, UNITS_ID INTEGER, INSTRUMENT_ID INTEGER)"
        )
        conn.executemany("INSERT INTO P_NITRATE VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
            (1, 0.0, 0.0, 5.0, 1, 3, 0, 1, 1, 1, 3, 1),
            (1, 100.0, 100.0, 4.0, 1, 3, 0, 2, 1, 1, 3, 1),
        ])
        conn.execute("CREATE TABLE UNITS (ID INTEGER PRIMARY KEY, NAME_SHORT TEXT)")
        conn.execute("INSERT INTO UNITS VALUES (3, 'µmol/kg')")
        conn.commit()
        conn.close()
        result = load_comfort(db_path, target_depths=[0, 100])
        assert result["NITRATE"].attrs["units"] == "µmol/kg"

    # Mixed source units for one parameter both warn and are still merged into one variable
    def test_load_comfort_warns_on_mixed_units(self, tmp_path, caplog):
        pytest.importorskip("xarray")
        db_path = str(tmp_path / "mixed_units.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
        conn.execute("INSERT INTO station VALUES (1, 45.0, -30.0, '2000-01-15')")
        conn.execute(
            "CREATE TABLE P_NITRATE (ID INTEGER, LEV_DBAR REAL, LEV_M REAL, VAL REAL, "
            "PQF1 INTEGER, PQF2 INTEGER, SQF INTEGER, BOTTLE_NUMBER INTEGER, "
            "PROFILE_NUMBER INTEGER, PROFILE_BEST INTEGER, UNITS_ID INTEGER, INSTRUMENT_ID INTEGER)"
        )
        conn.executemany("INSERT INTO P_NITRATE VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
            (1, 0.0, 0.0, 5.0, 1, 3, 0, 1, 1, 1, 3, 1),
            (1, 100.0, 100.0, 4.0, 1, 3, 0, 2, 1, 1, 7, 1),
        ])
        conn.execute("CREATE TABLE UNITS (ID INTEGER PRIMARY KEY, NAME_SHORT TEXT)")
        conn.execute("INSERT INTO UNITS VALUES (3, 'µmol/kg')")
        conn.execute("INSERT INTO UNITS VALUES (7, 'µmol/L')")
        conn.commit()
        conn.close()
        with caplog.at_level(logging.WARNING):
            result = load_comfort(db_path, target_depths=[0, 100])
        assert "different units" in caplog.text
        assert "NITRATE" in caplog.text
        assert "µmol/kg" in result["NITRATE"].attrs.get("units", "")


class TestReadParameter:
    """Tests for read_parameter with spatial/temporal filters."""

    # LATITUDE/LONGITUDE/DATEANDTIME are always joined in
    def test_station_columns_always_included(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        df = read_parameter(conn, "NITRATE")
        conn.close()
        assert "LATITUDE" in df.columns
        assert "LONGITUDE" in df.columns
        assert "DATEANDTIME" in df.columns
        assert len(df) == 6

    # lat_min restricts rows to the matching station's profile
    def test_lat_filter(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        df = read_parameter(conn, "NITRATE", lat_min=50.0)
        conn.close()
        assert (df["LATITUDE"] >= 50.0).all()
        assert df["PROFILE_NUMBER"].nunique() == 1

    # depth_max restricts rows to the given depth bound
    def test_depth_filter(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        df = read_parameter(conn, "NITRATE", depth_max=100.0)
        conn.close()
        assert (df["LEV_M"] <= 100.0).all()

    # date_min restricts rows to the matching station's profile
    def test_date_filter(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        df = read_parameter(conn, "NITRATE", date_min="2003-01-01")
        conn.close()
        assert len(df) > 0
        assert df["PROFILE_NUMBER"].nunique() == 1

    # quality_flags and a geo filter combine correctly in the same query
    def test_qc_and_geo_combined(self, comfort_db):
        from comfort.qc import QC_GOOD
        conn = sqlite3.connect(comfort_db)
        df = read_parameter(conn, "NITRATE", quality_flags=QC_GOOD, lat_min=40.0, lat_max=50.0)
        conn.close()
        assert (df["LATITUDE"] >= 40.0).all()
        assert (df["LATITUDE"] <= 50.0).all()

    # profile and a geo filter combine correctly in the same query
    def test_profiles_and_geo_combined(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        df = read_parameter(conn, "NITRATE", profiles=[1], lat_min=50.0)
        conn.close()
        assert len(df) == 3
        assert (df["PROFILE_NUMBER"] == 1).all()
        assert (df["LATITUDE"] == 55.0).all()

    # region restricts rows to the matching station via an arbitrary polygon
    @needs_shapely
    def test_region_filter(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        region = {"box": [(-25, 50), (-15, 50), (-15, 60), (-25, 60)]}
        df = read_parameter(conn, "NITRATE", region=region)
        conn.close()
        assert len(df) == 3
        assert (df["LATITUDE"] == 55.0).all()

    # A wrapped region box and explicit lon bounds both produce SQL conditions (ANDed)
    def test_wrapped_region_box_and_explicit_lon_are_anded(self):
        from comfort.io import _append_geo_conditions
        conditions, params = [], []
        _append_geo_conditions(
            conditions, params, "s.", "s.", "p.", "s.",
            None, None, -180.0, 0.0, None, None, None, None,
            region_box=(-10.0, 10.0, 170.0, -170.0),
        )
        assert "s.LONGITUDE >= ?" in conditions
        assert "s.LONGITUDE <= ?" in conditions
        assert "(s.LONGITUDE >= ? OR s.LONGITUDE <= ?)" in conditions
        assert params == [-180.0, 0.0, -10.0, 10.0, 170.0, -170.0]

    # extra_station_cols pulls additional station columns (e.g. CRUISE_ID)
    def test_extra_station_cols(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        conn.execute("ALTER TABLE station ADD COLUMN CRUISE_ID INTEGER")
        conn.execute("ALTER TABLE station ADD COLUMN ST_NUMBER_ORIGIN TEXT")
        conn.execute("UPDATE station SET CRUISE_ID=100, ST_NUMBER_ORIGIN='S-01' WHERE ID=1")
        conn.execute("UPDATE station SET CRUISE_ID=200, ST_NUMBER_ORIGIN='S-02' WHERE ID=2")
        conn.commit()

        default_df = read_parameter(conn, "NITRATE")
        assert "CRUISE_ID" not in default_df.columns

        df = read_parameter(conn, "NITRATE", extra_station_cols=["CRUISE_ID", "ST_NUMBER_ORIGIN"])
        conn.close()
        assert "CRUISE_ID" in df.columns
        assert "ST_NUMBER_ORIGIN" in df.columns
        assert set(df.loc[df["LATITUDE"] == 45.0, "CRUISE_ID"]) == {100}
        assert set(df.loc[df["LATITUDE"] == 55.0, "ST_NUMBER_ORIGIN"]) == {"S-02"}


class TestInfo:
    # Output is a DataFrame
    def test_returns_dataframe(self, comfort_db):
        result = info(comfort_db)
        assert isinstance(result, pd.DataFrame)

    # The summary table exposes the documented column set
    def test_has_expected_columns(self, comfort_db):
        result = info(comfort_db)
        assert {"parameter", "n_samples", "depth_min_m", "depth_max_m"}.issubset(result.columns)

    # Every parameter table in the database is listed
    def test_correct_parameter_list(self, comfort_db):
        result = info(comfort_db)
        assert set(result["parameter"]) == {"NITRATE", "OXYGEN"}

    # Sample counts match the number of rows per parameter table
    def test_correct_sample_counts(self, comfort_db):
        result = info(comfort_db).set_index("parameter")
        assert result.loc["NITRATE", "n_samples"] == 6
        assert result.loc["OXYGEN", "n_samples"] == 4

    # date_min/date_max columns are always present
    def test_date_range_present(self, comfort_db):
        result = info(comfort_db)
        assert "date_min" in result.columns
        assert "date_max" in result.columns

    # info() accepts an already-open connection, not just a file path
    def test_accepts_connection(self, comfort_db):
        conn = sqlite3.connect(comfort_db)
        result = info(conn)
        conn.close()
        assert len(result) == 2


class TestLoadComfortLimit:
    # limit argument caps the number of rows returned
    def test_limit_restricts_rows(self, comfort_db):
        result = load_comfort(comfort_db, parameters=["NITRATE"], as_xarray=False, limit=2)
        assert len(result["NITRATE"]) <= 2

    # limit=None (default) returns every matching row
    def test_limit_none_returns_all(self, comfort_db):
        result = load_comfort(comfort_db, parameters=["NITRATE"], as_xarray=False, limit=None)
        assert len(result["NITRATE"]) == 6


class TestLoadComfortUnitConversion:
    """Tests for load_comfort(convert_units=True): T/S enrichment, on_missing policy, unit attrs."""

    _param_ddl = (
        "CREATE TABLE {tbl} (ID INTEGER, LEV_DBAR REAL, LEV_M REAL, VAL REAL, "
        "PQF1 INTEGER, PQF2 INTEGER, SQF INTEGER, BOTTLE_NUMBER INTEGER, "
        "PROFILE_NUMBER INTEGER, PROFILE_BEST INTEGER, UNITS_ID INTEGER, INSTRUMENT_ID INTEGER)"
    )

    @pytest.fixture
    def oxygen_db(self, tmp_path):
        """Factory: P_OXYGEN with one percent-saturation row and one µmol/kg row."""
        def _make(with_ts_tables=True):
            db_path = str(tmp_path / "convert.sqlite")
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
            conn.execute("INSERT INTO station VALUES (1, 45.0, -30.0, '2000-01-15')")
            conn.execute(self._param_ddl.format(tbl="P_OXYGEN"))
            conn.executemany("INSERT INTO P_OXYGEN VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
                (1, 0.0, 0.0, 80.0, 1, 3, 0, 1, 1, 1, 10, 1),
                (1, 100.0, 100.0, 200.0, 1, 3, 0, 2, 1, 1, 3, 1),
            ])
            if with_ts_tables:
                conn.execute(self._param_ddl.format(tbl="P_TEMPERATURE"))
                conn.execute("INSERT INTO P_TEMPERATURE VALUES (1, 0.0, 0.0, 10.0, 1, 3, 0, 1, 1, 1, 1, 1)")
                conn.execute(self._param_ddl.format(tbl="P_SALINITY"))
                conn.execute("INSERT INTO P_SALINITY VALUES (1, 0.0, 0.0, 35.0, 1, 3, 0, 1, 1, 1, 2, 1)")
            conn.execute("CREATE TABLE UNITS (ID INTEGER PRIMARY KEY, NAME_SHORT TEXT)")
            conn.executemany("INSERT INTO UNITS VALUES (?,?)", [(3, "µmol/kg"), (10, "%")])
            conn.execute("CREATE TABLE DATABASE_TABLES (NAME_TABLE TEXT, UNITS_ID_DEFAULT INTEGER)")
            conn.execute("INSERT INTO DATABASE_TABLES VALUES ('P_OXYGEN', 3)")
            conn.commit()
            conn.close()
            return db_path
        return _make

    # The join must supply LEV_DBAR/LATITUDE/LONGITUDE alongside temperature/salinity
    def test_join_supplies_pressure_and_geo_alongside_ts(self, tmp_path):
        db_path = str(tmp_path / "geo_join.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
        conn.execute("INSERT INTO station VALUES (1, 45.0, -30.0, '2000-01-15')")
        conn.execute(self._param_ddl.format(tbl="P_OXYGEN"))
        conn.execute("INSERT INTO P_OXYGEN VALUES (1, 2000.0, 1975.0, 80.0, 1, 3, 0, 1, 1, 1, 10, 1)")
        conn.execute(self._param_ddl.format(tbl="P_TEMPERATURE"))
        conn.execute("INSERT INTO P_TEMPERATURE VALUES (1, 2000.0, 1975.0, 10.0, 1, 3, 0, 1, 1, 1, 1, 1)")
        conn.execute(self._param_ddl.format(tbl="P_SALINITY"))
        conn.execute("INSERT INTO P_SALINITY VALUES (1, 2000.0, 1975.0, 35.0, 1, 3, 0, 1, 1, 1, 2, 1)")
        conn.commit()
        conn.close()

        conn = sqlite3.connect(db_path)
        df = _read_with_geo(conn, "OXYGEN", quality_flags=None, with_temperature=True, with_salinity=True)
        conn.close()

        assert {"LEV_DBAR", "LATITUDE", "LONGITUDE"}.issubset(df.columns)
        assert df["temperature"].iloc[0] == 10.0
        assert df["salinity"].iloc[0] == 35.0

    # A leftover E_* view (from older COMFORT databases) must not change the result:
    # T/S are still joined, so the percent-saturation row is converted, not dropped
    def test_leftover_extended_view_is_ignored(self, oxygen_db):
        db_path = oxygen_db(with_ts_tables=True)
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE VIEW E_OXYGEN AS SELECT p.*, s.LATITUDE, s.LONGITUDE, s.DATEANDTIME "
            "FROM P_OXYGEN p JOIN station s ON p.ID = s.ID"
        )
        conn.commit()
        conn.close()

        result = load_comfort(db_path, parameters=["OXYGEN"], convert_units=True, as_xarray=False)
        df = result["OXYGEN"].sort_values("LEV_M")
        assert len(df) == 2
        assert (df["UNITS_ID"] == 3).all()

    # Default on_missing="drop": A row needing T/S but missing it is dropped, logged
    def test_unconvertible_rows_dropped_with_warning(self, oxygen_db, caplog):
        db_path = oxygen_db(with_ts_tables=False)
        with caplog.at_level(logging.WARNING):
            result = load_comfort(db_path, parameters=["OXYGEN"], convert_units=True, as_xarray=False)
        df = result["OXYGEN"]
        assert len(df) == 1
        assert df["OXYGEN"].iloc[0] == 200.0
        assert "missing temperature/salinity" in caplog.text

    # on_missing="keep" keeps the unconverted row instead of dropping it
    def test_unconvertible_rows_kept_on_request(self, oxygen_db):
        db_path = oxygen_db(with_ts_tables=False)
        result = load_comfort(db_path, parameters=["OXYGEN"], convert_units=True, on_missing="keep", as_xarray=False)
        df = result["OXYGEN"].sort_values("LEV_M")
        assert len(df) == 2
        assert df["OXYGEN"].iloc[0] == 80.0
        assert df["UNITS_ID"].iloc[0] == 10

    # on_missing="raise" turns the same situation into a hard error
    def test_unconvertible_rows_raise_on_request(self, oxygen_db):
        db_path = oxygen_db(with_ts_tables=False)
        with pytest.raises(ValueError, match="could not be converted"):
            load_comfort(db_path, parameters=["OXYGEN"], convert_units=True, on_missing="raise", as_xarray=False)

    # The xarray "units" attr reflects the post-conversion unit, not the mixed source units
    def test_xarray_units_attr_reflects_converted_unit(self, tmp_path):
        pytest.importorskip("xarray")
        db_path = str(tmp_path / "convert_attrs.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
        conn.execute("INSERT INTO station VALUES (1, 45.0, -30.0, '2000-01-15')")
        conn.execute(self._param_ddl.format(tbl="P_NITRATE"))
        conn.executemany("INSERT INTO P_NITRATE VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
            (1, 0.0, 0.0, 5.0, 1, 3, 0, 1, 1, 1, 3, 1),
            (1, 100.0, 100.0, 10.0, 1, 3, 0, 2, 1, 1, 7, 1),
        ])
        conn.execute("CREATE TABLE UNITS (ID INTEGER PRIMARY KEY, NAME_SHORT TEXT)")
        conn.executemany("INSERT INTO UNITS VALUES (?,?)", [(3, "µmol/kg"), (7, "mmol/L")])
        conn.execute("CREATE TABLE DATABASE_TABLES (NAME_TABLE TEXT, UNITS_ID_DEFAULT INTEGER)")
        conn.execute("INSERT INTO DATABASE_TABLES VALUES ('P_NITRATE', 3)")
        conn.commit()
        conn.close()

        result = load_comfort(db_path, convert_units=True, target_depths=[0, 100])
        # Attrs must reflect the unit after conversion, not the source units
        assert result["NITRATE"].attrs["units"] == "µmol/kg"

    # The join must supply salinity for use_density=True
    def test_join_supplies_salinity_for_dic(self, tmp_path):
        db_path = str(tmp_path / "dic_join.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
        conn.execute("INSERT INTO station VALUES (1, 45.0, -30.0, '2000-01-15')")
        conn.execute(self._param_ddl.format(tbl="P_DIC"))
        conn.execute("INSERT INTO P_DIC VALUES (1, 0.0, 0.0, 2.2, 1, 3, 0, 1, 1, 1, 7, 1)")
        conn.execute(self._param_ddl.format(tbl="P_SALINITY"))
        conn.execute("INSERT INTO P_SALINITY VALUES (1, 0.0, 0.0, 35.0, 1, 3, 0, 1, 1, 1, 2, 1)")
        conn.commit()
        conn.close()

        conn = sqlite3.connect(db_path)
        df = _read_with_geo(conn, "DIC", quality_flags=None, with_salinity=True)
        conn.close()
        assert df["salinity"].iloc[0] == 35.0

    # use_density=True must actually change load_comfort's output
    def test_use_density_changes_result_vs_constant_fallback(self, tmp_path):
        db_path = str(tmp_path / "convert_dens.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
        conn.execute("INSERT INTO station VALUES (1, 45.0, -30.0, '2000-01-15')")
        conn.execute(self._param_ddl.format(tbl="P_DIC"))
        conn.execute("INSERT INTO P_DIC VALUES (1, 0.0, 0.0, 2.2, 1, 3, 0, 1, 1, 1, 7, 1)")
        conn.execute(self._param_ddl.format(tbl="P_SALINITY"))
        conn.execute("INSERT INTO P_SALINITY VALUES (1, 0.0, 0.0, 35.0, 1, 3, 0, 1, 1, 1, 2, 1)")
        conn.execute("CREATE TABLE UNITS (ID INTEGER PRIMARY KEY, NAME_SHORT TEXT)")
        conn.executemany("INSERT INTO UNITS VALUES (?,?)", [(3, "umol/kg"), (7, "mmol/L")])
        conn.execute("CREATE TABLE DATABASE_TABLES (NAME_TABLE TEXT, UNITS_ID_DEFAULT INTEGER)")
        conn.execute("INSERT INTO DATABASE_TABLES VALUES ('P_DIC', 3)")
        conn.commit()
        conn.close()

        with_density = load_comfort(db_path, parameters=["DIC"], convert_units=True, use_density=True, as_xarray=False)
        constant_density = load_comfort(db_path, parameters=["DIC"], convert_units=True, use_density=False, as_xarray=False)
        assert not np.isclose(with_density["DIC"]["DIC"].iloc[0], constant_density["DIC"]["DIC"].iloc[0])

    # A PQF2<=2 salinity duplicate must not enter the joined salinity average
    def test_ts_join_excludes_bad_quality_salinity(self, tmp_path):
        from comfort.qc import QC_GOOD

        db_path = str(tmp_path / "dic_join_qc.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
        conn.execute("INSERT INTO station VALUES (1, 45.0, -30.0, '2000-01-15')")
        conn.execute(self._param_ddl.format(tbl="P_DIC"))
        conn.execute("INSERT INTO P_DIC VALUES (1, 0.0, 0.0, 2.2, 1, 3, 0, 1, 1, 1, 7, 1)")
        conn.execute(self._param_ddl.format(tbl="P_SALINITY"))
        conn.executemany("INSERT INTO P_SALINITY VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
            (1, 0.0, 0.0, 35.0, 1, 3, 0, 1, 1, 1, 2, 1),
            (1, 0.0, 0.0, 99.0, 1, 1, 0, 2, 1, 1, 2, 1),  # bad PQF2, must be excluded
        ])
        conn.commit()
        conn.close()

        conn = sqlite3.connect(db_path)
        df = _read_with_geo(conn, "DIC", QC_GOOD, with_salinity=True)
        conn.close()
        assert df["salinity"].iloc[0] == 35.0

    # T/S-like params (targets degC/PSU) must not join T/S onto themselves
    def test_no_ts_join_for_non_density_targets(self, tmp_path):
        db_path = str(tmp_path / "no_selfjoin.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
        conn.execute("INSERT INTO station VALUES (1, 45.0, -30.0, '2000-01-15')")
        conn.execute(self._param_ddl.format(tbl="P_SALINITY"))
        conn.execute("INSERT INTO P_SALINITY VALUES (1, 0.0, 0.0, 35.0, 1, 3, 0, 1, 1, 1, 2, 1)")
        conn.execute("CREATE TABLE UNITS (ID INTEGER PRIMARY KEY, NAME_SHORT TEXT)")
        conn.execute("INSERT INTO UNITS VALUES (2, 'PSU')")
        conn.execute("CREATE TABLE DATABASE_TABLES (NAME_TABLE TEXT, UNITS_ID_DEFAULT INTEGER)")
        conn.execute("INSERT INTO DATABASE_TABLES VALUES ('P_SALINITY', 2)")
        conn.commit()
        conn.close()

        result = load_comfort(db_path, parameters=["SALINITY"], convert_units=True,
                              use_density=True, as_xarray=False)
        df = result["SALINITY"]
        assert "salinity" not in df.columns
        assert "temperature" not in df.columns
        assert df["SALINITY"].iloc[0] == 35.0


class TestLoadComfortErrorHandling:
    """Tests for load_comfort's parameter check and the missing-lookup-table check."""

    # An unknown parameter raises before anything is loaded
    def test_unknown_parameter_raises(self, comfort_db):
        with pytest.raises(ValueError, match="Unknown parameters.*DOESNOTEXIST"):
            load_comfort(comfort_db, parameters=["NITRATE", "DOESNOTEXIST"], as_xarray=False)

    # Parameter names are matched case-insensitively
    def test_parameter_check_is_case_insensitive(self, comfort_db):
        result = load_comfort(comfort_db, parameters=["nitrate"], as_xarray=False)
        assert len(result) == 1

    # convert_units=True without UNITS/DATABASE_TABLES raises instead of silently skipping conversion
    def test_convert_units_without_lookup_tables_raises(self, comfort_db):
        with pytest.raises(ValueError):
            load_comfort(comfort_db, parameters=["NITRATE"], convert_units=True, as_xarray=False)
