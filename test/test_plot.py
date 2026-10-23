"""Smoke tests for the comfort.plot subpackage.

All tests use an in-memory SQLite database with synthetic data.
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed ([plot] extra)")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.contour import QuadContourSet

import comfort
from comfort.database import (
    count_samples_over_time,
    find_spatiotemporal_duplicates,
)
from comfort.plot import (
    boxplot,
    compare_stats,
    plot_annual_coverage,
    plot_correlation,
    plot_counts_bar,
    plot_depth_coverage,
    plot_duplicates,
    plot_histogram,
    plot_joint,
    plot_lat_lon_range,
    plot_missing_value_info,
    plot_missing_value_info_map_joint,
    plot_missing_value_info_map_over_depth,
    plot_monthly_histogram,
    plot_parameter_map,
    plot_profile,
    plot_section,
    plot_spatial_distribution,
    plot_ts_diagram,
)

_has_cartopy = True
try:
    import cartopy
except ImportError:
    _has_cartopy = False


@pytest.fixture()
def demo_conn():
    """In-memory COMFORT database with T, S, O2 (20 stations, 6 depths)."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, "
        "LONGITUDE REAL, DATEANDTIME TEXT)"
    )
    np.random.seed(42)
    n_stations = 20
    lats = np.random.uniform(30, 65, n_stations)
    lons = np.random.uniform(-60, 0, n_stations)
    years = np.random.randint(2000, 2020, n_stations)
    conn.executemany(
        "INSERT INTO station VALUES (?,?,?,?)",
        [(i, lats[i], lons[i], f"{years[i]}-06-15") for i in range(n_stations)],
    )
    ddl = (
        "CREATE TABLE {t} (ID INTEGER, LEV_DBAR REAL, LEV_M REAL, VAL REAL, "
        "PQF1 INTEGER, PQF2 INTEGER, SQF INTEGER, BOTTLE_NUMBER INTEGER, "
        "PROFILE_NUMBER INTEGER, PROFILE_BEST INTEGER, UNITS_ID INTEGER, "
        "INSTRUMENT_ID INTEGER)"
    )
    depths = [0, 50, 100, 200, 500, 1000]
    for t in ("P_TEMPERATURE", "P_SALINITY", "P_OXYGEN"):
        conn.execute(ddl.format(t=t))
    rows_t, rows_s, rows_o = [], [], []
    for sid in range(n_stations):
        for j, d in enumerate(depths):
            base = (sid, float(d), float(d))
            suffix = (1, 3, 0, j + 1, sid, 1, 1, 1)
            rows_t.append((*base, 18.0 - d * 0.012 + np.random.normal(0, 0.5), *suffix))
            rows_s.append((*base, 35.0 + d * 0.001 + np.random.normal(0, 0.1), *suffix))
            rows_o.append((*base, 250.0 - d * 0.15 + np.random.normal(0, 5), *suffix))
    conn.executemany("INSERT INTO P_TEMPERATURE VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows_t)
    conn.executemany("INSERT INTO P_SALINITY VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows_s)
    conn.executemany("INSERT INTO P_OXYGEN VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows_o)
    conn.commit()
    yield conn
    conn.close()


#--- Profile and section plots ---------------------------------------#

class TestPlotProfile:
    # A DataFrame with a profiles filter draws only the requested profiles
    def test_from_dataframe(self, demo_conn):
        df = comfort.io.read_parameter(demo_conn, "TEMPERATURE")
        ax = plot_profile(df, profiles=[0, 1])
        assert len(ax.lines) == 2  # one line per requested profile
        plt.close()

    # conn + parameter is the alternative to passing a DataFrame directly
    def test_from_database(self, demo_conn):
        ax = plot_profile(conn=demo_conn, parameter="TEMPERATURE", profiles=[0, 1, 2])
        assert ax is not None
        plt.close()

    # A profiles filter matching nothing returns None instead of an empty plot
    def test_empty_returns_none(self, demo_conn):
        df = comfort.io.read_parameter(demo_conn, "TEMPERATURE")
        ax = plot_profile(df, profiles=[9999])
        assert ax is None

    # show_points=True (default) marks each measurement on the profile line
    def test_show_points_default_adds_markers(self, demo_conn):
        df = comfort.io.read_parameter(demo_conn, "TEMPERATURE")
        ax = plot_profile(df, profiles=[0])
        assert ax.lines[0].get_marker() != "None"
        plt.close()

    # show_points=False draws a plain line with no markers
    def test_show_points_false_draws_no_markers(self, demo_conn):
        df = comfort.io.read_parameter(demo_conn, "TEMPERATURE")
        ax = plot_profile(df, profiles=[0], show_points=False)
        assert ax.lines[0].get_marker() == "None"
        plt.close()

    # Section plot works when loading straight from the database
    def test_section_from_database(self, demo_conn):
        ax = plot_section(conn=demo_conn, parameter="OXYGEN")
        assert ax is not None
        plt.close()

    # Section plot also works from an already-loaded DataFrame
    def test_section_from_dataframe(self, demo_conn):
        df = comfort.io.read_parameter(demo_conn, "OXYGEN")
        ax = plot_section(df)
        assert ax is not None
        plt.close()

    # binned=True draws an averaged pcolormesh instead of a raw scatter
    def test_section_binned(self, demo_conn):
        df = comfort.io.read_parameter(demo_conn, "OXYGEN")
        ax = plot_section(df, binned=True, along_bins=5, depth_bins=3)
        assert any(isinstance(c, QuadMesh) for c in ax.collections)
        plt.close()


#--- T-S diagram -----------------------------------------------------#

class TestPlotTSDiagram:
    # TEOS-10 variables (SA/CT) draw isopycnal contours
    def test_with_density_contours(self, demo_conn):
        dfs = comfort.load_comfort(demo_conn, parameters=["TEMPERATURE", "SALINITY"],
                                   as_xarray=False)
        df = dfs["TEMPERATURE"].rename(columns={"TEMPERATURE": "T"})
        df["S"] = dfs["SALINITY"]["SALINITY"].values
        ax = plot_ts_diagram(df, temp_col="T", sal_col="S", temp_type="CT", sal_type="SA")
        assert any(isinstance(c, QuadContourSet) for c in ax.collections)
        plt.close()

    # density_contours=False skips the isopycnal overlay
    def test_without_contours(self, demo_conn):
        dfs = comfort.load_comfort(demo_conn, parameters=["TEMPERATURE", "SALINITY"],
                                   as_xarray=False)
        df = dfs["TEMPERATURE"].rename(columns={"TEMPERATURE": "T"})
        df["S"] = dfs["SALINITY"]["SALINITY"].values
        ax = plot_ts_diagram(df, temp_col="T", sal_col="S", density_contours=False)
        assert not any(isinstance(c, QuadContourSet) for c in ax.collections)
        plt.close()

    # An empty input DataFrame returns None instead of an empty plot
    def test_empty_returns_none(self):
        df = pd.DataFrame({"T": [], "S": [], "LEV_M": []})
        ax = plot_ts_diagram(df, temp_col="T", sal_col="S")
        assert ax is None

    def test_axis_labels_teos10(self):
        """Default SA/CT types produce proper TEOS-10 axis labels."""
        df = pd.DataFrame({"CT": [10.0, 12.0], "SA": [35.0, 35.5], "LEV_M": [10, 20]})
        ax = plot_ts_diagram(df, temp_col="CT", sal_col="SA", temp_type="CT", sal_type="SA",
                             density_contours=False)
        assert r"$S_A$" in ax.get_xlabel()
        assert "g kg" in ax.get_xlabel()
        assert r"$\Theta$" in ax.get_ylabel()
        assert "°C" in ax.get_ylabel()
        plt.close()

    def test_axis_labels_practical(self):
        """SP/t types produce appropriate fallback labels."""
        df = pd.DataFrame({"T": [10.0, 12.0], "S": [35.0, 35.5], "LEV_M": [10, 20]})
        ax = plot_ts_diagram(df, temp_col="T", sal_col="S", temp_type="t", sal_type="SP",
                             density_contours=False)
        assert r"$S_P$" in ax.get_xlabel()
        assert r"$T$" in ax.get_ylabel()
        plt.close()

    def test_contour_warning_non_teos10(self, caplog):
        """Warning is emitted when density contours are drawn with non-TEOS-10 variables."""
        import logging
        df = pd.DataFrame({"T": [10.0, 12.0], "S": [35.0, 35.5], "LEV_M": [10, 20]})
        with caplog.at_level(logging.WARNING, logger="comfort.plot.relationships"):
            ax = plot_ts_diagram(df, temp_col="T", sal_col="S", temp_type="t", sal_type="SP",
                                 density_contours=True)
        assert ax is not None
        assert any("inaccurate" in r.message for r in caplog.records)
        plt.close()


#--- Correlation -----------------------------------------------------#

class TestPlotCorrelation:
    # Scatter + regression works directly on a DataFrame
    def test_from_dataframe(self):
        df = pd.DataFrame({"x": np.random.randn(50), "y": np.random.randn(50)})
        ax = plot_correlation(df, "x", "y")
        assert len(ax.lines) == 1  # the regression line (scatter itself is a collection)
        plt.close()

    # conn + a pair of parameters is the alternative to a DataFrame
    def test_from_database(self, demo_conn):
        ax = plot_correlation(conn=demo_conn, parameters=("TEMPERATURE", "OXYGEN"))
        assert ax is not None
        plt.close()

    # plot_joint wraps seaborn's JointGrid for the same database-mode input
    def test_joint_from_database(self, demo_conn):
        g = plot_joint(conn=demo_conn, parameters=("TEMPERATURE", "SALINITY"), kind="scatter")
        assert g is not None
        plt.close()


#--- Distributions ---------------------------------------------------#

class TestDistributions:
    # Default boxplot kind, loaded straight from the database
    def test_boxplot_from_database(self, demo_conn):
        ax = boxplot(conn=demo_conn, parameter="OXYGEN")
        assert ax is not None
        plt.close()

    # kind="violin" renders a filled density curve, not a box-and-whisker patch
    def test_violin_from_database(self, demo_conn):
        ax = boxplot(conn=demo_conn, parameter="OXYGEN", kind="violin")
        assert len(ax.collections) > 0  # violin body is a PolyCollection; boxplot has none
        plt.close()

    # Histogram loaded straight from the database
    def test_histogram_from_database(self, demo_conn):
        ax = plot_histogram(conn=demo_conn, parameter="TEMPERATURE")
        assert ax is not None
        plt.close()

    # Twelve-panel monthly histogram, loaded straight from the database
    def test_monthly_histogram_from_database(self, demo_conn):
        fig = plot_monthly_histogram(conn=demo_conn, parameter="TEMPERATURE")
        assert fig is not None
        assert len(fig.axes) == 12
        plt.close(fig)

    # compare_stats draws comparative boxplots across a grouping column
    def test_compare_stats(self, demo_conn):
        dfs = comfort.load_comfort(demo_conn, parameters=["TEMPERATURE", "SALINITY", "OXYGEN"], as_xarray=False)
        df = dfs["TEMPERATURE"][["PROFILE_NUMBER", "LEV_M"]].copy()
        df["TEMPERATURE"] = dfs["TEMPERATURE"]["TEMPERATURE"].values
        df["SALINITY"] = dfs["SALINITY"]["SALINITY"].values
        df["OXYGEN"] = dfs["OXYGEN"]["OXYGEN"].values
        df["depth_bin"] = pd.cut(df["LEV_M"], bins=[0, 200, 2000], labels=["Shallow", "Deep"])
        df = df.dropna(subset=["depth_bin"])
        ax = compare_stats(df, group_col="depth_bin", value_cols=["TEMPERATURE", "SALINITY", "OXYGEN"])
        assert [t.get_text() for t in ax.get_xticklabels()] == ["TEMPERATURE", "SALINITY", "OXYGEN"]
        assert {t.get_text() for t in ax.get_legend().get_texts()} == {"Shallow", "Deep"}
        plt.close()


#--- Counting --------------------------------------------------------#

class TestCounting:
    # The counts DataFrame renders as a bar chart, one bar per row
    def test_plot_counts_bar(self, demo_conn):
        counts = count_samples_over_time(demo_conn, "TEMPERATURE", time_mode="year")
        ax = plot_counts_bar(counts)
        assert len(ax.patches) == len(counts)
        plt.close()


#--- Annual coverage -------------------------------------------------#

class TestPlotAnnualCoverage:
    # One line per parameter dict entry
    def test_basic(self, demo_conn):
        dfs = {
            "TEMPERATURE": comfort.io.read_parameter(demo_conn, "TEMPERATURE"),
            "SALINITY": comfort.io.read_parameter(demo_conn, "SALINITY"),
        }
        ax = plot_annual_coverage(dfs)
        assert ax is not None
        assert len(ax.lines) == 2
        plt.close()

    # normalise=True works
    def test_normalized(self, demo_conn):
        dfs = {"TEMPERATURE": comfort.io.read_parameter(demo_conn, "TEMPERATURE")}
        ax = plot_annual_coverage(dfs, normalise=True)
        assert ax is not None
        plt.close()

    # A caller-supplied ax is drawn into and returned as-is
    def test_custom_ax(self, demo_conn):
        dfs = {"TEMPERATURE": comfort.io.read_parameter(demo_conn, "TEMPERATURE")}
        fig, ax = plt.subplots()
        result = plot_annual_coverage(dfs, ax=ax)
        assert result is ax
        plt.close()


#--- Depth coverage --------------------------------------------------#

class TestPlotDepthCoverage:
    # One line per parameter column, y-axis inverted (deeper = lower)
    def test_basic(self):
        df_wide = pd.DataFrame({
            "LEV_M": [0, 0, 50, 50, 100],
            "P_TEMPERATURE": [10, 11, np.nan, 9, 8],
            "P_SALINITY": [35, 35, 35, np.nan, np.nan],
        })
        ax = plot_depth_coverage(df_wide, ["P_TEMPERATURE", "P_SALINITY"])
        assert ax is not None
        assert len(ax.lines) == 2
        assert ax.yaxis_inverted()
        plt.close()

    # A caller-supplied ax is drawn into and returned as-is
    def test_custom_ax(self):
        df_wide = pd.DataFrame({"LEV_M": [0, 50], "P_TEMPERATURE": [10, 9]})
        fig, ax = plt.subplots()
        result = plot_depth_coverage(df_wide, ["P_TEMPERATURE"], ax=ax)
        assert result is ax
        plt.close()


#--- Dual-mode error handling ----------------------------------------#

class TestDualModeErrors:
    # Neither df nor conn/parameter given is a hard error
    def test_no_data_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            plot_histogram()

    # conn without parameter is equally invalid
    def test_conn_without_parameter_raises(self, demo_conn):
        with pytest.raises(ValueError, match="Provide either"):
            plot_histogram(conn=demo_conn)


class TestPlotDuplicates:
    # The plot draws the summary chart from the finder output
    def test_plot_duplicates(self, demo_conn):
        demo_conn.execute("INSERT INTO P_TEMPERATURE VALUES (0,0.0,0.0,19.0,1,3,0,7,0,1,1,1)")
        demo_conn.commit()
        summary, details = find_spatiotemporal_duplicates(demo_conn, ["TEMPERATURE"])
        ax = plot_duplicates(summary, details)
        assert ax is not None
        plt.close("all")

    # Without duplicates there is nothing to plot
    def test_plot_duplicates_empty(self, demo_conn):
        summary, details = find_spatiotemporal_duplicates(demo_conn, ["TEMPERATURE"])
        assert plot_duplicates(summary, details) is None


#--- Missing-value bar chart -----------------------------------------#

class TestPlotMissingValueInfo:
    # Bar chart of per-parameter missing-value fractions, one bar per parameter
    def test_basic(self):
        num_nulls = pd.DataFrame({"parameter": ["P_NITRATE", "P_OXYGEN"], "relative": [10.0, 25.0]})
        ax = plot_missing_value_info(num_nulls)
        assert len(ax.patches) == 2
        plt.close()


#--- Spatial / map plots (require cartopy) ---------------------------#

def _missing_value_wide_df():
    data = [
        (10.0, 10.0, 0.0, 5.0),
        (10.0, 10.0, 50.0, None),
        (10.0, 20.0, 0.0, 8.0),
        (10.0, 20.0, 50.0, None),
        (20.0, 10.0, 0.0, 6.0),
        (20.0, 10.0, 50.0, None),
        (20.0, 20.0, 0.0, 7.0),
        (20.0, 20.0, 50.0, None),
    ]
    return pd.DataFrame([
        {"LATITUDE": lat, "LONGITUDE": lon, "LEV_M": lev,
         "DATEANDTIME": "2000-01-01 00:00:00", "P_NITRATE": val}
        for lat, lon, lev, val in data
    ])


@pytest.mark.skipif(not _has_cartopy, reason="cartopy not installed ([geo] extra)")
class TestPlotSpatialDistribution:
    # Observation-density map from a plain lat/lon DataFrame
    def test_from_dataframe(self):
        df = pd.DataFrame({"LATITUDE": np.random.uniform(30, 60, 50),
                           "LONGITUDE": np.random.uniform(-40, 0, 50)})
        ax = plot_spatial_distribution(df)
        assert ax is not None
        plt.close()


@pytest.mark.skipif(not _has_cartopy, reason="cartopy not installed ([geo] extra)")
class TestPlotLatLonRange:
    # Draws the bounding-box outline (4 edges) on a world map
    def test_draws_bounding_box(self):
        ax = plot_lat_lon_range(30, 60, -40, 0)
        assert len(ax.lines) == 4
        plt.close()


@pytest.mark.skipif(not _has_cartopy, reason="cartopy not installed ([geo] extra)")
class TestPlotParameterMap:
    # One figure per depth/time slice is saved to savefig_folder
    def test_runs_without_error(self, tmp_path):
        plot_parameter_map(_missing_value_wide_df(), ["P_NITRATE"],
                           savefig_folder=str(tmp_path))
        assert list(tmp_path.iterdir())


@pytest.mark.skipif(not _has_cartopy, reason="cartopy not installed ([geo] extra)")
class TestPlotMissingValueInfoMapOverDepth:
    # Depth-integrated missingness map is saved to savefig_folder
    def test_runs_without_error(self, tmp_path):
        plot_missing_value_info_map_over_depth(_missing_value_wide_df(), ["P_NITRATE"],
                                               savefig_folder=str(tmp_path))
        assert list(tmp_path.iterdir())


@pytest.mark.skipif(not _has_cartopy, reason="cartopy not installed ([geo] extra)")
class TestPlotMissingValueInfoMapJoint:
    # Single combined map, collapsing params/depth/time into one figure
    def test_basic(self):
        ax = plot_missing_value_info_map_joint(_missing_value_wide_df(), ["P_NITRATE"])
        assert ax is not None
        plt.close()

    # A caller-supplied ax is drawn into and returned as-is
    def test_custom_ax(self):
        import cartopy.crs as ccrs
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        result = plot_missing_value_info_map_joint(_missing_value_wide_df(), ["P_NITRATE"], ax=ax)
        assert result is ax
        plt.close()
