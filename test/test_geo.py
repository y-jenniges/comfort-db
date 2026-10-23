"""Tests for comfort.geo, i.e. geospatial utility functions."""
import numpy as np
import pandas as pd
import pytest

from comfort.geo import (
    along_track_distance,
    classify_from_grid,
    distance_to_coast,
    load_biomes,
    load_jenniges_provinces,
    load_longhurst,
    load_basins,
    region_bbox,
    select_regions,
    water_mass_masks,
    water_mass_statistics,
)

_has_shapely = True
try:
    import shapely
except ImportError:
    _has_shapely = False

_has_geopandas = True
try:
    import geopandas
except ImportError:
    _has_geopandas = False

_has_xarray = True
try:
    import xarray
except ImportError:
    _has_xarray = False


@pytest.mark.skipif(not _has_shapely, reason="shapely not installed ([geo] extra)")
class TestDistanceToCoast:
    @staticmethod
    def _vertical_coastline():
        """Synthetic coastline: A vertical line at lon=0, lat from -10 to 10."""
        from shapely.geometry import LineString
        return LineString([(0.0, lat) for lat in range(-10, 11)])

    # Output is a pandas Series
    def test_returns_series(self):
        df = pd.DataFrame({"LATITUDE": [0.0], "LONGITUDE": [0.0]})
        result = distance_to_coast(df, coastline_geom=self._vertical_coastline())
        assert isinstance(result, pd.Series)

    # One distance value per input row
    def test_length_matches_input(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 5.0], "LONGITUDE": [1.0, 2.0]})
        result = distance_to_coast(df, coastline_geom=self._vertical_coastline())
        assert len(result) == 2

    # Output index matches the input DataFrame's index
    def test_preserves_index(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 0.0], "LONGITUDE": [1.0, 2.0]},
                          index=[10, 20])
        result = distance_to_coast(df, coastline_geom=self._vertical_coastline())
        assert list(result.index) == [10, 20]

    def test_point_on_coast_near_zero(self):
        # A point at lon=0 is on the coastline, i.e. distance should be ~0 km
        df = pd.DataFrame({"LATITUDE": [0.0], "LONGITUDE": [0.0]})
        result = distance_to_coast(df, coastline_geom=self._vertical_coastline())
        assert float(result.iloc[0]) < 1.0

    # Points farther from the coastline get a larger distance
    def test_distance_increases_with_separation(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 0.0], "LONGITUDE": [1.0, 5.0]})
        result = distance_to_coast(df, coastline_geom=self._vertical_coastline())
        assert result.iloc[0] < result.iloc[1]

    # Distance to the equator should be about right
    def test_approximate_distance_at_equator(self):
        # 1° of longitude at the equator ≈ 111 km
        df = pd.DataFrame({"LATITUDE": [0.0], "LONGITUDE": [1.0]})
        result = distance_to_coast(df, coastline_geom=self._vertical_coastline())
        assert 100.0 < float(result.iloc[0]) < 125.0

    # A list of coastline segments is accepted, not just a single geometry
    def test_list_of_geometries(self):
        from shapely.geometry import LineString
        geoms = [LineString([(0.0, lat), (0.0, lat + 1)]) for lat in range(-10, 10)]
        df = pd.DataFrame({"LATITUDE": [0.0], "LONGITUDE": [1.0]})
        result = distance_to_coast(df, coastline_geom=geoms)
        assert 100.0 < float(result.iloc[0]) < 125.0

    # Nearest vertex is also found at high latitude
    def test_nearest_vertex_at_high_latitude (self):
        from shapely.geometry import MultiPoint
        df = pd.DataFrame({"LATITUDE": [80.0], "LONGITUDE": [0.0]})  # observation at 80°N, 0°E
        # Coast vertex A is 10° east on the same parallel (~192 km away), nearer than vertex B, 2° north (~222 km away)
        coast = MultiPoint([(10.0, 80.0), (0.0, 82.0)])  # coast close to North Pole
        result = distance_to_coast(df, coastline_geom=coast)
        assert 185.0 < float(result.iloc[0]) < 200.0

    # Neighbours across the +-180 meridian have similar distance to coast
    def test_across_antimeridian(self):
        from shapely.geometry import LineString
        coast = LineString([(-179.0, lat) for lat in range(-10, 11)])
        df = pd.DataFrame({"LATITUDE": [0.0], "LONGITUDE": [179.0]})
        result = distance_to_coast(df, coastline_geom=coast)
        assert 215.0 < float(result.iloc[0]) < 230.0

    # Invalid coordinates give NaN, valid rows are unaffected
    def test_invalid_coordinates_give_nan(self, caplog):
        df = pd.DataFrame({"LATITUDE": [0.0, np.nan, 95.0, 0.0, 0.0],
                           "LONGITUDE": [1.0, 1.0, 1.0, np.nan, 200.0]})
        with caplog.at_level("WARNING"):
            result = distance_to_coast(df, coastline_geom=self._vertical_coastline())
        assert np.isfinite(result.iloc[0])
        assert result.iloc[1:].isna().all()
        assert "4 of 5" in caplog.text

    # Empty input df returns empty result (no error)
    def test_empty_dataframe(self):
        df = pd.DataFrame({"LATITUDE": [], "LONGITUDE": []})
        result = distance_to_coast(df, coastline_geom=self._vertical_coastline())
        assert len(result) == 0

    # Distances are never negative
    def test_non_negative(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 5.0, -5.0], "LONGITUDE": [3.0, 1.0, 7.0]})
        result = distance_to_coast(df, coastline_geom=self._vertical_coastline())
        assert (result >= 0).all()


class TestAlongTrackDistance:
    # The first station along the track is the distance origin
    def test_first_station_is_zero(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 0.0], "LONGITUDE": [0.0, 1.0]})
        result = along_track_distance(df)
        assert result.iloc[0] == 0.0

    # Cumulative distance grows monotonically along a straight track
    def test_monotonically_increases_along_straight_track(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 0.0, 0.0], "LONGITUDE": [0.0, 1.0, 2.0]})
        result = along_track_distance(df)
        assert result.iloc[0] < result.iloc[1] < result.iloc[2]

    # The distance the equator is about right
    def test_approximate_distance_at_equator(self):
        # 1 degree of longitude at the equator is about 111 km
        df = pd.DataFrame({"LATITUDE": [0.0, 0.0], "LONGITUDE": [0.0, 1.0]})
        result = along_track_distance(df)
        assert 100.0 < float(result.iloc[1]) < 125.0

    # Two depth levels at the same station (repeated lat/lon) share one distance
    def test_repeated_station_gets_same_distance(self):
        df = pd.DataFrame({
            "LATITUDE": [0.0, 0.0, 1.0],
            "LONGITUDE": [0.0, 0.0, 0.0],
            "LEV_M": [0, 10, 0],
        })
        result = along_track_distance(df)
        assert result.iloc[0] == result.iloc[1]
        assert result.iloc[2] > result.iloc[0]

    # Distances of out-of-order rows
    def test_order_col_reorders_stations(self):
        # Out-of-order rows: station at lon=2 was visited before lon=1
        df = pd.DataFrame({
            "LATITUDE": [0.0, 0.0, 0.0],
            "LONGITUDE": [0.0, 2.0, 1.0],
            "SEQ": [0, 2, 1],
        })
        result = along_track_distance(df, order_col="SEQ")

        # Ordered by SEQ: lon 0 -> 1 -> 2, so the lon=1 station is closer to start than lon=2
        dist_by_lon = dict(zip(df["LONGITUDE"], result))
        assert dist_by_lon[1.0] < dist_by_lon[2.0]

    # A track returning to an earlier position: station_col keeps the return leg
    def test_station_col_separates_revisited_position(self):
        df = pd.DataFrame({
            "ID": [1, 2, 3],
            "LATITUDE": [0.0, 0.0, 0.0],
            "LONGITUDE": [0.0, 1.0, 0.0],
        })
        by_coords = along_track_distance(df)
        by_id = along_track_distance(df, station_col="ID")
        assert by_coords.iloc[2] == by_coords.iloc[0] == 0.0
        assert 200.0 < float(by_id.iloc[2]) < 250.0
        assert by_id.iloc[1] < by_id.iloc[2]

    # A list of columns forms the station key, rows of one station (different depths) share a distance
    def test_station_col_list_and_depth_rows(self):
        df = pd.DataFrame({
            "CRUISE": ["x", "x", "x"],
            "ID": [1, 1, 2],
            "LATITUDE": [0.0, 0.0, 0.0],
            "LONGITUDE": [0.0, 0.0, 1.0],
            "LEV_M": [0, 10, 0],
        })
        result = along_track_distance(df, station_col=["CRUISE", "ID"])
        assert result.iloc[0] == result.iloc[1] == 0.0
        assert result.iloc[2] > 0.0

    # A station is placed by its earliest order value, not by its first row in df
    def test_order_uses_earliest_value_per_station(self):
        df = pd.DataFrame({
            "ID": [1, 1, 2],
            "LATITUDE": [0.0, 0.0, 0.0],
            "LONGITUDE": [2.0, 2.0, 1.0],
            "SEQ": [9, 1, 5],
        })
        result = along_track_distance(df, order_col="SEQ", station_col="ID")
        # Station 1 (SEQ min(1,9)=1 ) comes first, station 2 (SEQ 5) is 1 degree further
        assert result.iloc[0] == result.iloc[1] == 0.0
        assert 100.0 < float(result.iloc[2]) < 125.0

    # An unknown station column raises an error
    def test_missing_station_col_raises(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 1.0], "LONGITUDE": [0.0, 0.0]})
        with pytest.raises(KeyError):
            along_track_distance(df, station_col="ID")

    # Output index matches the input DataFrame's index
    def test_preserves_index(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 1.0], "LONGITUDE": [0.0, 0.0]}, index=[10, 20])
        result = along_track_distance(df)
        assert list(result.index) == [10, 20]

    # Distances are never negative
    def test_non_negative(self):
        df = pd.DataFrame({"LATITUDE": [0.0, -3.0, 5.0], "LONGITUDE": [3.0, 1.0, -7.0]})
        result = along_track_distance(df)
        assert (result >= 0).all()


def _obs_df():
    """Three observations: one north, one south, one outside both boxes."""
    return pd.DataFrame({
        "LATITUDE": [15.0, -10.0, 50.0],
        "LONGITUDE": [5.0, 5.0, 5.0],
        "VAL": [1.0, 2.0, 3.0],
    })


@pytest.mark.skipif(not _has_shapely, reason="shapely not installed ([geo] extra)")
class TestWaterMassMasks:
    @staticmethod
    def _box_regions():
        """Two non-overlapping rectangular regions as a dict of coordinate lists."""
        return {
            "north": [(0.0, 10.0), (10.0, 10.0), (10.0, 20.0), (0.0, 20.0)],
            "south": [(0.0, -20.0), (10.0, -20.0), (10.0, 0.0), (0.0, 0.0)],
        }

    # Output is a DataFrame
    def test_returns_dataframe(self):
        result = water_mass_masks(_obs_df(), self._box_regions())
        assert isinstance(result, pd.DataFrame)

    # A region column is added
    def test_adds_region_column(self):
        result = water_mass_masks(_obs_df(), self._box_regions())
        assert "region" in result.columns

    # Observations are assigned to the region containing them
    def test_correct_assignments(self):
        result = water_mass_masks(_obs_df(), self._box_regions())
        assert result["region"].iloc[0] == "north"
        assert result["region"].iloc[1] == "south"

    # An observation outside every region gets NaN, not a wrong label
    def test_outside_region_is_nan(self):
        result = water_mass_masks(_obs_df(), self._box_regions())
        assert pd.isna(result["region"].iloc[2])

    # output_col lets the caller rename the added column
    def test_custom_output_col(self):
        result = water_mass_masks(_obs_df(), self._box_regions(), output_col="zone")
        assert "zone" in result.columns
        assert "region" not in result.columns

    # The input DataFrame is not modified in place
    def test_does_not_mutate_input(self):
        df = _obs_df()
        water_mass_masks(df, self._box_regions())
        assert "region" not in df.columns

    # A dict of shapely Polygons (not just coordinate lists) is accepted directly
    def test_dict_with_shapely_polygons(self):
        from shapely.geometry import Polygon
        regions = {
            "box": Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        }
        df = pd.DataFrame({"LATITUDE": [5.0, 50.0], "LONGITUDE": [5.0, 50.0]})
        result = water_mass_masks(df, regions)
        assert result["region"].iloc[0] == "box"
        assert pd.isna(result["region"].iloc[1])

    # A CSV of WKT polygons is accepted
    def test_csv_input(self, tmp_path):
        """water_mass_masks accepts a CSV file with WKT geometry column."""
        csv_path = tmp_path / "regions.csv"
        csv_path.write_text(
            "name,geometry\n"
            "north,\"POLYGON ((0 10, 10 10, 10 20, 0 20, 0 10))\"\n"
            "south,\"POLYGON ((0 -20, 10 -20, 10 0, 0 0, 0 -20))\"\n"
        )
        result = water_mass_masks(_obs_df(), str(csv_path))
        assert result["region"].iloc[0] == "north"
        assert result["region"].iloc[1] == "south"

    # An empty region set leaves every observation unlabelled
    def test_empty_regions_all_nan(self):
        result = water_mass_masks(_obs_df(), {})
        assert result["region"].isna().all()

    # Row count is unchanged by labelling
    def test_length_preserved(self):
        result = water_mass_masks(_obs_df(), self._box_regions())
        assert len(result) == len(_obs_df())

    # Output index matches the input DataFrame's index
    def test_index_preserved(self):
        df = _obs_df()
        df.index = [10, 20, 30]
        result = water_mass_masks(df, self._box_regions())
        assert list(result.index) == [10, 20, 30]

    @staticmethod
    def _adjacent_boxes():
        """Two boxes sharing the edge lon=10."""
        return {
            "a": [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            "b": [(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)],
        }

    # Points on a region edge are assigned, not dropped
    def test_boundary_point_is_assigned(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 5.0], "LONGITUDE": [5.0, 0.0]})
        result = water_mass_masks(df, self._adjacent_boxes())
        assert list(result["region"]) == ["a", "a"]

    # A point on a shared edge goes to the first region, in either order
    def test_shared_boundary_first_region_wins(self):
        df = pd.DataFrame({"LATITUDE": [5.0], "LONGITUDE": [10.0]})
        boxes = self._adjacent_boxes()
        assert water_mass_masks(df, boxes)["region"].iloc[0] == "a"
        reversed_boxes = {k: boxes[k] for k in ("b", "a")}
        assert water_mass_masks(df, reversed_boxes)["region"].iloc[0] == "b"

    # Dict regions outside lon/lat range are rejected
    def test_dict_outside_lonlat_raises(self):
        regions = {"far": [(0.0, 0.0), (1e6, 0.0), (1e6, 1e6), (0.0, 1e6)]}
        with pytest.raises(ValueError, match="lon/lat"):
            water_mass_masks(_obs_df(), regions)

    @pytest.mark.skipif(not _has_geopandas, reason="geopandas not installed ([geo] extra)")
    # A projected GeoDataFrame is reprojected, not relabelled as lon/lat
    def test_projected_geodataframe_is_reprojected(self):
        import geopandas as gpd
        from shapely.geometry import Polygon
        gdf = gpd.GeoDataFrame(
            {"name": ["box"], "geometry": [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]},
            crs="EPSG:4326",
        ).to_crs("EPSG:3857")
        df = pd.DataFrame({"LATITUDE": [5.0, 50.0], "LONGITUDE": [5.0, 50.0]})
        result = water_mass_masks(df, gdf)
        assert result["region"].iloc[0] == "box"
        assert pd.isna(result["region"].iloc[1])

    @pytest.mark.skipif(not _has_geopandas, reason="geopandas not installed ([geo] extra)")
    # Without a CRS lon/lat is assumed, with a warning
    def test_geodataframe_without_crs_assumes_lonlat(self, caplog):
        import geopandas as gpd
        from shapely.geometry import Polygon
        gdf = gpd.GeoDataFrame({"name": ["box"], "geometry": [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]})
        df = pd.DataFrame({"LATITUDE": [5.0], "LONGITUDE": [5.0]})
        with caplog.at_level("WARNING"):
            result = water_mass_masks(df, gdf)
        assert result["region"].iloc[0] == "box"
        assert "no CRS" in caplog.text

    @pytest.mark.skipif(not _has_geopandas, reason="geopandas not installed ([geo] extra)")
    # Without a CRS, invalid lat/lon coordinates raise instead of silently matching nothing
    def test_geodataframe_without_crs_projected_raises(self):
        import geopandas as gpd
        from shapely.geometry import Polygon
        gdf = gpd.GeoDataFrame({"name": ["box"], "geometry": [Polygon([(0, 0), (1e6, 0), (1e6, 1e6), (0, 1e6)])]})
        with pytest.raises(ValueError, match="set_crs"):
            water_mass_masks(_obs_df(), gdf)


class TestWaterMassStatistics:
    @staticmethod
    def _labelled_df():
        df = _obs_df().copy()
        df["region"] = ["north", "south", np.nan]
        df["VAL2"] = [10.0, 20.0, 30.0]
        return df

    # Output is a DataFrame
    def test_returns_dataframe(self):
        result = water_mass_statistics(self._labelled_df(), parameters=["VAL"])
        assert isinstance(result, pd.DataFrame)

    # Rows are indexed by region name
    def test_rows_are_regions(self):
        result = water_mass_statistics(self._labelled_df(), parameters=["VAL"])
        assert set(result.index) == {"north", "south"}

    # Unlabelled (NaN-region) observations do not get their own row
    def test_nan_region_excluded(self):
        result = water_mass_statistics(self._labelled_df(), parameters=["VAL"])
        assert np.nan not in result.index

    # Standard descriptive statistics are all present
    def test_statistics_columns(self):
        result = water_mass_statistics(self._labelled_df(), parameters=["VAL"])
        assert set(result.columns.get_level_values(1)) >= {"count", "mean", "std", "min", "max"}

    # Multiple parameter columns each get their own statistics block
    def test_multiple_params(self):
        result = water_mass_statistics(self._labelled_df(), parameters=["VAL", "VAL2"])
        assert ("VAL", "mean") in result.columns
        assert ("VAL2", "mean") in result.columns

    # Each region's count matches its number of observations
    def test_count_correct(self):
        result = water_mass_statistics(self._labelled_df(), parameters=["VAL"])
        assert result.loc["north", ("VAL", "count")] == 1
        assert result.loc["south", ("VAL", "count")] == 1

    # Per-region mean is computed correctly
    def test_mean_correct(self):
        result = water_mass_statistics(self._labelled_df(), parameters=["VAL"])
        assert result.loc["north", ("VAL", "mean")] == pytest.approx(1.0)
        assert result.loc["south", ("VAL", "mean")] == pytest.approx(2.0)

    # Single parameter as string (not list) still works
    def test_string_param_col(self):
        result = water_mass_statistics(self._labelled_df(), parameters="VAL")
        assert ("VAL", "mean") in result.columns


@pytest.mark.skipif(not _has_shapely, reason="shapely not installed ([geo] extra)")
class TestRegionBbox:
    _regions = {
        "north": [(0, 40), (10, 40), (10, 50), (0, 50)],
        "south": [(0, -50), (10, -50), (10, -40), (0, -40)],
    }

    # Union bbox spans both named regions
    def test_union_bbox(self):
        result = region_bbox(self._regions)
        assert result == (-50.0, 50.0, 0.0, 10.0)

    # Bbox of a single region
    def test_single_region_bbox(self):
        result = region_bbox({"north": self._regions["north"]})
        assert result == (40.0, 50.0, 0.0, 10.0)

    # A projected GeoDataFrame yields a bbox in degrees
    @pytest.mark.skipif(not _has_geopandas, reason="geopandas not installed ([geo] extra)")
    def test_projected_geodataframe_gives_degrees(self):
        import geopandas as gpd
        from shapely.geometry import Polygon
        gdf = gpd.GeoDataFrame(
            {"name": ["box"], "geometry": [Polygon([(1, 2), (3, 2), (3, 4), (1, 4)])]},
            crs="EPSG:4326",
        ).to_crs("EPSG:3857")
        result = region_bbox(gdf)
        assert np.allclose(result, (2.0, 4.0, 1.0, 3.0))

    # Works with a shapely polygon wrapped in a dict
    def test_shapely_polygon_input(self):
        from shapely.geometry import Polygon
        result = region_bbox({"box": Polygon([(1, 2), (3, 2), (3, 4), (1, 4)])})
        assert result == (2.0, 4.0, 1.0, 3.0)

    # A region crossing the antimeridian (stored as two parts near +-180, as a real
    # Pacific basin shapefile would be) wraps correctly and returns lon_min > lon_max
    def test_antimeridian_crossing_region(self):
        from shapely.geometry import MultiPolygon, Polygon
        pacific_strip = MultiPolygon([
            Polygon([(170, -10), (180, -10), (180, 10), (170, 10)]),
            Polygon([(-180, -10), (-170, -10), (-170, 10), (-180, 10)]),
        ])
        result = region_bbox({"pacific_strip": pacific_strip})
        assert result == (-10.0, 10.0, 170.0, -170.0)


@pytest.mark.skipif(not _has_shapely, reason="shapely not installed ([geo] extra)")
class TestSelectRegions:
    _regions = {
        "north": [(0, 40), (10, 40), (10, 50), (0, 50)],
        "south": [(0, -50), (10, -50), (10, -40), (0, -40)],
    }

    # Selecting one name returns just that region
    def test_selects_single_name(self):
        result = select_regions(self._regions, "north")
        assert set(result) == {"north"}

    # Selecting multiple names returns all of them
    def test_selects_multiple_names(self):
        result = select_regions(self._regions, ["north", "south"])
        assert set(result) == {"north", "south"}

    # An unknown region name raises an error
    def test_raises_on_missing_name(self):
        with pytest.raises(ValueError, match="not found"):
            select_regions(self._regions, "pacific")

    # Explicit region_name_col selects a non-default column (e.g. Longhurst's "ProvDescr")
    @pytest.mark.skipif(not _has_geopandas, reason="geopandas not installed ([geo] extra)")
    def test_explicit_region_name_col_selects_non_default_column(self, tmp_path):
        import geopandas as gpd
        from shapely.geometry import Polygon

        # Dummy Longhurst-style provinces
        gdf = gpd.GeoDataFrame({
            "ProvCode": [1, 2],
            "ProvDescr": ["North Atlantic Drift", "Gulf Stream"],
            "geometry": [
                Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                Polygon([(10, 0), (20, 0), (20, 10), (10, 10)]),
            ],
        }, crs="EPSG:4326")
        path = tmp_path / "longhurst.shp"
        gdf.to_file(path)

        # Passing region_name_col should override its default value ("name")
        result = select_regions(path, "Gulf Stream", region_name_col="ProvDescr")
        assert set(result) == {"Gulf Stream"}

    # A missing default "name" column raises an error
    @pytest.mark.skipif(not _has_geopandas, reason="geopandas not installed ([geo] extra)")
    def test_raises_error_when_name_column_not_found(self):
        import geopandas as gpd
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame({
            "ProvDescr": ["box"],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        })
        with pytest.raises(KeyError, match="not found"):
            select_regions(gdf, "box")


@pytest.mark.skipif(not _has_geopandas, reason="geopandas not installed ([geo] extra)")
class TestLoadLonghurst:
    @staticmethod
    def _write_shapefile(tmp_path, code_col="ProvCode", name_col="ProvDescr"):
        import geopandas as gpd
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame({
            code_col: [1, 2],
            name_col: ["North Atlantic Drift", "Gulf Stream"],
            "geometry": [
                Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                Polygon([(10, 0), (20, 0), (20, 10), (10, 10)]),
            ],
        }, crs="EPSG:4326")
        path = tmp_path / "longhurst.shp"
        gdf.to_file(path)
        return path

    # Column names (ProvCode/ProvDescr) are auto-detected and renamed
    def test_auto_detects_columns(self, tmp_path):
        result = load_longhurst(self._write_shapefile(tmp_path))
        assert {"province_code", "province_name", "geometry"} <= set(result.columns)

    # Province names survive the read/rename process
    def test_values_preserved(self, tmp_path):
        result = load_longhurst(self._write_shapefile(tmp_path))
        assert set(result["province_name"]) == {"North Atlantic Drift", "Gulf Stream"}

    # code_col/name_col override auto-detection of names
    def test_explicit_column_names(self, tmp_path):
        path = self._write_shapefile(tmp_path, code_col="code", name_col="label")
        result = load_longhurst(path, code_col="code", region_name_col="label")
        assert "province_code" in result.columns
        assert "province_name" in result.columns

    # Neither auto-detection nor an explicit column name found is a hard error
    def test_raises_when_columns_not_found(self, tmp_path):
        path = self._write_shapefile(tmp_path, code_col="foo", name_col="bar")
        with pytest.raises(ValueError, match="Could not detect"):
            load_longhurst(path)


@pytest.mark.skipif(not _has_geopandas, reason="geopandas not installed ([geo] extra)")
class TestLoadBasins:
    @staticmethod
    def _write_shapefile(tmp_path, name_col="name"):
        # Dummy Global Oceans and Seas (GOaS) data
        import geopandas as gpd
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame({
            name_col: ["North Atlantic Ocean", "South Atlantic Ocean"],
            "geometry": [
                Polygon([(-80, 0), (0, 0), (0, 65), (-80, 65)]),
                Polygon([(-70, -60), (20, -60), (20, 0), (-70, 0)]),
            ],
        }, crs="EPSG:4326")
        path = tmp_path / "goas.shp"
        gdf.to_file(path)
        return path

    # Column name (name) is auto-detected case-insensitively and renamed
    def test_auto_detects_columns(self, tmp_path):
        result = load_basins(self._write_shapefile(tmp_path))
        assert list(result.columns) == ["basin_name", "geometry"]

    # Basin names survive the read/rename process
    def test_values_preserved(self, tmp_path):
        result = load_basins(self._write_shapefile(tmp_path))
        assert set(result["basin_name"]) == {"North Atlantic Ocean", "South Atlantic Ocean"}

    # name_col overrides auto-detection
    def test_explicit_column_name(self, tmp_path):
        path = self._write_shapefile(tmp_path, name_col="label")
        result = load_basins(path, region_name_col="label")
        assert "basin_name" in result.columns

    # No name column, auto-detected or explicit, raises an error
    def test_raises_when_column_not_found(self, tmp_path):
        path = self._write_shapefile(tmp_path, name_col="bar")
        with pytest.raises(ValueError, match="Could not detect"):
            load_basins(path)

    # The result composes with water_mass_masks
    def test_composes_with_water_mass_masks(self, tmp_path):
        gdf = load_basins(self._write_shapefile(tmp_path))
        obs = pd.DataFrame({"LATITUDE": [30.0, -30.0], "LONGITUDE": [-40.0, -20.0]})
        result = water_mass_masks(obs, gdf, region_name_col="basin_name")
        assert list(result["region"]) == ["North Atlantic Ocean", "South Atlantic Ocean"]


class TestLoadJennigesProvinces:
    @staticmethod
    def _jenniges_csv(tmp_path):
        """Write a minimal Jenniges-style CSV and return its path."""
        path = tmp_path / "cluster_set.csv"
        path.write_text(
            "LEV_M,LATITUDE,LONGITUDE,label,color\n"
            "0,50.0,-30.0,1,#ff0000\n"
            "0,55.0,-25.0,2,#00ff00\n"
            "0,60.0,-20.0,1,#ff0000\n"
            "100,50.0,-30.0,3,#0000ff\n"
            "100,55.0,-25.0,3,#0000ff\n"
        )
        return path

    # Output is a DataFrame
    def test_returns_dataframe(self, tmp_path):
        result = load_jenniges_provinces(self._jenniges_csv(tmp_path))
        assert isinstance(result, pd.DataFrame)

    # depth=None returns every row across all depth levels
    def test_all_depths_when_none(self, tmp_path):
        result = load_jenniges_provinces(self._jenniges_csv(tmp_path), depth=None)
        assert len(result) == 5

    # An exact depth match filters to just that level
    def test_filter_to_depth(self, tmp_path):
        result = load_jenniges_provinces(self._jenniges_csv(tmp_path), depth=0)
        assert len(result) == 3
        assert (result["LEV_M"] == 0).all()

    # A depth with no exact match falls back to the closest available level
    def test_closest_depth(self, tmp_path):
        result = load_jenniges_provinces(self._jenniges_csv(tmp_path), depth=10)
        assert len(result) == 3
        assert (result["LEV_M"] == 0).all()

    # A missing file is a FileNotFoundError, not a silent empty result
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_jenniges_provinces(tmp_path / "nonexistent.csv")

    # A required column absent from the CSV raises with its name in the message
    def test_missing_column(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("LATITUDE,LONGITUDE\n50,30\n")
        with pytest.raises(ValueError, match="label"):
            load_jenniges_provinces(path)


@pytest.mark.skipif(not _has_xarray, reason="xarray not installed ([grid] extra)")
class TestLoadBiomes:
    @staticmethod
    def _write_biomes_nc(tmp_path, with_time=False):
        # Generate dummy Fay and McKinley biome data
        import xarray as xr

        lat = np.array([-10.0, 0.0, 10.0])
        lon = np.array([-30.0, -20.0])
        if with_time:
            data = np.array([[[1, 2], [np.nan, 3], [2, 2]],
                              [[1, 1], [3, 3], [2, np.nan]]])
            da = xr.DataArray(
                data, dims=("time", "lat", "lon"),
                coords={"time": [0, 1], "lat": lat, "lon": lon},
            )
        else:
            data = np.array([[1, 2], [np.nan, 3], [2, 2]])
            da = xr.DataArray(data, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
        ds = xr.Dataset({"biomes": da})
        path = tmp_path / "biomes.nc"
        ds.to_netcdf(path)
        return path

    # Output is flattened to LATITUDE, LONGITUDE, label
    def test_returns_flattened_dataframe(self, tmp_path):
        result = load_biomes(self._write_biomes_nc(tmp_path), biome_var="biomes")
        assert list(result.columns) == ["LATITUDE", "LONGITUDE", "label"]

    # NaN cells (e.g. land) are dropped
    def test_drops_nan_cells(self, tmp_path):
        result = load_biomes(self._write_biomes_nc(tmp_path), biome_var="biomes")
        assert len(result) == 5
        assert result["label"].notna().all()

    # Result composes with classify_from_grid for point classification
    def test_composes_with_classify_from_grid(self, tmp_path):
        grid = load_biomes(self._write_biomes_nc(tmp_path), biome_var="biomes")
        obs = pd.DataFrame({"LATITUDE": [0.1], "LONGITUDE": [-19.9]})
        result = classify_from_grid(obs, grid)
        assert result["label"].iloc[0] == 3

    # A time-varying mask is returned in long form - one row per grid
    # cell per step, with an extra dimension column
    def test_time_dimension_returns_long_form(self, tmp_path):
        path = self._write_biomes_nc(tmp_path, with_time=True)
        result = load_biomes(path, biome_var="biomes")
        assert list(result.columns) == ["LATITUDE", "LONGITUDE", "TIME", "label"]
        assert len(result) == 10
        assert set(result["TIME"]) == {0, 1}

    # The long-form output composes with classify_from_grid's time_col to
    # match each observation to its own step
    def test_time_aware_composes_with_classify_from_grid(self, tmp_path):
        path = self._write_biomes_nc(tmp_path, with_time=True)
        grid = load_biomes(path, biome_var="biomes")
        obs = pd.DataFrame({"LATITUDE": [0.1], "LONGITUDE": [-19.9], "TIME": [1]})
        result = classify_from_grid(obs, grid, time_col="TIME")
        assert result["label"].iloc[0] == 3

    # An unknown biome variable raises an error
    def test_raises_when_variable_not_found(self, tmp_path):
        import xarray as xr
        ds = xr.Dataset({"foo": xr.DataArray([[1]], dims=("lat", "lon"), coords={"lat": [0.0], "lon": [0.0]})})
        path = tmp_path / "bad.nc"
        ds.to_netcdf(path)
        with pytest.raises(ValueError, match="not found"):
            load_biomes(path, biome_var="missing")


class TestClassifyFromGrid:
    @staticmethod
    def _grid():
        return pd.DataFrame({
            "LATITUDE":  [50.0, 55.0, 60.0],
            "LONGITUDE": [-30.0, -25.0, -20.0],
            "label":     [1, 2, 3],
        })

    # Output is a DataFrame
    def test_returns_dataframe(self):
        df = pd.DataFrame({"LATITUDE": [50.0], "LONGITUDE": [-30.0]})
        result = classify_from_grid(df, self._grid())
        assert isinstance(result, pd.DataFrame)

    # A point exactly on a grid node gets that node's label
    def test_exact_match(self):
        df = pd.DataFrame({"LATITUDE": [55.0], "LONGITUDE": [-25.0]})
        result = classify_from_grid(df, self._grid())
        assert result["label"].iloc[0] == 2

    # A point between grid nodes gets the nearest one's label
    def test_nearest_neighbour(self):
        df = pd.DataFrame({"LATITUDE": [51.0], "LONGITUDE": [-29.0]})
        result = classify_from_grid(df, self._grid())
        assert result["label"].iloc[0] == 1

    # output_col lets the caller rename the added label column
    def test_custom_output_col(self):
        df = pd.DataFrame({"LATITUDE": [50.0], "LONGITUDE": [-30.0]})
        result = classify_from_grid(df, self._grid(), output_col="cluster")
        assert "cluster" in result.columns
        assert "label" not in result.columns

    # The input DataFrame is not modified in place
    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"LATITUDE": [50.0], "LONGITUDE": [-30.0]})
        classify_from_grid(df, self._grid())
        assert "label" not in df.columns

    # Row count is unchanged by classification
    def test_length_preserved(self):
        df = pd.DataFrame({"LATITUDE": [50.0, 55.0, 60.0], "LONGITUDE": [-30.0, -25.0, -20.0]})
        result = classify_from_grid(df, self._grid())
        assert len(result) == 3

    # Output index matches the input DataFrame's index
    def test_index_preserved(self):
        df = pd.DataFrame({"LATITUDE": [50.0, 55.0], "LONGITUDE": [-30.0, -25.0]},
                          index=[10, 20])
        result = classify_from_grid(df, self._grid())
        assert list(result.index) == [10, 20]

    @staticmethod
    def _time_grid():
        # Temporal grid
        return pd.DataFrame({
            "LATITUDE":  [50.0, 55.0, 60.0, 50.0, 55.0, 60.0],
            "LONGITUDE": [-30.0, -25.0, -20.0, -30.0, -25.0, -20.0],
            "YEAR":      [2000, 2000, 2000, 2010, 2010, 2010],
            "label":     [1, 2, 3, 10, 20, 30],
        })

    # time_col restricts the spatial lookup to the observation's own year
    def test_time_aware_matches_own_year(self):
        df = pd.DataFrame({"LATITUDE": [50.0], "LONGITUDE": [-30.0], "YEAR": [2000]})
        result = classify_from_grid(df, self._time_grid(), time_col="YEAR")
        assert result["label"].iloc[0] == 1

    # A different year matches a different grid slice, same location
    def test_time_aware_matches_different_year(self):
        df = pd.DataFrame({"LATITUDE": [50.0], "LONGITUDE": [-30.0], "YEAR": [2010]})
        result = classify_from_grid(df, self._time_grid(), time_col="YEAR")
        assert result["label"].iloc[0] == 10

    # An observation outside the grid's time range snaps to the nearest available year
    def test_time_aware_snaps_to_nearest_year(self):
        df = pd.DataFrame({"LATITUDE": [50.0], "LONGITUDE": [-30.0], "YEAR": [2023]})
        result = classify_from_grid(df, self._time_grid(), time_col="YEAR")
        assert result["label"].iloc[0] == 10

    # grid_time_col lets the grid use a different time column name than the observations
    def test_time_aware_custom_grid_time_col(self):
        grid = self._time_grid().rename(columns={"YEAR": "grid_year"})
        df = pd.DataFrame({"LATITUDE": [50.0], "LONGITUDE": [-30.0], "YEAR": [2000]})
        result = classify_from_grid(df, grid, time_col="YEAR", grid_time_col="grid_year")
        assert result["label"].iloc[0] == 1

    # Observations from different years in one call each get their own year's label
    def test_time_aware_mixed_years_in_one_call(self):
        df = pd.DataFrame({"LATITUDE": [50.0, 55.0], "LONGITUDE": [-30.0, -25.0], "YEAR": [2000, 2010]})
        result = classify_from_grid(df, self._time_grid(), time_col="YEAR")
        assert list(result["label"]) == [1, 20]


class TestColumnCasing:
    # Column arguments in another casing give the same result
    def test_along_track_distance_lower_case_arguments(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 0.0, 0.0], "LONGITUDE": [0.0, 1.0, 2.0]})
        expected = along_track_distance(df)
        result = along_track_distance(df, lat_col="latitude", lon_col="longitude")
        pd.testing.assert_series_equal(result, expected)

    # A missing region name column is a KeyError that lists the available columns
    def test_unknown_column_raises_key_error(self):
        df = pd.DataFrame({"LATITUDE": [0.0], "LONGITUDE": [0.0]})
        with pytest.raises(KeyError, match="Available columns"):
            along_track_distance(df, lat_col="lat")
