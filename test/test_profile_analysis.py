"""Tests for comfort.profile_analysis, i.e. profile-level analysis functions."""
import numpy as np
import pandas as pd
import pytest

from comfort.profile_analysis import (
    average_duplicate_records_per_profile,
    detect_depth_col,
    interpolate_depth_levels,
    mixed_layer_depth,
    profile_completeness,
    pycnocline_depth,
    vertical_gradient,
)


class TestAverageDuplicateRecordsPerProfile:
    # An empty input dict returns an empty dict
    def test_empty_raw_returns_empty_dict(self):
        assert average_duplicate_records_per_profile({}) == {}

    # Two readings at the same (ID, PROFILE_NUMBER, depth) are averaged into one
    def test_averages_duplicate_reading_within_profile(self):
        raw = {"OXYGEN": pd.DataFrame({
            "ID": [1, 1], "PROFILE_NUMBER": [1, 1], "LEV_M": [10.0, 10.0],
            "OXYGEN": [200.0, 220.0],
        })}
        result = average_duplicate_records_per_profile(raw)
        assert len(result["OXYGEN"]) == 1
        assert result["OXYGEN"]["OXYGEN"].iloc[0] == pytest.approx(210.0)

    # Readings at different depths within the same profile stay separate rows
    def test_distinct_depths_not_merged(self):
        raw = {"OXYGEN": pd.DataFrame({
            "ID": [1, 1], "PROFILE_NUMBER": [1, 1], "LEV_M": [10.0, 20.0],
            "OXYGEN": [200.0, 220.0],
        })}
        result = average_duplicate_records_per_profile(raw)
        assert len(result["OXYGEN"]) == 2

    # Readings with different IDs and same profile number stay separate
    def test_same_profile_number_different_station_not_merged(self):
        # PROFILE_NUMBER alone is not globally unique
        raw = {"OXYGEN": pd.DataFrame({
            "ID": [1, 2], "PROFILE_NUMBER": [1, 1], "LEV_M": [10.0, 10.0],
            "OXYGEN": [200.0, 220.0],
        })}
        result = average_duplicate_records_per_profile(raw)
        assert len(result["OXYGEN"]) == 2

    # Constant station metadata (e.g. LATITUDE) is kept via "first", not averaged away
    def test_keeps_station_metadata_via_first(self):
        raw = {"OXYGEN": pd.DataFrame({
            "ID": [1, 1], "PROFILE_NUMBER": [1, 1], "LEV_M": [10.0, 10.0],
            "LATITUDE": [45.0, 45.0], "LONGITUDE": [-30.0, -30.0],
            "OXYGEN": [200.0, 220.0],
        })}
        result = average_duplicate_records_per_profile(raw)
        assert result["OXYGEN"]["LATITUDE"].iloc[0] == pytest.approx(45.0)

    # profile_col can be overridden when ID is not part of the grouping key
    def test_explicit_profile_col(self):
        raw = {"OXYGEN": pd.DataFrame({
            "PROFILE_NUMBER": [1, 1], "LEV_M": [10.0, 10.0],
            "OXYGEN": [200.0, 220.0],
        })}
        result = average_duplicate_records_per_profile(raw, profile_col="PROFILE_NUMBER")
        assert len(result["OXYGEN"]) == 1
        assert result["OXYGEN"]["OXYGEN"].iloc[0] == pytest.approx(210.0)

    # Each parameter in the dict is averaged on its own, independently of the others
    def test_multiple_parameters_averaged_independently(self):
        raw = {
            "OXYGEN": pd.DataFrame({
                "ID": [1, 1], "PROFILE_NUMBER": [1, 1], "LEV_M": [10.0, 10.0],
                "OXYGEN": [200.0, 220.0],
            }),
            "NITRATE": pd.DataFrame({
                "ID": [1, 1], "PROFILE_NUMBER": [1, 1], "LEV_M": [10.0, 10.0],
                "NITRATE": [5.0, 7.0],
            }),
        }
        result = average_duplicate_records_per_profile(raw)
        assert result["OXYGEN"]["OXYGEN"].iloc[0] == pytest.approx(210.0)
        assert result["NITRATE"]["NITRATE"].iloc[0] == pytest.approx(6.0)


class TestVerticalGradient:
    # A gradient column is added to the output
    def test_adds_gradient_column(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1], "LEV_DBAR": [0, 100, 200],
                           "VAL": [10.0, 8.0, 6.0]})
        out = vertical_gradient(df)
        assert "VAL_gradient" in out.columns

    # A linear profile has a constant gradient throughout
    def test_uniform_gradient(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1], "LEV_DBAR": [0, 100, 200],
                           "VAL": [10.0, 8.0, 6.0]})
        out = vertical_gradient(df)
        # -2 per 100 dbar = -0.02 per dbar everywhere
        assert np.allclose(out.sort_values("LEV_DBAR")["VAL_gradient"], -0.02)

    # A NaN value propagates into the gradient at that level
    def test_nan_values_propagate(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1], "LEV_DBAR": [0, 100, 200],
                           "VAL": [10.0, np.nan, 6.0]})
        out = vertical_gradient(df)
        assert out[out["LEV_DBAR"] == 100]["VAL_gradient"].isna().all()

    # A profile with a single depth level has no gradient to compute
    def test_single_level_profile_skipped(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1], "LEV_DBAR": [0], "VAL": [10.0]})
        out = vertical_gradient(df)
        assert out["VAL_gradient"].isna().all()

    # Each profile's gradient is computed independently
    def test_multiple_profiles(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 2, 2],
                           "LEV_DBAR": [0, 100, 0, 100],
                           "VAL": [10.0, 8.0, 5.0, 3.0]})
        out = vertical_gradient(df)
        assert out["VAL_gradient"].notna().all()


class TestMixedLayerDepth:
    @staticmethod
    def _profile_with_mld_at_100():
        return pd.DataFrame({
            "PROFILE_NUMBER": [1, 1, 1, 1],
            "LEV_DBAR": [0.0, 50.0, 100.0, 200.0],
            "sigma0": [25.0, 25.0, 25.04, 25.2],
        })

    # Output is a DataFrame
    def test_returns_dataframe(self):
        result = mixed_layer_depth(self._profile_with_mld_at_100(), density_col="sigma0")
        assert isinstance(result, pd.DataFrame)

    # Output carries profile ID and MLD depth
    def test_expected_columns(self):
        result = mixed_layer_depth(self._profile_with_mld_at_100(), density_col="sigma0")
        assert {"PROFILE_NUMBER", "MLD_m"}.issubset(result.columns)

    def test_mld_detected_at_correct_depth(self):
        # ref at 0m (nearest to 10m reference): sigma0=25.0
        # 50m: delta=0.00 < 0.03 → skip
        # 100m: delta=0.04 > 0.03 → crossing interpolated between 50m and 100m
        # target = 25.0 + 0.03 = 25.03 → 50 + (25.03-25.0)*(100-50)/(25.04-25.0) = 87.5
        result = mixed_layer_depth(self._profile_with_mld_at_100(),
                                   density_col="sigma0", delta_density=0.03, max_ref_distance=None)
        assert result["MLD_m"].iloc[0] == pytest.approx(87.5)

    # When the threshold is never crossed, there is no valid MLD
    def test_no_exceedance_returns_nan(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1],
                           "LEV_DBAR": [0.0, 100.0, 200.0],
                           "sigma0": [25.0, 25.0, 25.0]})
        result = mixed_layer_depth(df, density_col="sigma0", delta_density=0.03)
        assert pd.isna(result["MLD_m"].iloc[0])

    # Temperature threshold also computes MLD correctly
    def test_temperature_criterion(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1, 1],
                           "LEV_DBAR": [0.0, 50.0, 100.0, 200.0],
                           "CT": [15.0, 15.0, 14.75, 12.0]})
        result = mixed_layer_depth(df, temp_col="CT", delta_temp=0.2, max_ref_distance=None)
        # ref=15.0, 100m: |14.75-15.0|=0.25>0.2 → crossing interpolated between 50m and 100m
        # target = 15.0 - 0.2 = 14.8 → 50 + (14.8-15.0)*(100-50)/(14.75-15.0) = 90.0
        assert result["MLD_m"].iloc[0] == pytest.approx(90.0)

    # Density decreasing with depth does not end the mixed layer
    def test_density_decrease_is_not_exceedance(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1],
                           "LEV_DBAR": [10.0, 100.0, 200.0],
                           "sigma0": [25.0, 24.9, 24.8]})
        result = mixed_layer_depth(df, density_col="sigma0", delta_density=0.03)
        assert pd.isna(result["MLD_m"].iloc[0])

    # Temperature increase counts
    def test_temperature_increase_counts(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1],
                           "LEV_DBAR": [10.0, 100.0, 200.0],
                           "CT": [10.0, 10.0, 10.5]})
        result = mixed_layer_depth(df, temp_col="CT", delta_temp=0.2)
        assert result["MLD_m"].iloc[0] == pytest.approx(140.0)

    # No level near the reference depth means no valid MLD
    def test_reference_too_far_returns_nan(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1],
                           "LEV_DBAR": [80.0, 100.0, 200.0],
                           "sigma0": [25.0, 25.0, 25.2]})
        assert pd.isna(mixed_layer_depth(df, density_col="sigma0")["MLD_m"].iloc[0])
        result = mixed_layer_depth(df, density_col="sigma0", max_ref_distance=None)
        assert not pd.isna(result["MLD_m"].iloc[0])

    # When both are given, density_col takes priority over temp_col
    def test_density_preferred_over_temp(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1],
                           "LEV_DBAR": [0.0, 100.0, 200.0],
                           "sigma0": [25.0, 25.0, 25.0],
                           "CT": [15.0, 15.0, 14.0]})
        result = mixed_layer_depth(df, density_col="sigma0", temp_col="CT", delta_density=0.03)
        # density never exceeds → uses density, not temperature
        assert pd.isna(result["MLD_m"].iloc[0])

    # At least one of density_col/temp_col must be given
    def test_no_column_raises(self):
        with pytest.raises(ValueError, match="Provide at least one"):
            mixed_layer_depth(self._profile_with_mld_at_100())

    # Each profile gets its own MLD row
    def test_multiple_profiles(self):
        df = pd.concat([self._profile_with_mld_at_100(),
                        self._profile_with_mld_at_100().assign(PROFILE_NUMBER=2)])
        result = mixed_layer_depth(df, density_col="sigma0")
        assert len(result) == 2

    # Different IDs ensure that same profile IDs are kept separate
    def test_composite_key_separates_stations(self):
        # PROFILE_NUMBER=1 appears in two different stations (ID=10 and ID=20)
        # Without composite key these would be merged into one group
        base = self._profile_with_mld_at_100()
        df = pd.concat([
            base.assign(ID=10),
            base.assign(ID=20),
        ])
        result = mixed_layer_depth(df, density_col="sigma0")
        assert len(result) == 2
        assert set(result["ID"]) == {10, 20}

    # A profile with a single depth level is skipped, not returned as a degenerate row
    def test_short_profile_skipped(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1], "LEV_DBAR": [0.0], "sigma0": [25.0]})
        result = mixed_layer_depth(df, density_col="sigma0")
        assert result.empty


class TestPycnoclineDepth:
    @staticmethod
    def _profile_with_pycno_at_100():
        # Gradient is largest at 100 dbar:
        # i=1 (50m) central: (26.0-25.0)/(100-0) = 0.01
        # i=2 (100m) central: (26.1-25.0)/(150-50) = 0.011  ← max
        # i=3 (150m) central: (26.2-26.0)/(200-100) = 0.002
        return pd.DataFrame({
            "PROFILE_NUMBER": [1, 1, 1, 1, 1],
            "LEV_DBAR": [0.0, 50.0, 100.0, 150.0, 200.0],
            "sigma0": [25.0, 25.0, 26.0, 26.1, 26.2],
        })

    # Output is a DataFrame
    def test_returns_dataframe(self):
        result = pycnocline_depth(self._profile_with_pycno_at_100(), density_col="sigma0")
        assert isinstance(result, pd.DataFrame)

    # Output carries profile ID, pycnocline depth and max gradient
    def test_expected_columns(self):
        result = pycnocline_depth(self._profile_with_pycno_at_100(), density_col="sigma0")
        assert {"PROFILE_NUMBER", "pycnocline_depth_m", "max_gradient"}.issubset(result.columns)

    # min_gradient=0.0 (default) means every pycnocline counts as significant
    def test_significant_by_default_gives_valid_depth(self):
        result = pycnocline_depth(self._profile_with_pycno_at_100(), density_col="sigma0")
        assert not pd.isna(result["pycnocline_depth_m"].iloc[0])

    # A high min_gradient threshold marks a weak pycnocline as not significant -
    # depth is NaN, but max_gradient is still reported
    def test_below_min_gradient_gives_nan_depth(self):
        result = pycnocline_depth(self._profile_with_pycno_at_100(), density_col="sigma0",
                                  min_gradient=1.0)
        assert pd.isna(result["pycnocline_depth_m"].iloc[0])
        assert not pd.isna(result["max_gradient"].iloc[0])

    # The pycnocline is placed at the depth of maximum |d(density)/dz|
    def test_pycnocline_at_correct_depth(self):
        result = pycnocline_depth(self._profile_with_pycno_at_100(), density_col="sigma0")
        assert result["pycnocline_depth_m"].iloc[0] == 100.0

    # max_gradient is reported as a magnitude, not signed
    def test_max_gradient_positive(self):
        result = pycnocline_depth(self._profile_with_pycno_at_100(), density_col="sigma0")
        assert result["max_gradient"].iloc[0] > 0

    # A profile with a single depth level has no gradient to compute
    def test_short_profile_skipped(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1], "LEV_DBAR": [0.0], "sigma0": [25.0]})
        assert pycnocline_depth(df, density_col="sigma0").empty

    # Each profile gets its own pycnocline row
    def test_multiple_profiles(self):
        df = pd.concat([self._profile_with_pycno_at_100(),
                        self._profile_with_pycno_at_100().assign(PROFILE_NUMBER=2)])
        result = pycnocline_depth(df, density_col="sigma0")
        assert len(result) == 2
        assert (result["pycnocline_depth_m"] == 100.0).all()


class TestProfileCompleteness:
    # Level count and depth range are computed per profile
    def test_basic(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1], "LEV_DBAR": [0, 100, 200],
                           "VAL": [1.0, 2.0, 3.0]})
        result = profile_completeness(df)
        assert result["n_levels"].iloc[0] == 3
        assert result["depth_range"].iloc[0] == 200.0

    # coverage_fraction reports the share of reference depths matched within tolerance
    def test_coverage_fraction_with_reference(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1], "LEV_DBAR": [0, 100, 200],
                           "VAL": [1.0, 2.0, 3.0]})
        result = profile_completeness(df, reference_depths=[0, 50, 100, 150, 200])
        # Covers 0, 100, 200 within ±5 dbar → 3/5
        assert abs(result["coverage_fraction"].iloc[0] - 3/5) < 1e-9


class TestInterpolateDepthLevels:
    # Values are interpolated onto the target depths with the default (pchip)
    # method; on perfectly linear data pchip reduces to the same result as linear
    def test_basic(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1],
                           "LEV_DBAR": [0.0, 100.0, 200.0],
                           "VAL": [10.0, 8.0, 6.0]})
        result = interpolate_depth_levels(df, target_depths=[0, 50, 100, 150, 200])
        result_linear = interpolate_depth_levels(df, target_depths=[0, 50, 100, 150, 200], method="linear")
        p1 = result.sort_values("LEV_DBAR")
        p2 = result_linear.sort_values("LEV_DBAR")
        v1 = p1[p1["LEV_DBAR"] == 50]["VAL"].iloc[0]
        v2 = p2[p2["LEV_DBAR"] == 50]["VAL"].iloc[0]
        assert v1 == pytest.approx(v2, abs=0.01)

    # method="linear" reproduces plain piecewise-linear interpolation
    def test_linear_method(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1],
                           "LEV_DBAR": [0.0, 100.0, 200.0],
                           "VAL": [10.0, 8.0, 6.0]})
        result = interpolate_depth_levels(df, target_depths=[50], method="linear")
        assert result["VAL"].iloc[0] == pytest.approx(9.0)

    # pchip (default) and linear disagree on curved (non-collinear) data, since
    # pchip fits a shape-preserving cubic through the surrounding points
    def test_pchip_differs_from_linear_on_curved_data(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1],
                           "LEV_DBAR": [0.0, 50.0, 100.0],
                           "VAL": [0.0, 1.0, 8.0]})
        pchip_result = interpolate_depth_levels(df, target_depths=[25], method="pchip")
        linear_result = interpolate_depth_levels(df, target_depths=[25], method="linear")
        assert linear_result["VAL"].iloc[0] == pytest.approx(0.5)
        assert pchip_result["VAL"].iloc[0] != pytest.approx(0.5)

    # Unknown interpolation methods raises
    def test_invalid_method_raises(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1],
                           "LEV_DBAR": [0.0, 100.0], "VAL": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Unknown method"):
            interpolate_depth_levels(df, target_depths=[50], method="bogus")

    # Target depths outside the profile's measured range are dropped, not extrapolated
    def test_no_extrapolation(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1],
                           "LEV_DBAR": [50.0, 150.0], "VAL": [8.0, 6.0]})
        result = interpolate_depth_levels(df, target_depths=[0, 50, 100, 150, 200])
        # 0 and 200 are outside profile range
        assert not (result["LEV_DBAR"] == 0).any()
        assert not (result["LEV_DBAR"] == 200).any()

    # A target depth bracketed by two measurements further apart than max_gap is dropped
    def test_max_gap_drops_wide_bracket(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1],
                           "LEV_DBAR": [0.0, 500.0], "VAL": [10.0, 0.0]})
        result = interpolate_depth_levels(df, target_depths=[250], max_gap=100)
        assert result.empty

    # The same target depth is kept when the bracket is within max_gap
    def test_max_gap_keeps_narrow_bracket(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1],
                           "LEV_DBAR": [200.0, 300.0], "VAL": [10.0, 8.0]})
        result = interpolate_depth_levels(df, target_depths=[250], max_gap=100)
        assert len(result) == 1

    # A target depth that exactly matches a measured depth is always kept
    def test_max_gap_keeps_exact_match(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1],
                           "LEV_DBAR": [0.0, 500.0], "VAL": [10.0, 0.0]})
        result = interpolate_depth_levels(df, target_depths=[0, 500], max_gap=1)
        assert len(result) == 2

    # max_gap=None (default) applies no filtering
    def test_max_gap_none_by_default(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1],
                           "LEV_DBAR": [0.0, 500.0], "VAL": [10.0, 0.0]})
        result = interpolate_depth_levels(df, target_depths=[250])
        assert len(result) == 1


class TestDetectParamCol:
    # A single non-standard value column (e.g. NITRATE) is auto-detected
    def test_auto_detects_renamed_column(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1],
                           "LEV_DBAR": [0.0, 100.0, 200.0],
                           "NITRATE": [10.0, 8.0, 6.0]})
        result = vertical_gradient(df)
        assert "NITRATE_gradient" in result.columns

    # The default "VAL" column still works when present
    def test_falls_through_with_val(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1],
                           "LEV_DBAR": [0.0, 100.0, 200.0],
                           "VAL": [10.0, 8.0, 6.0]})
        result = vertical_gradient(df)
        assert "VAL_gradient" in result.columns

    # Auto-detection also applies to interpolate_depth_levels
    def test_interpolate_auto_detects(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1],
                           "LEV_DBAR": [0.0, 100.0, 200.0],
                           "OXYGEN": [200.0, 180.0, 160.0]})
        result = interpolate_depth_levels(df, target_depths=[0, 100, 200])
        assert "OXYGEN" in result.columns


class TestDetectDepthCol:
    # LEV_M is used when present
    def test_lev_m_found(self):
        df = pd.DataFrame({"LEV_M": [0.0], "VAL": [1.0]})
        assert detect_depth_col(df) == "LEV_M"

    # LEV_DBAR is the fallback when LEV_M is absent
    def test_fallback_to_lev_dbar(self):
        df = pd.DataFrame({"LEV_DBAR": [0.0], "VAL": [1.0]})
        assert detect_depth_col(df) == "LEV_DBAR"

    # Error is raised if neither depth column is present
    def test_raises_when_neither(self):
        df = pd.DataFrame({"DEPTH": [0.0], "VAL": [1.0]})
        with pytest.raises(KeyError, match="Depth column"):
            detect_depth_col(df)

    # detect_depth_col's fallback is used transparently by vertical_gradient
    def test_analysis_with_lev_m(self):
        df = pd.DataFrame({"PROFILE_NUMBER": [1, 1, 1],
                           "LEV_M": [0.0, 100.0, 200.0],
                           "VAL": [10.0, 8.0, 6.0]})
        result = vertical_gradient(df)
        assert "VAL_gradient" in result.columns
