"""Tests for comfort.physics, i.e. TEOS-10 physical oceanography functions."""
import logging

import gsw
import pandas as pd
import numpy as np
import pytest

from comfort.physics import (
    add_teos10_variables,
    compute_aou,
    compute_buoyancy_frequency,
    compute_density,
    compute_potential_vorticity,
    compute_spiciness,
    convert_salinity,
    convert_temperature,
    convert_to_potential_temperature,
    resolve_pressure,
)


class TestResolvePressure:
    # A measured pressure is used as is
    def test_measured_pressure_wins(self):
        df = pd.DataFrame({"LEV_DBAR": [100.0], "LEV_M": [50.0], "LATITUDE": [45.0]})
        assert resolve_pressure(df).iloc[0] == 100.0

    # Missing pressure is computed from depth (positive down) and latitude
    def test_pressure_from_depth_and_latitude(self):
        df = pd.DataFrame({"LEV_M": [1000.0], "LATITUDE": [45.0]})
        expected = float(gsw.p_from_z(-1000.0, 45.0))
        assert np.isclose(resolve_pressure(df).iloc[0], expected)

    # Each row is resolved on its own
    def test_rows_are_resolved_independently(self):
        df = pd.DataFrame({"LEV_DBAR": [250.0, np.nan], "LEV_M": [240.0, 1000.0], "LATITUDE": [45.0, 45.0]})
        result = resolve_pressure(df)
        assert result.iloc[0] == 250.0
        assert np.isclose(result.iloc[1], float(gsw.p_from_z(-1000.0, 45.0)))

    # Without latitude the depth [m] is used as a proxy (logged)
    def test_depth_proxy_without_latitude(self, caplog):
        df = pd.DataFrame({"LEV_M": [10.0, 20.0]})
        with caplog.at_level(logging.WARNING):
            result = resolve_pressure(df)
        assert list(result) == [10.0, 20.0]
        assert "2 of 2 rows without pressure and latitude" in caplog.text

    # Nothing to resolve from: NaN (logged)
    def test_nan_without_any_information(self, caplog):
        df = pd.DataFrame({"VAL": [1.0, 2.0]})
        with caplog.at_level(logging.WARNING):
            result = resolve_pressure(df)
        assert result.isna().all()
        assert "2 of 2 rows have no pressure information" in caplog.text

    # A row with neither pressure nor depth stays NaN, the others are resolved
    def test_partial_nan(self):
        df = pd.DataFrame({"LEV_DBAR": [np.nan, np.nan], "LEV_M": [100.0, np.nan], "LATITUDE": [45.0, 45.0]})
        result = resolve_pressure(df)
        assert np.isfinite(result.iloc[0])
        assert np.isnan(result.iloc[1])

    # Depths far above the surface are invalid and must not crash gsw or serve as a proxy
    def test_invalid_negative_depth_is_nan(self):
        df = pd.DataFrame({"LEV_M": [100.0, -50.0], "LATITUDE": [45.0, 45.0]})
        result = resolve_pressure(df)
        assert np.isfinite(result.iloc[0])
        assert np.isnan(result.iloc[1])

    # Column names are matched case-insensitively and the result keeps the input index
    def test_case_insensitive_and_index_preserved(self):
        df = pd.DataFrame({"lev_m": [1000.0, 2000.0], "latitude": [45.0, 45.0]}, index=[7, 9])
        result = resolve_pressure(df)
        assert list(result.index) == [7, 9]
        assert np.isfinite(result).all()

    # Custom column names are honoured
    def test_custom_columns(self):
        df = pd.DataFrame({"p": [5.0], "z": [100.0], "lat": [0.0]})
        assert resolve_pressure(df, pressure_col="p", depth_col="z", lat_col="lat").iloc[0] == 5.0


class TestConvertSalinity:
    # Output is always a pandas Series
    def test_returns_series(self):
        assert isinstance(convert_salinity(sp=34.5, pressure=200, longitude=-20, latitude=30), pd.Series)

    # A scalar input yields a length-1 Series
    def test_scalar_gives_length_one(self):
        assert len(convert_salinity(sp=34.5, pressure=0, longitude=0, latitude=0)) == 1

    # Array-like inputs are converted element-wise
    def test_array_input(self):
        result = convert_salinity(sp=np.array([34.0, 34.5, 35.0]), pressure=np.array([0, 100, 200]),
                                  longitude=np.array([-20, -20, -20]), latitude=np.array([30, 30, 30]))
        assert len(result) == 3

    # Check broad direction: Absolute Salinity is slightly higher than Practical Salinity in open ocean
    def test_sa_greater_than_sp(self):
        result = convert_salinity(sp=34.5, pressure=200, longitude=-20, latitude=30)
        assert float(result.iloc[0]) > 34.5


class TestConvertTemperature:
    # Output is always a pandas Series
    def test_returns_series(self):
        assert isinstance(convert_temperature(t_insitu=10.0, sa=34.6, pressure=200), pd.Series)

    # An unrecognised target raises value error
    def test_invalid_to_raises(self):
        with pytest.raises(ValueError, match="Unknown target"):
            convert_temperature(t_insitu=10.0, sa=34.6, pressure=200, to="bad")

    # Array-like inputs are converted element-wise
    def test_array_input(self):
        result = convert_temperature(t_insitu=np.array([10.0, 15.0, 20.0]), sa=np.array([34.6, 35.0, 35.5]),
                                     pressure=np.array([200, 100, 0]))
        assert len(result) == 3

    # Check broad direction: Conservative Temperature differs from in-situ temperature away from the surface
    def test_ct_differs_from_in_situ_at_depth(self):
        result = convert_temperature(t_insitu=10.0, sa=34.6, pressure=200)
        assert float(result.iloc[0]) != 10.0


class TestComputeDensity:
    # Default quantity (sigma0) returns a pandas Series
    def test_returns_series(self):
        assert isinstance(compute_density(sa=34.6, ct=10.0), pd.Series)

    # sigma1 (referenced to 1000 dbar) also returns a Series
    def test_sigma1_returns_series(self):
        assert isinstance(compute_density(sa=34.6, ct=10.0, quantity="sigma1"), pd.Series)

    # sigma2 (referenced to 2000 dbar) also returns a Series
    def test_sigma2_returns_series(self):
        assert isinstance(compute_density(sa=34.6, ct=10.0, quantity="sigma2"), pd.Series)

    # In-situ density (rho) at depth falls within a plausible range
    def test_rho_at_depth(self):
        rho = compute_density(34.6, 10.0, 200, quantity="rho")
        assert 1025.0 < float(rho.iloc[0]) < 1035.0

    # rho requires a pressure argument, unlike the sigma* quantities
    def test_rho_without_pressure_raises(self):
        with pytest.raises(ValueError, match="pressure is required"):
            compute_density(34.6, 10.0, quantity="rho")

    # An unrecognised quantity name raises instead of guessing
    def test_invalid_quantity_raises(self):
        with pytest.raises(ValueError, match="Unknown quantity"):
            compute_density(34.6, 10.0, quantity="bad")

    # Cold, salty water is denser (higher sigma0) than warm, fresh water
    def test_denser_water_has_higher_sigma0(self):
        light = compute_density(33.0, 25.0, quantity="sigma0")
        heavy = compute_density(35.0, 2.0, quantity="sigma0")
        assert float(heavy.iloc[0]) > float(light.iloc[0])


class TestComputeBuoyancyFrequency:
    @staticmethod
    def _stable_df():
        return pd.DataFrame({
            "PROFILE_NUMBER": [1, 1, 1],
            "LEV_DBAR": [0.0, 100.0, 200.0],
            "SA": [34.0, 34.5, 35.0],
            "CT": [20.0, 15.0, 10.0],
            "LATITUDE": [30.0, 30.0, 30.0],
        })

    # Output is a DataFrame
    def test_returns_dataframe(self):
        result = compute_buoyancy_frequency(self._stable_df(), "SA", "CT")
        assert isinstance(result, pd.DataFrame)

    # Output carries profile ID, mid-point pressure and N2 columns
    def test_expected_columns(self):
        result = compute_buoyancy_frequency(self._stable_df(), "SA", "CT")
        assert {"PROFILE_NUMBER", "pressure_mid", "N2"}.issubset(result.columns)

    # N mid-points are computed between consecutive depth levels
    def test_n_rows_is_n_levels_minus_one(self):
        result = compute_buoyancy_frequency(self._stable_df(), "SA", "CT")
        assert len(result) == 2  # 3 levels -> 2 mid-points

    # A profile that gets denser with depth is stably stratified (N2 > 0)
    def test_stable_profile_positive_n2(self):
        result = compute_buoyancy_frequency(self._stable_df(), "SA", "CT")
        assert (result["N2"] > 0).all()

    # Mid-point pressures fall strictly between their bounding levels
    def test_mid_depths_between_levels(self):
        result = compute_buoyancy_frequency(self._stable_df(), "SA", "CT")
        assert 0 < result["pressure_mid"].iloc[0] < 100
        assert 100 < result["pressure_mid"].iloc[1] < 200

    # A profile with only one level has no gradient to compute
    def test_single_level_profile_skipped(self):
        df = pd.DataFrame({
            "PROFILE_NUMBER": [1], "LEV_DBAR": [0.0],
            "SA": [34.0], "CT": [20.0],
        })
        assert compute_buoyancy_frequency(df, "SA", "CT").empty

    # Each profile is processed independently
    def test_multiple_profiles(self):
        df = pd.concat([self._stable_df(),
                        self._stable_df().assign(PROFILE_NUMBER=2)])
        result = compute_buoyancy_frequency(df, "SA", "CT")
        assert set(result["PROFILE_NUMBER"].unique()) == {1, 2}
        assert len(result) == 4

    # LATITUDE is optional (only refines the gravity constant)
    def test_works_without_latitude_column(self):
        df = self._stable_df().drop(columns="LATITUDE")
        result = compute_buoyancy_frequency(df, "SA", "CT")
        assert len(result) == 2

    # PROFILE_NUMBER is only unique within a station -> Stations sharing it must not be merged
    def test_stations_sharing_profile_number_are_separate(self):
        df = pd.concat([self._stable_df().assign(ID=1), self._stable_df().assign(ID=2)], ignore_index=True)
        result = compute_buoyancy_frequency(df, "SA", "CT")
        assert len(result) == 4
        assert {"ID", "PROFILE_NUMBER", "pressure_mid", "N2"}.issubset(result.columns)
        assert sorted(result["ID"].unique()) == [1, 2]
        assert (result["N2"] > 0).all()

    # Each station gives the same result as when it is processed alone
    def test_station_result_matches_processing_it_alone(self):
        df = pd.concat([self._stable_df().assign(ID=1), self._stable_df().assign(ID=2)], ignore_index=True)
        combined = compute_buoyancy_frequency(df, "SA", "CT")
        alone = compute_buoyancy_frequency(self._stable_df().assign(ID=1), "SA", "CT")
        assert np.allclose(combined[combined["ID"] == 1]["N2"], alone["N2"])

    # Without an ID column the profile number alone is the key
    def test_without_id_column_no_id_in_output(self):
        result = compute_buoyancy_frequency(self._stable_df(), "SA", "CT")
        assert "ID" not in result.columns

    # An explicit profile_col is used as given
    def test_explicit_profile_col(self):
        df = pd.concat([self._stable_df().assign(ID=1), self._stable_df().assign(ID=1, PROFILE_NUMBER=2)], ignore_index=True)
        result = compute_buoyancy_frequency(df, "SA", "CT", profile_col="PROFILE_NUMBER")
        assert list(result.columns[:1]) == ["PROFILE_NUMBER"]
        assert len(result) == 4

    # Without any profile (empty input) the result still has the key and N2 columns
    def test_empty_input_keeps_columns(self):
        df = pd.DataFrame({c: pd.Series(dtype=float) for c in ["ID", "PROFILE_NUMBER", "LEV_DBAR", "SA", "CT"]})
        result = compute_buoyancy_frequency(df, "SA", "CT")
        assert result.empty
        assert list(result.columns) == ["ID", "PROFILE_NUMBER", "pressure_mid", "N2"]


class TestComputePotentialVorticity:
    # Output is a pandas Series
    def test_returns_series(self):
        assert isinstance(compute_potential_vorticity(1e-4, 45.0), pd.Series)

    # A scalar input yields a length-1 Series
    def test_scalar_gives_length_one(self):
        assert len(compute_potential_vorticity(1e-4, 45.0)) == 1

    # Array-like inputs are converted element-wise
    def test_array_input(self):
        result = compute_potential_vorticity(np.array([1e-4, 2e-4]), np.array([45.0, 45.0]))
        assert len(result) == 2

    # Northern Hemisphere (f > 0): Stable stratification (N2 > 0) gives positive PV
    def test_positive_in_northern_hemisphere(self):
        assert float(compute_potential_vorticity(1e-4, 45.0).iloc[0]) > 0

    # Southern Hemisphere (f < 0) flips the sign of PV for the same N2
    def test_negative_in_southern_hemisphere(self):
        assert float(compute_potential_vorticity(1e-4, -45.0).iloc[0]) < 0

    # At the equator f = 0, so PV vanishes regardless of stratification
    def test_zero_at_equator(self):
        assert float(compute_potential_vorticity(1e-4, 0.0).iloc[0]) == pytest.approx(0.0, abs=1e-15)

    # PV scales linearly with N2 at fixed latitude
    def test_scales_linearly_with_n2(self):
        pv_1x = compute_potential_vorticity(1e-4, 45.0).iloc[0]
        pv_2x = compute_potential_vorticity(2e-4, 45.0).iloc[0]
        assert float(pv_2x) == pytest.approx(2 * float(pv_1x))

    # Composes with compute_buoyancy_frequency's N2 output (the intended pipeline)
    def test_composes_with_buoyancy_frequency(self):
        df = pd.DataFrame({
            "PROFILE_NUMBER": [1, 1, 1],
            "LEV_DBAR": [0.0, 100.0, 200.0],
            "SA": [34.0, 34.5, 35.0],
            "CT": [20.0, 15.0, 10.0],
            "LATITUDE": [30.0, 30.0, 30.0],
        })
        n2_result = compute_buoyancy_frequency(df, "SA", "CT")
        pv = compute_potential_vorticity(n2_result["N2"], 30.0)
        assert len(pv) == len(n2_result)
        assert (pv > 0).all()  # stable stratification, Northern Hemisphere


class TestComputeSpiciness:
    # Output is a pandas Series
    def test_returns_series(self):
        assert isinstance(compute_spiciness(34.6, 10.0), pd.Series)

    # Warm, salty water is spicier than cold, fresh water
    def test_warm_salty_spicier_than_cold_fresh(self):
        spicy = compute_spiciness(35.5, 20.0)
        bland = compute_spiciness(33.0, 5.0)
        assert float(spicy.iloc[0]) > float(bland.iloc[0])

    # reference_pressure=2000 selects the spiciness2 formulation
    def test_reference_2000_returns_series(self):
        assert isinstance(compute_spiciness(34.6, 10.0, reference_pressure=2000), pd.Series)

    # Only reference pressures with a defined formulation are accepted
    def test_invalid_reference_raises(self):
        with pytest.raises(ValueError, match="reference_pressure must be"):
            compute_spiciness(34.6, 10.0, reference_pressure=500)


class TestComputeAOU:
    # Output is a pandas Series
    def test_returns_series(self):
        assert isinstance(compute_aou(275.0, 34.6, 10.0), pd.Series)

    # Oxygen exactly at equilibrium solubility gives AOU == 0
    def test_at_equilibrium_gives_zero(self):
        o2_eq = gsw.O2sol_SP_pt(34.6, 10.0)
        assert abs(float(compute_aou(o2_eq, 34.6, 10.0).iloc[0])) < 1e-6

    # Undersaturated water (less O2 than equilibrium) gives positive AOU
    def test_undersaturated_gives_positive_aou(self):
        assert float(compute_aou(200.0, 34.6, 10.0).iloc[0]) > 0

    # Supersaturated water (more O2 than equilibrium) gives negative AOU
    def test_supersaturated_gives_negative_aou(self):
        assert float(compute_aou(350.0, 34.6, 10.0).iloc[0]) < 0


class TestAddTeos10Variables:
    @staticmethod
    def _df():
        return pd.DataFrame({
            "LEV_DBAR": [0.0, 100.0],
            "SP": [34.5, 34.8],
            "TEMP": [15.0, 12.0],
            "LONGITUDE": [-20.0, -20.0],
            "LATITUDE": [30.0, 30.0],
        })

    # SA, CT and sigma0 columns are all added in one call
    def test_adds_sa_ct_sigma0(self):
        result = add_teos10_variables(self._df(), sp_col="SP", t_col="TEMP")
        assert {"SA", "CT", "sigma0"}.issubset(result.columns)

    # The input DataFrame is not modified in place
    def test_original_df_not_mutated(self):
        df = self._df()
        original_cols = set(df.columns)
        add_teos10_variables(df, sp_col="SP", t_col="TEMP")
        assert set(df.columns) == original_cols

    # Check broad direction: Absolute Salinity is slightly higher than Practical Salinity in open ocean
    def test_sa_greater_than_sp(self):
        result = add_teos10_variables(self._df(), sp_col="SP", t_col="TEMP")
        assert (result["SA"] > result["SP"]).all()


class TestConvertToPotentialTemperature:
    """Operates on the per-parameter dict shape, one row per location."""

    def _loc(self, **overrides):
        base = {"LATITUDE": 10.0, "LONGITUDE": 20.0, "LEV_M": 1000.0,
                "LEV_DBAR": 1009.0, "DATEANDTIME": "2000-01-15"}
        base.update(overrides)
        return base

    # At depth, potential temperature is measurably below in-situ temperature
    def test_differs_from_in_situ_temperature_at_depth(self):
        averaged = {
            "TEMPERATURE": pd.DataFrame([{**self._loc(), "TEMPERATURE": 4.0}]),
            "SALINITY": pd.DataFrame([{**self._loc(), "SALINITY": 35.0}]),
        }
        converted = convert_to_potential_temperature(averaged)
        assert converted["TEMPERATURE"]["TEMPERATURE"].iloc[0] < averaged["TEMPERATURE"]["TEMPERATURE"].iloc[0]

    # Parameters other than TEMPERATURE pass through unchanged
    def test_leaves_other_parameters_untouched(self):
        averaged = {
            "TEMPERATURE": pd.DataFrame([{**self._loc(), "TEMPERATURE": 4.0}]),
            "SALINITY": pd.DataFrame([{**self._loc(), "SALINITY": 35.0}]),
            "OXYGEN": pd.DataFrame([{**self._loc(), "OXYGEN": 200.0}]),
        }
        result = convert_to_potential_temperature(averaged)
        assert result["SALINITY"]["SALINITY"].iloc[0] == pytest.approx(35.0)
        assert result["OXYGEN"]["OXYGEN"].iloc[0] == pytest.approx(200.0)

    # SALINITY is required to convert TEMPERATURE
    def test_raises_when_salinity_missing(self):
        averaged = {"TEMPERATURE": pd.DataFrame([{**self._loc(), "TEMPERATURE": 4.0}])}
        with pytest.raises(KeyError):
            convert_to_potential_temperature(averaged)

    # TEMPERATURE itself must be present to have anything to convert
    def test_raises_when_temperature_missing(self):
        averaged = {"SALINITY": pd.DataFrame([{**self._loc(), "SALINITY": 35.0}])}
        with pytest.raises(KeyError):
            convert_to_potential_temperature(averaged)
