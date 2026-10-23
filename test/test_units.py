"""Tests for comfort.units."""
import logging
import sqlite3
import gsw
import numpy as np
import pandas as pd
import pytest

from comfort.units import (
    ConversionFormulas,
    UnitsConverter,
)
from comfort.database.information import does_table_exist


class TestConversionFormulas:
    @pytest.fixture
    def formulas(self):
        return ConversionFormulas()

    # The registry covers exactly the (source, target) unit pairs comfort-db supports
    def test_registry_contains_known_conversions(self, formulas):
        expected_pairs = {(5, 3), (4, 14), (7, 3), (12, 19), (21, 3), (14, 3), (4, 3), (26, 3), (10, 3)}
        assert expected_pairs == set(formulas.conversions)

    # nanomol/kg -> femtomol/kg is a straight x 1e6 scale
    def test_nanomol_to_femtomol(self, formulas):
        df = pd.DataFrame({"VAL": [1.0, 2.0], "UNITS_ID": [12, 12]})
        result = formulas.nanomolPerKilogram_femtomolPerKilogram(df, "SF6")
        assert list(result["VAL"]) == [1_000_000.0, 2_000_000.0]
        assert list(result["UNITS_ID"]) == [19, 19]

    # Molar mass lookup succeeds for a known parameter table
    def test_get_molar_mass_known(self, formulas):
        mass = formulas.get_molar_mass("P_NITRATE")
        assert abs(mass - 14.00672) < 1e-5

    # An unregistered parameter table raises
    def test_get_molar_mass_unknown_raises(self, formulas):
        with pytest.raises(ValueError, match="not defined"):
            formulas.get_molar_mass("P_UNKNOWN_PARAM")

    # Without salinity/temperature columns, density falls back to the constant 1.025
    def test_density_falls_back_without_columns(self, formulas, caplog):
        df = pd.DataFrame({"VAL": [2.0], "UNITS_ID": [7]})
        with caplog.at_level(logging.WARNING):
            density = formulas.real_density(df, use_density=True)
        assert (density == 1.025).all()  # density fell back to constant
        assert "1 of 1 rows undefined" in caplog.text  # undefined density count is logged
        assert "constant density" in caplog.text  # density policy is logged

    @staticmethod
    def _ts_frame(salinity):
        return pd.DataFrame({
            "VAL": [2.0] * len(salinity), "UNITS_ID": [7] * len(salinity),
            "salinity": salinity, "LATITUDE": [45.0] * len(salinity),
            "LONGITUDE": [-30.0] * len(salinity),
        })

    # NaN salinity rows get the constant (default density policy), computed rows are untouched, counts logged
    def test_density_nan_rows_use_constant_and_log_count(self, formulas, caplog):
        df = self._ts_frame([35.0, np.nan, 34.0])
        with caplog.at_level(logging.WARNING):
            # Warning is logged that density falls back to constant
            density = formulas.lab_density(df, use_density=True)
        assert density.iloc[1] == 1.025  # constant density for nan-salinity row
        assert density.iloc[0] != 1.025 and density.iloc[2] != 1.025  # not-constant density for not-nan-salinity rows
        assert "1 of 3 rows undefined" in caplog.text  # counts are logged

    # density_fallback='drop' removes rows without a density from the conversion
    def test_density_fallback_drop_removes_rows(self, formulas):
        df = self._ts_frame([35.0, np.nan, 34.0])
        result = formulas.millimolPerLiter_micromolPerKilogram(
            df, "P_DIC", use_density=True, density_fallback="drop")
        assert list(result.index) == [0, 2]  # index is not changed
        assert result["VAL"].notna().all()  # no nan-values

    # density_fallback='drop' with a missing column removes every row
    def test_density_fallback_drop_missing_column_removes_all(self, formulas):
        df = pd.DataFrame({"VAL": [2.0], "UNITS_ID": [7]})
        result = formulas.millimolPerLiter_micromolPerKilogram(
            df, "P_DIC", use_density=True, density_fallback="drop")
        assert result.empty

    # density_fallback='raise' fails on NaN rows and on missing columns
    def test_density_fallback_raise(self, formulas):
        with pytest.raises(ValueError, match="1 of 2 rows undefined"):
            formulas.lab_density(self._ts_frame([35.0, np.nan]), True, "raise")
        with pytest.raises(ValueError, match="columns missing"):
            formulas.lab_density(pd.DataFrame({"VAL": [1.0]}), True, "raise")

    # Complete inputs never trigger the fallback, whatever the policy
    def test_density_fallback_irrelevant_when_inputs_complete(self, formulas):
        df = self._ts_frame([35.0, 34.0])
        expected = formulas.lab_density(df, True)
        for policy in ("constant", "drop", "raise"):
            assert formulas.lab_density(df, True, policy).equals(expected)

    # The constant mode (use_density=False) is independent of the policy
    def test_density_fallback_ignored_without_use_density(self, formulas):
        df = pd.DataFrame({"VAL": [1.0]})
        for policy in ("constant", "drop", "raise"):
            assert formulas.lab_density(df, False, policy) == 1.025

    # Unknown policies are rejected
    def test_density_fallback_invalid_raises(self, formulas):
        with pytest.raises(ValueError, match="density_fallback"):
            formulas.lab_density(pd.DataFrame({"VAL": [1.0]}), True, "bogus")

    # App. C lab_dens: Sample S, T = 22 °C, atmospheric pressure (p = 0)
    def test_lab_density_uses_22C_at_surface(self, formulas):
        import gsw
        df = pd.DataFrame({
            "VAL": [2.0], "UNITS_ID": [7], "salinity": [35.0], "temperature": [2.0],
            "LATITUDE": [45.0], "LONGITUDE": [-30.0],
        })
        density = formulas.lab_density(df, use_density=True)
        sa = gsw.SA_from_SP(35.0, 0.0, -30.0, 45.0)
        expected = gsw.density.rho_t_exact(sa, 22.0, 0.0) / 1000
        assert np.isclose(float(density.iloc[0]), expected)
        # In-situ temperature (2 °C) must NOT be used
        insitu = gsw.density.rho_t_exact(sa, 2.0, 0.0) / 1000
        assert not np.isclose(float(density.iloc[0]), insitu)

    # App. C real_dens: Sample S, in-situ T, atmospheric pressure (p = 0)
    def test_real_density_uses_insitu_temperature_at_surface(self, formulas):
        import gsw
        df = pd.DataFrame({
            "VAL": [2.0], "UNITS_ID": [21], "SALINITY": [34.84], "TEMPERATURE": [6.29],
            "LATITUDE": [69.07], "LONGITUDE": [17.33],
        })
        density = formulas.real_density(df, use_density=True)
        sa = gsw.SA_from_SP(34.84, 0.0, 17.33, 69.07)
        expected = gsw.density.rho_t_exact(sa, 6.29, 0.0) / 1000
        assert np.isclose(float(density.iloc[0]), expected)

    # App. C: same 4->14 conversion, but chlorophyll real_dens vs ammonium lab_dens
    def test_chlorophyll_uses_real_density_ammonium_lab_density(self, formulas):
        df = pd.DataFrame({
            "VAL": [1.0], "UNITS_ID": [4], "salinity": [35.0], "temperature": [2.0],
            "LATITUDE": [45.0], "LONGITUDE": [-30.0],
        })
        chl = formulas.microgramPerLiter_microgramPerKilogram(df, "P_CHLOROPHYLL", True)
        amm = formulas.microgramPerLiter_microgramPerKilogram(df, "P_AMMONIUM", True)
        assert not np.isclose(chl["VAL"].iloc[0], amm["VAL"].iloc[0])
        expected_chl = 1.0 / float(formulas.real_density(df, True).iloc[0])
        expected_amm = 1.0 / float(formulas.lab_density(df, True).iloc[0])
        assert np.isclose(chl["VAL"].iloc[0], expected_chl)
        assert np.isclose(amm["VAL"].iloc[0], expected_amm)


def _make_converter(default_unit=3, table="P_OXYGEN"):
    """UnitsConverter for DataFrame-only tests (no database access needed)."""
    default_units = pd.DataFrame({"NAME_TABLE": [table], "UNITS_ID_DEFAULT": [default_unit]})
    units = pd.DataFrame({"ID": list(range(30)), "NAME_SHORT": [str(i) for i in range(30)]})
    return UnitsConverter(connection=None, default_units=default_units, units=units)


class TestConvertDataframe:
    # Rows already in the default unit pass through unchanged
    def test_already_default_unit_passthrough(self):
        uc = _make_converter()
        df = pd.DataFrame({"VAL": [200.0, 210.0], "UNITS_ID": [3, 3]})
        result = uc.convert_dataframe(df, "P_OXYGEN")
        assert result.equals(df)

    # density_fallback is threaded through convert_dataframe (DIC mmol/L -> umol/kg, UNITS_ID 7 -> 3)
    def test_density_fallback_policies(self):
        uc = _make_converter(table="P_DIC")
        df = pd.DataFrame({"VAL": [2.0, 2.0], "UNITS_ID": [7, 7], "salinity": [35.0, np.nan],
                           "LATITUDE": [45.0, 45.0], "LONGITUDE": [-30.0, -30.0]})

        # Constant density policy
        constant = uc.convert_dataframe(df, "P_DIC", use_density=True)
        assert np.isclose(constant["VAL"].iloc[1], 2.0 * 1000 / 1.025)  # constand density due to missing salinity
        assert not np.isclose(constant["VAL"].iloc[0], 2.0 * 1000 / 1.025)  # density should have been computed

        # Drop nan-densities policy
        dropped = uc.convert_dataframe(df, "P_DIC", use_density=True, density_fallback="drop")
        assert list(dropped.index) == [0]

        # Error raising
        with pytest.raises(ValueError, match="undefined"):
            # On un-computable density
            uc.convert_dataframe(df, "P_DIC", use_density=True, density_fallback="raise")
        with pytest.raises(ValueError, match="density_fallback"):
            # On unknown density policy
            uc.convert_dataframe(df, "P_DIC", density_fallback="bogus")

    # Percent saturation converts to µmol/kg using lowercase T/S columns
    def test_percent_with_ts_converts(self):
        uc = _make_converter()
        df = pd.DataFrame({"VAL": [80.0], "UNITS_ID": [10], "salinity": [35.0], "temperature": [10.0]})
        result = uc.convert_dataframe(df, "P_OXYGEN")
        expected = 80.0 * float(gsw.O2sol_SP_pt(35.0, 10.0)) / 100
        assert np.isclose(result["VAL"].iloc[0], expected)
        assert result["UNITS_ID"].iloc[0] == 3

    # T/S column detection is case-insensitive
    def test_percent_with_uppercase_ts_columns(self):
        uc = _make_converter()
        df = pd.DataFrame({"VAL": [80.0], "UNITS_ID": [10],
                           "SALINITY": [35.0], "TEMPERATURE": [10.0]})
        result = uc.convert_dataframe(df, "P_OXYGEN")
        expected = 80.0 * float(gsw.O2sol_SP_pt(35.0, 10.0)) / 100
        assert np.isclose(result["VAL"].iloc[0], expected)

    # Without pressure/lat/lon, in-situ T is used as pt0 (logged)
    def test_percent_without_geo_warns_and_approximates(self, caplog):
        uc = _make_converter()
        df = pd.DataFrame({"VAL": [80.0], "UNITS_ID": [10], "salinity": [35.0], "temperature": [10.0]})
        with caplog.at_level(logging.WARNING):
            uc.convert_dataframe(df, "P_OXYGEN")
        assert "using in-situ temperature as an approximation" in caplog.text

    # With pressure/lat/lon present, real potential temperature (pt0) is used
    def test_percent_with_geo_uses_real_pt0(self):
        uc = _make_converter()
        df = pd.DataFrame({"VAL": [80.0], "UNITS_ID": [10],
                           "salinity": [35.0], "temperature": [10.0],
                           "LEV_DBAR": [2000.0], "LATITUDE": [45.0], "LONGITUDE": [-30.0]})
        result = uc.convert_dataframe(df, "P_OXYGEN")

        sa = gsw.SA_from_SP(35.0, 2000.0, -30.0, 45.0)
        pt0 = gsw.pt0_from_t(sa, 10.0, 2000.0)
        expected = 80.0 * float(gsw.O2sol_SP_pt(35.0, pt0)) / 100
        approx_with_insitu_t = 80.0 * float(gsw.O2sol_SP_pt(35.0, 10.0)) / 100

        assert np.isclose(result["VAL"].iloc[0], expected)
        assert not np.isclose(result["VAL"].iloc[0], approx_with_insitu_t)

    @staticmethod
    def _saturation_frame(**extra):
        data = {"VAL": [80.0], "UNITS_ID": [10], "salinity": [35.0], "temperature": [10.0], "LATITUDE": [45.0], "LONGITUDE": [-30.0]}
        data.update({k: [v] for k, v in extra.items()})
        return pd.DataFrame(data)

    # Without LEV_DBAR, pressure is derived from depth (LEV_M, positive down) and latitude
    def test_percent_pressure_from_depth(self):
        uc = _make_converter()
        from_depth = uc.convert_dataframe(self._saturation_frame(LEV_M=1000.0), "P_OXYGEN")
        pressure = float(gsw.p_from_z(-1000.0, 45.0))
        from_pressure = uc.convert_dataframe(self._saturation_frame(LEV_DBAR=pressure), "P_OXYGEN")
        # Depth- and pressure-based conversions are similar
        assert np.isclose(from_depth["VAL"].iloc[0], from_pressure["VAL"].iloc[0])

    # An explicit LEV_DBAR is selected over LEV_M
    def test_percent_lev_dbar_takes_precedence_over_depth(self):
        uc = _make_converter()
        both = uc.convert_dataframe(self._saturation_frame(LEV_DBAR=2000.0, LEV_M=10.0), "P_OXYGEN")
        only_pressure = uc.convert_dataframe(self._saturation_frame(LEV_DBAR=2000.0), "P_OXYGEN")
        assert np.isclose(both["VAL"].iloc[0], only_pressure["VAL"].iloc[0])

    # NaN LEV_DBAR rows are filled from LEV_M
    def test_percent_nan_pressure_filled_from_depth(self):
        uc = _make_converter()
        pressure = float(gsw.p_from_z(-1000.0, 45.0))
        df = pd.concat([self._saturation_frame(LEV_DBAR=np.nan, LEV_M=1000.0),
                        self._saturation_frame(LEV_DBAR=pressure, LEV_M=1000.0)], ignore_index=True)
        result = uc.convert_dataframe(df, "P_OXYGEN")
        assert np.isclose(result["VAL"].iloc[0], result["VAL"].iloc[1])

    # Neither pressure nor depth: In-situ T approximation with a warning
    def test_percent_without_pressure_or_depth_warns(self, caplog):
        uc = _make_converter()
        with caplog.at_level(logging.WARNING):
            result = uc.convert_dataframe(self._saturation_frame(), "P_OXYGEN")
        assert "using in-situ temperature as an approximation" in caplog.text
        expected = 80.0 * float(gsw.O2sol_SP_pt(35.0, 10.0)) / 100
        assert np.isclose(result["VAL"].iloc[0], expected)

    # A physically impossible negative depth does not crash gsw, the row is treated as not-convertible
    def test_percent_negative_depth_row_is_unconvertible(self):
        uc = _make_converter()
        df = pd.concat([self._saturation_frame(LEV_M=100.0),
                        self._saturation_frame(LEV_M=-50.0)], ignore_index=True)
        result = uc.convert_dataframe(df, "P_OXYGEN")
        assert list(result.index) == [0]

    # Default on_missing="drop": A row needing T/S but missing it is dropped (logged)
    def test_percent_without_ts_dropped(self, caplog):
        uc = _make_converter()
        df = pd.DataFrame({"VAL": [80.0, 200.0], "UNITS_ID": [10, 3]})
        with caplog.at_level(logging.WARNING):
            result = uc.convert_dataframe(df, "P_OXYGEN")
        assert len(result) == 1
        assert result["VAL"].iloc[0] == 200.0
        assert "missing temperature/salinity" in caplog.text

    # on_missing="keep" keeps the unconverted row
    def test_percent_without_ts_kept(self):
        uc = _make_converter()
        df = pd.DataFrame({"VAL": [80.0, 200.0], "UNITS_ID": [10, 3]})
        result = uc.convert_dataframe(df, "P_OXYGEN", on_missing="keep")
        assert len(result) == 2
        assert result["VAL"].iloc[0] == 80.0
        assert result["UNITS_ID"].iloc[0] == 10

    # on_missing="raise" raises
    def test_percent_without_ts_raises(self):
        uc = _make_converter()
        df = pd.DataFrame({"VAL": [80.0], "UNITS_ID": [10]})
        with pytest.raises(ValueError, match="could not be converted"):
            uc.convert_dataframe(df, "P_OXYGEN", on_missing="raise")

    # A NaN T/S value counts as missing input
    def test_percent_nan_ts_row_dropped(self, caplog):
        uc = _make_converter()
        df = pd.DataFrame({"VAL": [80.0, 90.0], "UNITS_ID": [10, 10], "salinity": [35.0, 35.0], "temperature": [10.0, np.nan]})
        with caplog.at_level(logging.WARNING):
            result = uc.convert_dataframe(df, "P_OXYGEN")
        assert len(result) == 1
        assert "missing temperature/salinity" in caplog.text

    # A NaN UNITS_ID is dropped by default (logged)
    def test_nan_units_id_dropped(self, caplog):
        uc = _make_converter(table="P_DIC")
        df = pd.DataFrame({"VAL": [1.0, 2.0], "UNITS_ID": [np.nan, 7]})
        with caplog.at_level(logging.WARNING):
            result = uc.convert_dataframe(df, "P_DIC")
        assert len(result) == 1
        assert np.isclose(result["VAL"].iloc[0], 2.0 * 1000 / 1.025)
        assert "without UNITS_ID" in caplog.text

    # on_missing="keep" keeps NaN-UNITS_ID rows unconverted
    def test_nan_units_id_kept(self):
        uc = _make_converter(table="P_DIC")
        df = pd.DataFrame({"VAL": [1.0, 2.0], "UNITS_ID": [np.nan, 7]})
        result = uc.convert_dataframe(df, "P_DIC", on_missing="keep")
        assert len(result) == 2
        assert result["VAL"].iloc[0] == 1.0
        assert pd.isna(result["UNITS_ID"].iloc[0])

    # on_missing="raise" turns a NaN UNITS_ID into an error
    def test_nan_units_id_raises(self):
        uc = _make_converter(table="P_DIC")
        df = pd.DataFrame({"VAL": [1.0, 2.0], "UNITS_ID": [np.nan, 7]})
        with pytest.raises(ValueError, match="no UNITS_ID"):
            uc.convert_dataframe(df, "P_DIC", on_missing="raise")

    # A UNITS_ID with no registered formula is dropped instead of crashing
    def test_unknown_units_id_no_crash(self, caplog):
        uc = _make_converter(table="P_DIC")
        df = pd.DataFrame({"VAL": [1.0, 2.0], "UNITS_ID": [999, 3]})
        with caplog.at_level(logging.WARNING):
            result = uc.convert_dataframe(df, "P_DIC")
        assert len(result) == 1
        assert result["VAL"].iloc[0] == 2.0
        assert "no conversion formula" in caplog.text

    # on_missing="raise" turns an unregistered UNITS_ID into an error
    def test_unknown_units_id_raises(self):
        uc = _make_converter(table="P_DIC")
        df = pd.DataFrame({"VAL": [1.0], "UNITS_ID": [999]})
        with pytest.raises(ValueError, match="no conversion"):
            uc.convert_dataframe(df, "P_DIC", on_missing="raise")

    # Converting different unit groups separately must not reorder the rows
    def test_row_order_preserved(self):
        uc = _make_converter(table="P_DIC")
        df = pd.DataFrame({"VAL": [1.0, 2.0, 3.0, 4.0], "UNITS_ID": [3, 7, 3, 7]})
        result = uc.convert_dataframe(df, "P_DIC")
        expected = [1.0, 2.0 * 1000 / 1.025, 3.0, 4.0 * 1000 / 1.025]
        assert np.allclose(result["VAL"], expected)
        assert list(result.index) == [0, 1, 2, 3]

    # An unrecognised on_missing policy string is rejected
    def test_invalid_on_missing_raises(self):
        uc = _make_converter()
        df = pd.DataFrame({"VAL": [1.0], "UNITS_ID": [3]})
        with pytest.raises(ValueError, match="on_missing"):
            uc.convert_dataframe(df, "P_OXYGEN", on_missing="bogus")


class TestParameterRestrictions:
    """Conversions listed in allowed_parameters are only applied to those parameters."""

    # Every restricted pair is a registered conversion (guards against typos in the registry)
    def test_restricted_pairs_are_registered(self):
        formulas = ConversionFormulas()
        assert set(formulas.allowed_parameters) <= set(formulas.conversions)

    # Checks for is_allowed
    def test_is_allowed(self):
        formulas = ConversionFormulas()
        assert formulas.is_allowed(5, 3, "P_ALKALINITY")  # listed param
        assert formulas.is_allowed(5, 3, "p_alkalinity")  # case-insensitive
        assert not formulas.is_allowed(5, 3, "P_NITRATE")  # restricted pair (only for alkalinity)
        assert formulas.is_allowed(12, 19, "P_ANYTHING")  # unrestricted pair

    # on_missing="drop" (default) drops and logs entries of a parameter that a conversion was not written for
    def test_disallowed_parameter_dropped(self, caplog):
        uc = _make_converter(table="P_NITRATE")
        df = pd.DataFrame({"VAL": [1.0, 2.0], "UNITS_ID": [21, 3]})
        with caplog.at_level(logging.WARNING):
            result = uc.convert_dataframe(df, "P_NITRATE")
        assert list(result["VAL"]) == [2.0]
        assert "not implemented for P_NITRATE" in caplog.text

    # on_missing="keep" retains the rows with their original value and unit
    def test_disallowed_parameter_kept(self):
        uc = _make_converter(table="P_NITRATE")
        df = pd.DataFrame({"VAL": [1.0], "UNITS_ID": [21]})
        result = uc.convert_dataframe(df, "P_NITRATE", on_missing="keep")
        assert result["VAL"].iloc[0] == 1.0
        assert result["UNITS_ID"].iloc[0] == 21

    # on_missing="raise" raises
    def test_disallowed_parameter_raises(self):
        uc = _make_converter(table="P_NITRATE")
        df = pd.DataFrame({"VAL": [1.0], "UNITS_ID": [21]})
        with pytest.raises(ValueError, match="no conversion"):
            uc.convert_dataframe(df, "P_NITRATE", on_missing="raise")

    # A mass concentration of a parameter without molar mass follows on_missing policy
    def test_unknown_molar_mass_parameter_follows_on_missing(self):
        uc = _make_converter(table="P_UNKNOWN_PARAM")
        df = pd.DataFrame({"VAL": [1.0, 2.0], "UNITS_ID": [4, 3]})
        assert list(uc.convert_dataframe(df, "P_UNKNOWN_PARAM")["VAL"]) == [2.0]
        with pytest.raises(ValueError, match="no conversion"):
            uc.convert_dataframe(df, "P_UNKNOWN_PARAM", on_missing="raise")


class TestRequiredInputs:
    """UnitsConverter.required_inputs tells load_comfort which co-located T/S to join."""

    @staticmethod
    def _converter():
        # Dummy converters
        tables = {"P_OXYGEN": 3, "P_CHLOROPHYLL": 14, "P_AMMONIUM": 14, "P_DIC": 3, "P_NITRATE": 3, "P_AOU": 3, "P_SALINITY": 2}
        default_units = pd.DataFrame({"NAME_TABLE": list(tables), "UNITS_ID_DEFAULT": list(tables.values())})
        units = pd.DataFrame({"ID": list(range(30)), "NAME_SHORT": [str(i) for i in range(30)]})
        return UnitsConverter(connection=None, default_units=default_units, units=units)

    # Oxygen % saturation always needs T and S (independent of density setting)
    @pytest.mark.parametrize("use_density", [False, True])
    def test_oxygen_needs_temperature_and_salinity(self, use_density):
        assert self._converter().required_inputs("P_OXYGEN", use_density=use_density) == (True, True)

    # Chlorophyll uses real_dens, which requires T and S (only with use_density)
    def test_chlorophyll(self):
        uc = self._converter()
        assert uc.required_inputs("P_CHLOROPHYLL", use_density=False) == (False, False)
        assert uc.required_inputs("P_CHLOROPHYLL", use_density=True) == (True, True)

    # lab_dens parameters only need salinity (only with use_density)
    @pytest.mark.parametrize("table", ["P_AMMONIUM", "P_DIC", "P_NITRATE"])
    def test_lab_density_parameters(self, table):
        uc = self._converter()
        assert uc.required_inputs(table, False) == (False, False)
        assert uc.required_inputs(table, True) == (False, True)

    # No conversion into the default unit (or no default unit) needs nothing
    @pytest.mark.parametrize("table", ["P_AOU", "P_SALINITY", "P_NOT_REGISTERED"])
    @pytest.mark.parametrize("use_density", [False, True])
    def test_parameters_without_conversion(self, table, use_density):
        assert self._converter().required_inputs(table, use_density) == (False, False)

    # The declared inputs match what the formulas really use:
    # - With all inputs the computed density is used
    # - Without a needed column the constant is used
    # - Without an unneeded column nothing changes
    @pytest.mark.parametrize("source,target,table", [
        (5, 3, "P_ALKALINITY"), (7, 3, "P_DIC"), (4, 3, "P_NITRATE"),
        (4, 14, "P_AMMONIUM"), (4, 14, "P_CHLOROPHYLL"), (21, 3, "P_OXYGEN"),
    ])
    def test_declared_inputs_match_formulas(self, source, target, table):
        uc = _make_converter(default_unit=target, table=table)
        full = pd.DataFrame({"VAL": [2.0], "UNITS_ID": [source], "salinity": [35.0], "temperature": [10.0],
                             "LATITUDE": [45.0], "LONGITUDE": [-30.0]})
        def convert(df, use_density):
            # Conversion returning only first row
            return float(uc.convert_dataframe(df, table, use_density=use_density)["VAL"].iloc[0])

        # Conversion results with/without density differ (-> computed density is used)
        constant = convert(full, use_density=False)
        computed = convert(full, use_density=True)
        assert not np.isclose(computed, constant)

        # Conversion without T/S
        declared = uc.formulas.t_s_inputs_needed(source, target, table, True)  # infer if T and/or S are needed
        for column in ("temperature", "salinity"):
            without = convert(full.drop(columns=column), use_density=True)  # no T/S conversion
            assert np.isclose(without, constant if column in declared else computed)  # close to conversion without density (if T/S was required)


class TestDefaultUnitName:
    # A default UNITS_ID resolves to its short name
    def test_known_unit(self):
        uc = _make_converter()
        assert uc.default_unit_name("P_OXYGEN") == "3"

    # A table with no default_units entry returns None instead of raising
    def test_unknown_table_returns_none(self):
        uc = _make_converter()
        assert uc.default_unit_name("P_DOESNOTEXIST") is None

    # A default UNITS_ID absent from the units lookup also returns None
    def test_unknown_unit_id_returns_none(self):
        uc = _make_converter(default_unit=99)
        assert uc.default_unit_name("P_OXYGEN") is None


class TestUnitsConverter:
    @pytest.fixture
    def in_memory_connection(self):
        """Return a fresh in-memory SQLite connection."""
        conn = sqlite3.connect(":memory:")
        yield conn
        conn.close()

    def _setup(self, conn, rows):
        conn.execute(
            "CREATE TABLE dummy (VAL REAL, UNITS_ID INTEGER, LATITUDE REAL, "
            "LONGITUDE REAL, LEV_DBAR REAL, salinity REAL, temperature REAL)"
        )
        conn.executemany("INSERT INTO dummy VALUES (?,?,?,?,?,?,?)", rows)
        conn.commit()
        default_units = pd.DataFrame({"NAME_TABLE": ["dummy"], "UNITS_ID_DEFAULT": [19]})
        units = pd.DataFrame({"ID": list(range(27))})
        return UnitsConverter(connection=conn, default_units=default_units, units=units)

    # override_old_tables=False writes a separate converted_<table>
    def test_convert_units_no_override(self, in_memory_connection):
        conn = in_memory_connection
        uc = self._setup(conn, [
            (10, 12, 50, 50, 0, 31, 4),
            (20, 19, -10, 30, 10, 35, 10),
        ])
        new_tables = uc.convert_units(["dummy"], use_density=False, override_old_tables=False)
        assert new_tables["dummy"] == "converted_dummy"
        assert does_table_exist(conn, "converted_dummy")

        unit_ids = [r[0] for r in conn.execute("SELECT DISTINCT UNITS_ID FROM converted_dummy;").fetchall()]
        assert unit_ids == [19]

    # override_old_tables=True replaces the original table in place
    def test_convert_units_with_override(self, in_memory_connection):
        conn = in_memory_connection
        uc = self._setup(conn, [
            (10, 12, 50, 50, 0, 31, 4),
            (20, 19, -10, 30, 10, 35, 10),
        ])
        new_tables = uc.convert_units(["dummy"], use_density=False, override_old_tables=True)
        assert new_tables["dummy"] == "dummy"
        assert not does_table_exist(conn, "converted_dummy")

        unit_ids = [r[0] for r in conn.execute("SELECT DISTINCT UNITS_ID FROM dummy;").fetchall()]
        assert unit_ids == [19]

    # When every row is already in the default unit, no converted_* table is created
    def test_convert_units_noop_keeps_original_table(self, in_memory_connection):
        conn = in_memory_connection
        uc = self._setup(conn, [
            (10, 19, 50, 50, 0, 31, 4),
            (20, 19, -10, 30, 10, 35, 10),
        ])
        new_tables = uc.convert_units(["dummy"], use_density=False, override_old_tables=False)
        assert new_tables["dummy"] == "dummy"
        assert not does_table_exist(conn, "converted_dummy")

    # A malicious/invalid table name is rejected before it reaches SQL
    def test_convert_units_invalid_table_name_raises(self, in_memory_connection):
        uc = self._setup(in_memory_connection, [(10, 19, 50, 50, 0, 31, 4)])
        with pytest.raises(ValueError, match="identifier"):
            uc.convert_units(["dummy; DROP TABLE dummy"])


# Reference conversion values, hand-verified against the COMFORT dataset v3 report, Appendix C.
# Densities were computed using the gsw library.
#
# Every row carries station_id + LEV_M identifying the exact COMFORT record(s) it was taken from
# Salinity/temperature values can be reproduced with:
#   SELECT AVG(VAL) FROM P_SALINITY WHERE ID=<station_id> AND LEV_M=<LEV_M>
CONVERSION_REFERENCE = pd.DataFrame([
    # --- ALKALINITY 5->3: meq/L -> umol/kg, VAL * 1000 / density (App. C: lab_dens) ---
    dict(param_name="ALKALINITY", UNITS_ID_in=5, VAL=2.327, station_id=3000009, LEV_M=0,
         salinity=33.2100, temperature=4.460000, LATITUDE=60.30000, LONGITUDE=2.95000,
         VAL_CONVERTED_NO_DENSITY=2.327 * 1000 / 1.025,
         VAL_CONVERTED_DENSITY=2.327 * 1000 / 1.022863),
    dict(param_name="ALKALINITY", UNITS_ID_in=5, VAL=8.080, station_id=3049211, LEV_M=20,
         salinity=29.9400, temperature=6.700000, LATITUDE=69.67500, LONGITUDE=60.33333,
         VAL_CONVERTED_NO_DENSITY=8.080 * 1000 / 1.025,
         VAL_CONVERTED_DENSITY=8.080 * 1000 / 1.020383),
    dict(param_name="ALKALINITY", UNITS_ID_in=5, VAL=32.000, station_id=3050147, LEV_M=75,
         salinity=34.6500, temperature=-0.940000, LATITUDE=78.66500, LONGITUDE=71.13333,
         VAL_CONVERTED_NO_DENSITY=32.000 * 1000 / 1.025,
         VAL_CONVERTED_DENSITY=32.000 * 1000 / 1.023957),

    # --- AMMONIUM 4->14: ug/L -> ug/kg, VAL / density (App. C: lab_dens) ---
    dict(param_name="AMMONIUM", UNITS_ID_in=4, VAL=0.411635, station_id=2000519, LEV_M=28.9506,
         salinity=34.271000, temperature=-1.731350, LATITUDE=82.16267, LONGITUDE=94.81550,
         VAL_CONVERTED_NO_DENSITY=0.411635 / 1.025,
         VAL_CONVERTED_DENSITY=0.411635 / 1.023669),
    dict(param_name="AMMONIUM", UNITS_ID_in=4, VAL=0.207408, station_id=2000520, LEV_M=3.6478,
         salinity=33.0145, temperature=-0.554325, LATITUDE=81.80217, LONGITUDE=94.29367,
         VAL_CONVERTED_NO_DENSITY=0.207408 / 1.025,
         VAL_CONVERTED_DENSITY=0.207408 / 1.022715),
    dict(param_name="AMMONIUM", UNITS_ID_in=4, VAL=5.509733, station_id=2000531, LEV_M=99.2243,
         salinity=34.5550, temperature=-0.791340, LATITUDE=77.28983, LONGITUDE=125.81100,
         VAL_CONVERTED_NO_DENSITY=5.509733 / 1.025,
         VAL_CONVERTED_DENSITY=5.509733 / 1.023886),

    # --- CHLOROPHYLL 4->14: ug/L -> ug/kg, VAL / density (App. C: real_dens) ---
    dict(param_name="CHLOROPHYLL", UNITS_ID_in=4, VAL=0.215819, station_id=2000403, LEV_M=50.418,
         salinity=34.584922, temperature=-0.571510, LATITUDE=80.98483, LONGITUDE=72.92750,
         VAL_CONVERTED_NO_DENSITY=0.215819 / 1.025,
         VAL_CONVERTED_DENSITY=0.215819 / 1.027801),
    dict(param_name="CHLOROPHYLL", UNITS_ID_in=4, VAL=3.841, station_id=2000326, LEV_M=5,
         salinity=32.471, temperature=-1.620000, LATITUDE=82.25600, LONGITUDE=60.09000,
         VAL_CONVERTED_NO_DENSITY=3.841 / 1.025,
         VAL_CONVERTED_DENSITY=3.841 / 1.026126),
    dict(param_name="CHLOROPHYLL", UNITS_ID_in=4, VAL=43.680, station_id=32035816, LEV_M=2,
         salinity=30.1350, temperature=6.720000, LATITUDE=34.55000, LONGITUDE=135.28000,
         VAL_CONVERTED_NO_DENSITY=43.680 / 1.025,
         VAL_CONVERTED_DENSITY=43.680 / 1.023626),

    # --- DIC 7->3: mmol/L -> umol/kg, VAL * 1000 / density (App. C: lab_dens) ---
    dict(param_name="DIC", UNITS_ID_in=7, VAL=1.966, station_id=33133561, LEV_M=0,
         salinity=34.3033, temperature=27.800000, LATITUDE=12.10000, LONGITUDE=154.90000,
         VAL_CONVERTED_NO_DENSITY=1.966 * 1000 / 1.025,
         VAL_CONVERTED_DENSITY=1.966 * 1000 / 1.023693),
    dict(param_name="DIC", UNITS_ID_in=7, VAL=2.2903, station_id=32929404, LEV_M=173.5,
         salinity=34.6240, temperature=-1.878000, LATITUDE=-76.50097, LONGITUDE=177.96730,
         VAL_CONVERTED_NO_DENSITY=2.2903 * 1000 / 1.025,
         VAL_CONVERTED_DENSITY=2.2903 * 1000 / 1.023939),

    # --- DIN 4->14: ug/L -> ug/kg, VAL / density (App. C: lab_dens)
    # Rows 1 and 2 average two duplicate-cast readings at the same (ID, LEV_M) ---
    dict(param_name="DIN", UNITS_ID_in=4, VAL=0.463373, station_id=2000519, LEV_M=2.9203,
         salinity=32.1265, temperature=-1.501450, LATITUDE=82.16267, LONGITUDE=94.81550,
         VAL_CONVERTED_NO_DENSITY=0.463373 / 1.025,
         VAL_CONVERTED_DENSITY=0.463373 / 1.022041),
    dict(param_name="DIN", UNITS_ID_in=4, VAL=0.943360, station_id=2000524, LEV_M=10.1239,
         salinity=33.1130, temperature=-1.310550, LATITUDE=82.09733, LONGITUDE=94.78450,
         VAL_CONVERTED_NO_DENSITY=0.943360 / 1.025,
         VAL_CONVERTED_DENSITY=0.943360 / 1.022790),
    dict(param_name="DIN", UNITS_ID_in=4, VAL=9.666824, station_id=2000529, LEV_M=100.214,
         salinity=34.5440, temperature=-1.307950, LATITUDE=77.12217, LONGITUDE=125.79833,
         VAL_CONVERTED_NO_DENSITY=9.666824 / 1.025,
         VAL_CONVERTED_DENSITY=9.666824 / 1.023877),

    # --- SF6 12->19: nmol/kg -> fmol/kg, VAL * 1e6 (mass-based, no density involved)
    # SYNTHETIC: There are no P_SF6 rows with UNITS_ID=12 in the database ---
    dict(param_name="SF6", UNITS_ID_in=12, VAL=0.028420, station_id=None, LEV_M=None,
         VAL_CONVERTED_NO_DENSITY=0.028420 * 1_000_000,
         VAL_CONVERTED_DENSITY=0.028420 * 1_000_000),

    # --- OXYGEN 21->3: mL/L -> umol/kg, VAL * 44.661 / density (App. C: real_dens) ---
    dict(param_name="OXYGEN", UNITS_ID_in=21, VAL=6.793, station_id=2000188, LEV_M=0,
         salinity=29.7800, temperature=1.608000, LATITUDE=76.73800, LONGITUDE=125.90500,
         VAL_CONVERTED_NO_DENSITY=6.793 * 44.661 / 1.025,
         VAL_CONVERTED_DENSITY=6.793 * 44.661 / 1.023820),
    dict(param_name="OXYGEN", UNITS_ID_in=21, VAL=6.523, station_id=2000190, LEV_M=54.4164,
         salinity=34.0970, temperature=-1.327000, LATITUDE=77.16600, LONGITUDE=125.98900,
         VAL_CONVERTED_NO_DENSITY=6.523 * 44.661 / 1.025,
         VAL_CONVERTED_DENSITY=6.523 * 44.661 / 1.027437),
    dict(param_name="OXYGEN", UNITS_ID_in=21, VAL=6.199, station_id=2000208, LEV_M=1096.317,
         salinity=34.8830, temperature=-0.235000, LATITUDE=79.93100, LONGITUDE=142.24500,
         VAL_CONVERTED_NO_DENSITY=6.199 * 44.661 / 1.025,
         VAL_CONVERTED_DENSITY=6.199 * 44.661 / 1.028028),

    # --- 14->3: ug/kg -> umol/kg, VAL / molar_mass (no density term at all)
    # P_NITRATENITRITE is SYNTHETIC: There are no P_NITRATENITRITE rows with UNITS_ID=14 in the database ---
    dict(param_name="P_BARIUM", UNITS_ID_in=14, VAL=39.56, station_id=2000403, LEV_M=9.9441,
         VAL_CONVERTED_NO_DENSITY=39.56 / 137.327, VAL_CONVERTED_DENSITY=39.56 / 137.327),
    dict(param_name="P_NITRATE", UNITS_ID_in=14, VAL=0.001431, station_id=2000403, LEV_M=9.9441,
         VAL_CONVERTED_NO_DENSITY=0.001431 / 14.00672, VAL_CONVERTED_DENSITY=0.001431 / 14.00672),
    dict(param_name="P_NITRATENITRITE", UNITS_ID_in=14, VAL=7.00336, station_id=None, LEV_M=None,
         VAL_CONVERTED_NO_DENSITY=7.00336 / 14.00672, VAL_CONVERTED_DENSITY=7.00336 / 14.00672),
    dict(param_name="P_NITRITE", UNITS_ID_in=14, VAL=0.018164, station_id=2000403, LEV_M=9.9441,
         VAL_CONVERTED_NO_DENSITY=0.018164 / 14.00672, VAL_CONVERTED_DENSITY=0.018164 / 14.00672),
    dict(param_name="P_PHOSPHATE", UNITS_ID_in=14, VAL=0.225507, station_id=2000403, LEV_M=9.9441,
         VAL_CONVERTED_NO_DENSITY=0.225507 / 30.973762, VAL_CONVERTED_DENSITY=0.225507 / 30.973762),
    dict(param_name="P_SILICATE", UNITS_ID_in=14, VAL=0.679448, station_id=2000403, LEV_M=9.9441,
         VAL_CONVERTED_NO_DENSITY=0.679448 / 28.085530, VAL_CONVERTED_DENSITY=0.679448 / 28.085530),

    # --- 4->3: ug/L -> umol/kg, VAL / (molar_mass * density) (App. C: lab_dens) ---
    dict(param_name="P_NITRATE", UNITS_ID_in=4, VAL=2.186492, station_id=2000519, LEV_M=28.9506,
         salinity=34.271, LATITUDE=82.16267, LONGITUDE=94.81550,
         VAL_CONVERTED_NO_DENSITY=2.186492 / (14.00672 * 1.025),
         VAL_CONVERTED_DENSITY=2.186492 / (14.00672 * 1.023669)),
    dict(param_name="P_NITRATENITRITE", UNITS_ID_in=4, VAL=2.238267, station_id=2000519, LEV_M=28.9506,
         salinity=34.271, LATITUDE=82.16267, LONGITUDE=94.81550,
         VAL_CONVERTED_NO_DENSITY=2.238267 / (14.00672 * 1.025),
         VAL_CONVERTED_DENSITY=2.238267 / (14.00672 * 1.023669)),
    dict(param_name="P_NITRITE", UNITS_ID_in=4, VAL=0.051775, station_id=2000519, LEV_M=28.9506,
         salinity=34.271, LATITUDE=82.16267, LONGITUDE=94.81550,
         VAL_CONVERTED_NO_DENSITY=0.051775 / (14.00672 * 1.025),
         VAL_CONVERTED_DENSITY=0.051775 / (14.00672 * 1.023669)),
    dict(param_name="P_PHOSPHATE", UNITS_ID_in=4, VAL=0.206446, station_id=2000519, LEV_M=28.9506,
         salinity=34.271, LATITUDE=82.16267, LONGITUDE=94.81550,
         VAL_CONVERTED_NO_DENSITY=0.206446 / (30.973762 * 1.025),
         VAL_CONVERTED_DENSITY=0.206446 / (30.973762 * 1.023669)),
    dict(param_name="P_SILICATE", UNITS_ID_in=4, VAL=0.971907, station_id=2000519, LEV_M=28.9506,
         salinity=34.271, LATITUDE=82.16267, LONGITUDE=94.81550,
         VAL_CONVERTED_NO_DENSITY=0.971907 / (28.085530 * 1.025),
         VAL_CONVERTED_DENSITY=0.971907 / (28.085530 * 1.023669)),

    # --- 26->3: ugAtom/kg -> umol/kg, 1:1 equivalence ---
    dict(param_name="P_NITRATE", UNITS_ID_in=26, VAL=4.817830, station_id=2000318, LEV_M=50,
         VAL_CONVERTED_NO_DENSITY=4.817830, VAL_CONVERTED_DENSITY=4.817830),
    dict(param_name="P_NITRITE", UNITS_ID_in=26, VAL=0.087597, station_id=2000318, LEV_M=50,
         VAL_CONVERTED_NO_DENSITY=0.087597, VAL_CONVERTED_DENSITY=0.087597),
    dict(param_name="P_PHOSPHATE", UNITS_ID_in=26, VAL=0.516686, station_id=2000318, LEV_M=50,
         VAL_CONVERTED_NO_DENSITY=0.516686, VAL_CONVERTED_DENSITY=0.516686),
    dict(param_name="P_SILICATE", UNITS_ID_in=26, VAL=2.089831, station_id=2000318, LEV_M=50,
         VAL_CONVERTED_NO_DENSITY=2.089831, VAL_CONVERTED_DENSITY=2.089831),

    # --- OXYGEN 10->3: percent saturation -> umol/kg, VAL * gsw.O2sol_SP_pt(S, pt0) / 100
    dict(param_name="OXYGEN", UNITS_ID_in=10, VAL=94.0, station_id=3048001, LEV_M=0,
         salinity=34.05, temperature=-1.60, LATITUDE=81.85000, LONGITUDE=57.75000, LEV_DBAR=0.0,
         VAL_CONVERTED_NO_DENSITY=94.0 * 365.660594 / 100,   # pt0=-1.600000 (surface)
         VAL_CONVERTED_DENSITY=94.0 * 365.660594 / 100),
    dict(param_name="OXYGEN", UNITS_ID_in=10, VAL=87.0, station_id=3048002, LEV_M=100,
         salinity=34.69, temperature=-0.46, LATITUDE=81.96667, LONGITUDE=57.08333, LEV_DBAR=101.0997,
         VAL_CONVERTED_NO_DENSITY=87.0 * 352.994255 / 100,   # pt0=-0.463193
         VAL_CONVERTED_DENSITY=87.0 * 352.994255 / 100),
    dict(param_name="OXYGEN", UNITS_ID_in=10, VAL=91.8, station_id=3060515, LEV_M=395,
         salinity=34.84, temperature=6.29, LATITUDE=69.06667, LONGITUDE=17.33333, LEV_DBAR=399.4032,
         VAL_CONVERTED_NO_DENSITY=91.8 * 298.837173 / 100,   # pt0=6.254190
         VAL_CONVERTED_DENSITY=91.8 * 298.837173 / 100),
])


def _load_test_cases(param_name, units_id_in=None):
    """Select CONVERSION_REFERENCE rows for param_name (and UNITS_ID_in, if given)."""
    df = CONVERSION_REFERENCE[CONVERSION_REFERENCE["param_name"] == param_name]
    if units_id_in is not None:
        df = df[df["UNITS_ID_in"] == units_id_in]
    if df.empty:
        pytest.skip(f"no reference rows for {param_name} (UNITS_ID_in={units_id_in})")
    return df


def _assert_conversion(formulas, func, df_in, param_name, precision):
    """Run func with use_density False and True, check both against CONVERSION_REFERENCE."""
    df_out_f = func(df_in, param_name, False)
    df_out_t = func(df_in, param_name, True)
    diff_f = (df_out_f["VAL"] - df_in["VAL_CONVERTED_NO_DENSITY"]).abs()
    diff_t = (df_out_t["VAL"] - df_in["VAL_CONVERTED_DENSITY"]).abs()
    assert (diff_f <= precision).all(), f"No-density diff exceeded {precision}: {diff_f.max()}"
    assert (diff_t <= precision).all(), f"Density diff exceeded {precision}: {diff_t.max()}"


class TestConversionFormulasReference:
    """Compares each conversion formula against CONVERSION_REFERENCE, hand-verified
    against the COMFORT dataset v3 report Appendix C."""

    @pytest.fixture
    def formulas(self):
        return ConversionFormulas()

    # ALKALINITY: milliequivalent/L -> µmol/kg, both density conventions
    def test_milliEquivalentPerLiter_micromolPerKilogram(self, formulas):
        df_in = _load_test_cases("ALKALINITY")
        _assert_conversion(formulas, formulas.milliEquivalentPerLiter_micromolPerKilogram,
                           df_in, "ALKALINITY", precision=0.01)

    # CHLOROPHYLL: µg/L -> µg/kg, both density conventions (App. C: real_dens)
    def test_microgramPerLiter_microgramPerKilogram_chlorophyll(self, formulas):
        df_in = _load_test_cases("CHLOROPHYLL")
        _assert_conversion(formulas, formulas.microgramPerLiter_microgramPerKilogram,
                           df_in, "CHLOROPHYLL", precision=1e-3)

    # AMMONIUM, DIN: µg/L -> µg/kg, both density conventions (App. C: lab_dens)
    @pytest.mark.parametrize("param_name", ["AMMONIUM", "DIN"])
    def test_microgramPerLiter_microgramPerKilogram_nutrients(self, formulas, param_name):
        df_in = _load_test_cases(param_name)
        _assert_conversion(formulas, formulas.microgramPerLiter_microgramPerKilogram,
                           df_in, param_name, precision=1e-3)

    # DIC: millimol/L -> µmol/kg, constant-density path
    def test_millimolPerLiter_micromolPerKilogram(self, formulas):
        df_in = _load_test_cases("DIC")
        df_out = formulas.millimolPerLiter_micromolPerKilogram(df_in, "DIC", False)
        diff = (df_out["VAL"] - df_in["VAL_CONVERTED_NO_DENSITY"]).abs()
        assert (diff <= 1e-3).all(), f"No-density diff exceeded 1e-3: {diff.max()}"

    # DIC: millimol/L -> µmol/kg, in-situ-density path
    def test_millimolPerLiter_micromolPerKilogram_density(self, formulas):
        df_in = _load_test_cases("DIC")
        df_out = formulas.millimolPerLiter_micromolPerKilogram(df_in, "DIC", True)
        diff = (df_out["VAL"] - df_in["VAL_CONVERTED_DENSITY"]).abs()
        assert (diff <= 1e-3).all(), f"Density diff exceeded 1e-3: {diff.max()}"

    # SF6: nanomol/kg -> femtomol/kg, both density conventions
    def test_nanomolPerKilogram_femtomolPerKilogram(self, formulas):
        df_in = _load_test_cases("SF6", units_id_in=12)
        _assert_conversion(formulas, formulas.nanomolPerKilogram_femtomolPerKilogram,
                           df_in, "SF6", precision=0.005)

    # OXYGEN: mL/L -> µmol/kg, both density conventions
    def test_milliliterPerLiter_micromolPerKilogram(self, formulas):
        df_in = _load_test_cases("OXYGEN", units_id_in=21)
        _assert_conversion(formulas, formulas.milliliterPerLiter_micromolPerKilogram,
                           df_in, "OXYGEN", precision=0.05)

    # OXYGEN: percent saturation -> µmol/kg (density-independent)
    def test_percent_micromolPerKilogram(self, formulas):
        df_in = _load_test_cases("OXYGEN", units_id_in=10)
        _assert_conversion(formulas, formulas.percent_micromolPerKilogram,
                           df_in, "OXYGEN", precision=0.01)

    # BARIUM, NITRATE, NITRATENITRITE, NITRITE, PHOSPHATE, SILICATE (ug/kg -> umol/kg, molar-mass only, no density)
    @pytest.mark.parametrize("param_name", ["P_BARIUM", "P_NITRATE", "P_NITRATENITRITE", "P_NITRITE", "P_PHOSPHATE", "P_SILICATE"])
    def test_microgramPerKilogram_micromolPerKilogram(self, formulas, param_name):
        df_in = _load_test_cases(param_name, units_id_in=14)
        _assert_conversion(formulas, formulas.microgramPerKilogram_micromolPerKilogram,
                           df_in, param_name, precision=1e-6)

    # NITRATE, NITRATENITRITE, NITRITE, PHOSPHATE, SILICATE (ug/L -> umol/kg, molar mass + lab_dens - App. C nutrients)
    @pytest.mark.parametrize("param_name", ["P_NITRATE", "P_NITRATENITRITE", "P_NITRITE", "P_PHOSPHATE", "P_SILICATE"])
    def test_microgramPerLiter_micromolPerKilogram(self, formulas, param_name):
        df_in = _load_test_cases(param_name, units_id_in=4)
        _assert_conversion(formulas, formulas.microgramPerLiter_micromolPerKilogram,
                           df_in, param_name, precision=1e-3)

    # NITRATE, NITRITE, PHOSPHATE, SILICATE (ugAtom/kg -> umol/kg, 1:1 equivalence)
    @pytest.mark.parametrize("param_name", ["P_NITRATE", "P_NITRITE", "P_PHOSPHATE", "P_SILICATE"])
    def test_microgramAtomPerKilogram_micromolPerKilogram(self, formulas, param_name):
        df_in = _load_test_cases(param_name, units_id_in=26)
        _assert_conversion(formulas, formulas.microgramAtomPerKilogram_micromolPerKilogram,
                           df_in, param_name, precision=1e-9)
