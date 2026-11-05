import pytest

import datetime

import pandas as pd

from comfort.util.types import GridMode, InterpMethod, LoadOutput, check_choice, check_timestamp, parse_timestamps


class TestCheckChoice:
    # A valid option passes silently
    def test_valid_value_passes(self):
        check_choice("dict", LoadOutput, "output")

    # The error names the option, the allowed values and the offending value
    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match=r"output must be one of \('xarray', 'dict'\), got 'netcdf'"):
            check_choice("netcdf", LoadOutput, "output")

    # Casing matters
    def test_case_sensitive(self):
        with pytest.raises(ValueError, match="mode"):
            check_choice("y", GridMode, "mode")

    # Every alias is a usable Literal
    def test_interp_method_options(self):
        check_choice("pchip", InterpMethod, "method")
        check_choice("linear", InterpMethod, "method")


class TestCheckTimestamp:
    # The long format and datetime objects pass
    @pytest.mark.parametrize("value", ["2020-12-31 23:59:59",
                                       datetime.datetime(2020, 12, 31), pd.Timestamp("2020-12-31")])
    def test_explicit_instants_pass(self, value):
        check_timestamp(value, "date_max")

    # Date-only strings, partial times and plain dates are rejected
    @pytest.mark.parametrize("value", ["2020-12-31", "2020-12", "2020-12-31 23:59", "31.12.2020 23:59:59",
                                       datetime.date(2020, 12, 31)])
    def test_other_values_raise(self, value):
        with pytest.raises(ValueError, match="date_max must be given as 'YYYY-MM-DD HH:MM:SS'"):
            check_timestamp(value, "date_max")


class TestParseTimestamps:
    # Strings with and without fractional seconds in one column are parsed alike (pd.to_datetime raises or gives NaT),
    def test_mixed_fractional_seconds_are_truncated(self):
        result = parse_timestamps(pd.Series(["2009-10-09 04:02:02.000970", "2009-10-09 04:02:02", "2009-10-09 04:02:02.9"]))
        assert list(result) == [pd.Timestamp("2009-10-09 04:02:02")] * 3

    # Date-only strings are valid ISO 8601
    def test_date_only(self):
        assert list(parse_timestamps(pd.Series(["2000-01-01"]))) == [pd.Timestamp("2000-01-01")]

    # A value that is no date raises instead of becoming NaT
    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            parse_timestamps(pd.Series(["2000-01-01 00:00:00", "not a date"]))
