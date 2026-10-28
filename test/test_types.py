import pytest

from comfort.util.types import GridMode, InterpMethod, LoadOutput, check_choice


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
