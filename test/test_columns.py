"""Tests for comfort.util.columns."""
import pandas as pd
import pytest

from comfort.util.columns import find_col, resolve_param_columns


class TestFindCol:
    # An exact match is returned as it is
    def test_exact(self):
        assert find_col(pd.DataFrame({"VAL": [1]}), "VAL") == "VAL"

    # Another casing finds the column under its own spelling
    @pytest.mark.parametrize("spelling", ["val", "Val", "vAl"])
    def test_other_casing(self, spelling):
        assert find_col(pd.DataFrame({"VAL": [1]}), spelling) == "VAL"

    # An exact match wins over a case-variant of the same name
    def test_exact_wins_over_variant(self):
        df = pd.DataFrame({"Val": [1], "VAL": [2]})
        assert find_col(df, "VAL") == "VAL"
        assert find_col(df, "Val") == "Val"

    # Several variants and no exact match is ambiguous
    def test_ambiguous_raises(self):
        df = pd.DataFrame({"Val": [1], "VAL": [2]})
        with pytest.raises(KeyError, match="ambiguous"):
            find_col(df, "val")

    # A missing column raises KeyError and lists the available columns
    def test_missing_raises(self):
        with pytest.raises(KeyError, match="Available columns"):
            find_col(pd.DataFrame({"VAL": [1]}), "OXYGEN")

    # required=False returns None instead
    def test_missing_not_required(self):
        assert find_col(pd.DataFrame({"VAL": [1]}), "OXYGEN", required=False) is None

    # None passes through, so optional column arguments can be resolved unconditionally
    def test_none_passes_through(self):
        assert find_col(pd.DataFrame({"VAL": [1]}), None) is None

    # A list is resolved element by element
    def test_list(self):
        df = pd.DataFrame({"ID": [1], "PROFILE_NUMBER": [1]})
        assert find_col(df, ["id", "profile_number"]) == ["ID", "PROFILE_NUMBER"]


class TestResolveParamColumns:
    df = pd.DataFrame({"P_NITRATE": [1.0], "OXYGEN": [2.0]})

    # Any spelling finds the P_* column or the bare column
    @pytest.mark.parametrize("spelling", ["NITRATE", "nitrate", "P_NITRATE", "p_nitrate"])
    def test_spellings_find_prefixed_column(self, spelling):
        assert resolve_param_columns(self.df, spelling) == ["P_NITRATE"]

    @pytest.mark.parametrize("spelling", ["OXYGEN", "oxygen", "P_OXYGEN", "p_oxygen"])
    def test_spellings_find_bare_column(self, spelling):
        assert resolve_param_columns(self.df, [spelling]) == ["OXYGEN"]

    # The order of the request is kept
    def test_order(self):
        assert resolve_param_columns(self.df, ["oxygen", "nitrate"]) == ["OXYGEN", "P_NITRATE"]

    # Missing column raises
    def test_missing_raises(self):
        with pytest.raises(KeyError, match="SALINITY"):
            resolve_param_columns(self.df, ["SALINITY"])
