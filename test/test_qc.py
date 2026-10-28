"""Tests for comfort.qc."""
import logging
import sqlite3
import pandas as pd
import pytest

from comfort.qc import (QC_ALL, QC_GOOD, QCFilter, apply_qc_flags,
                        build_where_clause, flag_salinity_like_oxygen, qc_conditions)


class TestBuildWhereClause:
    # No flags means no WHERE clause at all
    def test_empty_flags_returns_empty(self):
        assert build_where_clause([]) == ""

    # None is treated the same as an empty list
    def test_none_flags_returns_empty(self):
        assert build_where_clause(None) == ""

    # A single filter renders as a plain WHERE
    def test_single_flag(self):
        flags = [QCFilter("PQF1", ">0")]
        assert build_where_clause(flags) == "WHERE PQF1>0"

    # Multiple filters are AND-joined
    def test_multiple_flags(self):
        result = build_where_clause(QC_GOOD)
        assert "WHERE" in result
        assert "PQF1>0" in result
        assert "PQF2>2" in result
        assert "AND" in result

    # table_alias prefixes every column
    def test_table_alias(self):
        flags = [QCFilter("PQF1", ">0")]
        result = build_where_clause(flags, table_alias="t")
        assert result == "WHERE t.PQF1>0"

    # keyword="AND" is used for appending onto an existing WHERE clause
    def test_keyword_and(self):
        flags = [QCFilter("PQF1", ">0")]
        result = build_where_clause(flags, keyword="AND")
        assert result == "AND PQF1>0"

    # table_alias and keyword combine for a multi-table join filter
    def test_alias_and_keyword_combined(self):
        flags = [QCFilter("PQF1", ">0"), QCFilter("PQF2", ">2")]
        result = build_where_clause(flags, table_alias="q", keyword="AND")
        assert result.startswith("AND ")
        assert "q.PQF1>0" in result
        assert "q.PQF2>2" in result

    # Plain tuples work the same as QCFilter tuples
    def test_tuple_input(self):
        flags = [("PQF1", ">0"), ("PQF2", ">2")]
        result = build_where_clause(flags)
        assert "PQF1>0" in result
        assert "PQF2>2" in result

    # QC_ALL is the no-filtering preset
    def test_qc_all_is_empty(self):
        assert build_where_clause(QC_ALL) == ""

    # A condition carrying extra SQL is rejected
    def test_injection_in_condition_raises(self):
        with pytest.raises(ValueError, match="Invalid quality flag condition"):
            build_where_clause([("PQF1", ">0 OR 1=1")])

    # An unsafe column name is rejected
    def test_injection_in_column_raises(self):
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            build_where_clause([("PQF1; DROP TABLE station", ">0")])

    # An unsafe table alias is rejected
    def test_invalid_alias_raises(self):
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            build_where_clause(QC_GOOD, table_alias="p; --")

    # Unsupported operators are rejected on the SQL side too
    @pytest.mark.parametrize("condition", ["=1", "<>1", "IN (1,2)", ">", "1", ">0.", ""])
    def test_invalid_condition_raises(self, condition):
        with pytest.raises(ValueError, match="Invalid quality flag condition"):
            build_where_clause([("PQF1", condition)])

    # Whitespace in the condition is normalised away in the generated SQL
    def test_whitespace_normalised(self):
        assert build_where_clause([("PQF1", " >= 2 ")]) == "WHERE PQF1>=2"

    # The SQL and the DataFrame path select the same rows for the same flags
    @pytest.mark.parametrize("flags", [QC_GOOD, [("PQF1", "!=0")], [("PQF2", "<=3"), ("PQF1", "==1")]])
    def test_sql_and_pandas_agree(self, flags):
        df = pd.DataFrame({"PQF1": [0, 1, 1, 1], "PQF2": [0, 1, 3, 5], "VAL": [10.0, 20.0, 30.0, 40.0]})
        conn = sqlite3.connect(":memory:")
        df.to_sql("t", conn, index=False)
        rows = conn.execute("SELECT VAL FROM t " + build_where_clause(flags) + " ORDER BY VAL").fetchall()
        conn.close()
        assert [r[0] for r in rows] == apply_qc_flags(df, flags)["VAL"].tolist()


class TestQcConditions:
    # No flags yield no conditions
    @pytest.mark.parametrize("flags", [None, [], QC_ALL])
    def test_empty(self, flags):
        assert qc_conditions(flags) == []

    # One condition per flag, in order
    def test_list_of_conditions(self):
        assert qc_conditions(QC_GOOD) == ["PQF1>0", "PQF2>2"]

    # The alias prefixes every column
    def test_alias(self):
        assert qc_conditions(QC_GOOD, "p") == ["p.PQF1>0", "p.PQF2>2"]


class TestApplyQcFlags:
    @staticmethod
    def _df():
        # Test df
        return pd.DataFrame({
            "PQF1": [0, 1, 1, 1],
            "PQF2": [0, 1, 3, 5],
            "VAL": [10.0, 20.0, 30.0, 40.0],
        })

    # quality_flags=None passes every row through
    def test_none_returns_copy(self):
        df = self._df()
        result = apply_qc_flags(df, None)
        assert len(result) == 4
        assert result is not df

    # An empty filter list also passes every row through
    def test_empty_returns_copy(self):
        df = self._df()
        result = apply_qc_flags(df, [])
        assert len(result) == 4

    # ">" operator string is parsed and applied
    def test_gt_operator(self):
        result = apply_qc_flags(self._df(), [QCFilter("PQF1", ">0")])
        assert (result["PQF1"] > 0).all()
        assert len(result) == 3

    # ">=" operator string is parsed and applied
    def test_ge_operator(self):
        result = apply_qc_flags(self._df(), [QCFilter("PQF2", ">=3")])
        assert (result["PQF2"] >= 3).all()
        assert len(result) == 2

    # "<" operator string is parsed and applied
    def test_lt_operator(self):
        result = apply_qc_flags(self._df(), [QCFilter("PQF2", "<3")])
        assert (result["PQF2"] < 3).all()

    # "<=" operator string is parsed and applied
    def test_le_operator(self):
        result = apply_qc_flags(self._df(), [QCFilter("PQF2", "<=3")])
        assert (result["PQF2"] <= 3).all()

    # "==" operator string is parsed and applied
    def test_eq_operator(self):
        result = apply_qc_flags(self._df(), [QCFilter("PQF1", "==1")])
        assert (result["PQF1"] == 1).all()

    # "!=" operator string is parsed and applied
    def test_ne_operator(self):
        result = apply_qc_flags(self._df(), [QCFilter("PQF1", "!=0")])
        assert (result["PQF1"] != 0).all()

    # The QC_GOOD preset combines both PQF1 and PQF2 thresholds
    def test_qc_good_preset(self):
        # QC_GOOD means PQF1 > 0 AND PQF2 > 2
        result = apply_qc_flags(self._df(), QC_GOOD)
        assert len(result) == 2
        assert (result["PQF1"] > 0).all()
        assert (result["PQF2"] > 2).all()

    # A filter on a column absent from df raises
    def test_missing_column_raises(self):
        with pytest.raises(KeyError, match="SQF"):
            apply_qc_flags(self._df(), [QCFilter("SQF", ">0")])

    # An unsupported operator raises instead of silently filtering nothing
    def test_invalid_operator_raises(self):
        with pytest.raises(ValueError, match="Invalid quality flag condition"):
            apply_qc_flags(self._df(), [QCFilter("PQF1", "=1")])

    # A non-numeric threshold raises a clear error
    def test_invalid_number_raises(self):
        with pytest.raises(ValueError, match="Invalid quality flag condition"):
            apply_qc_flags(self._df(), [QCFilter("PQF1", ">abc")])

    # Whitespace around the operator and number is tolerated, as in SQL
    def test_whitespace_tolerated(self):
        result = apply_qc_flags(self._df(), [QCFilter("PQF1", " > 0 ")])
        assert len(result) == 3

    # Negative thresholds are parsed
    def test_negative_threshold(self):
        df = pd.DataFrame({"PQF1": [-2, -1, 0]})
        assert len(apply_qc_flags(df, [QCFilter("PQF1", ">=-1")])) == 2

    # The returned DataFrame is independent of the input
    def test_returns_copy(self):
        df = self._df()
        result = apply_qc_flags(df, QC_GOOD)
        result["VAL"] = 999.0
        assert df["VAL"].iloc[0] != 999.0

    # Plain tuples work the same as QCFilter tuples
    def test_tuple_input(self):
        result = apply_qc_flags(self._df(), [("PQF1", ">0")])
        assert (result["PQF1"] > 0).all()


class TestFlagSalinityLikeOxygen:
    def _df(self):
        """Two genuine oxygen rows and one salinity value mislabelled as oxygen."""
        sp = 35.0
        return pd.DataFrame({
            "OXYGEN": [250.0, 180.0, sp],
            "SALINITY": [sp, 34.5, sp],
        })

    # Only the row where OXYGEN numerically matches SALINITY is flagged
    def test_flags_mislabelled_row_only(self, caplog):
        with caplog.at_level(logging.WARNING):
            mask = flag_salinity_like_oxygen(self._df())
        assert list(mask) == [False, False, True]
        assert "mislabelled salinity" in caplog.text

    # Only exact copies are flagged by default
    def test_float_noise_needs_tolerance(self):
        df = self._df()
        df.loc[2, "OXYGEN"] += 1e-9
        assert not flag_salinity_like_oxygen(df).any()
        assert list(flag_salinity_like_oxygen(df, tol=1e-6)) == [False, False, True]

    # A value that is merely close is not flagged
    def test_close_but_not_identical_not_flagged(self):
        df = self._df()
        df.loc[2, "OXYGEN"] = 35.0 * 35.16504 / 35
        assert not flag_salinity_like_oxygen(df).any()

    # Column detection is case-insensitive
    def test_lowercase_salinity_column(self):
        df = self._df().rename(columns={"SALINITY": "salinity"})
        mask = flag_salinity_like_oxygen(df)
        assert list(mask) == [False, False, True]

    # salinity_col lets the caller point at a non-standard column name
    def test_explicit_salinity_col(self):
        df = self._df().rename(columns={"SALINITY": "PSAL"})
        mask = flag_salinity_like_oxygen(df, salinity_col="PSAL")
        assert list(mask) == [False, False, True]

    # A NaN salinity value can never match, so the row is not flagged
    def test_nan_rows_not_flagged(self):
        df = self._df()
        df.loc[2, "SALINITY"] = float("nan")
        mask = flag_salinity_like_oxygen(df)
        assert not mask.any()

    # OXYGEN is a required column
    def test_missing_oxygen_column_raises(self):
        with pytest.raises(KeyError, match="OXYGEN"):
            flag_salinity_like_oxygen(pd.DataFrame({"SALINITY": [35.0]}))

    # A salinity column is required
    def test_missing_salinity_column_raises(self):
        with pytest.raises(KeyError, match="SALINITY"):
            flag_salinity_like_oxygen(pd.DataFrame({"OXYGEN": [250.0]}))

    # tol widens the matching range: a 0.0005 difference is only flagged with a larger tol
    def test_custom_tolerance(self):
        df = self._df()
        df.loc[2, "OXYGEN"] += 0.0005
        assert not flag_salinity_like_oxygen(df).any()
        assert list(flag_salinity_like_oxygen(df, tol=1e-3)) == [False, False, True]

    # Negative tolerance raises
    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError, match="tol"):
            flag_salinity_like_oxygen(self._df(), tol=-1.0)


class TestColumnCasing:
    # A flag column in another casing is found
    def test_apply_qc_flags_other_casing(self):
        df = pd.DataFrame({"PQF1": [0, 1, 1], "VAL": [10.0, 20.0, 30.0]})
        result = apply_qc_flags(df, [QCFilter("pqf1", ">0")])
        assert result["VAL"].tolist() == [20.0, 30.0]
