"""Tests for comfort.database.information."""
import sqlite3
import pandas as pd
import pytest

from comfort.database.information import (
    count_negative_samples,
    count_samples_over_time,
    count_samples_per_parameter,
    find_spatiotemporal_duplicates,
    get_min_max_dates,
    get_minmax,
    get_names_of_all_parameter_tables,
    get_num_samples,
)
from comfort.qc import QC_GOOD


@pytest.fixture
def comfort_conn():
    """In-memory COMFORT-like database with station, P_NITRATE and P_OXYGEN."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE station ("
        "  ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)"
    )
    conn.executemany("INSERT INTO station VALUES (?,?,?,?)", [
        (1, 45.0, -30.0, "2000-01-15"),
        (2, 55.0, -20.0, "2005-06-10"),
        (3, 60.0, -10.0, "1998-11-02"),
    ])

    _param_ddl = (
        "CREATE TABLE {tbl} ("
        "  ID INTEGER, LEV_DBAR REAL, LEV_M REAL, VAL REAL,"
        "  PQF1 INTEGER, PQF2 INTEGER, SQF INTEGER,"
        "  BOTTLE_NUMBER INTEGER, PROFILE_NUMBER INTEGER,"
        "  PROFILE_BEST INTEGER, UNITS_ID INTEGER, INSTRUMENT_ID INTEGER)"
    )
    for tbl in ("P_NITRATE", "P_OXYGEN"):
        conn.execute(_param_ddl.format(tbl=tbl))

    conn.executemany("INSERT INTO P_NITRATE VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        (1, 0.0, 0.0, 5.0, 1, 3, 0, 1, 1, 1, 3, 1),
        (1, 100.0, 100.0, 4.0, 1, 3, 0, 2, 1, 1, 3, 1),
        (2, 0.0, 0.0, 8.0, 0, 1, 0, 1, 2, 1, 3, 2),  # fails QC_GOOD
    ])
    conn.executemany("INSERT INTO P_OXYGEN VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        (1, 0.0, 0.0, 200.0, 1, 3, 0, 1, 1, 1, 3, 1),
        (2, 0.0, 0.0, 220.0, 1, 3, 0, 1, 2, 1, 3, 2),
    ])
    conn.commit()
    yield conn
    conn.close()


class TestGetMinMaxDates:
    # Fetch min/max DATEANDTIME across all station rows
    def test_returns_min_and_max(self, comfort_conn):
        min_date, max_date = get_min_max_dates(comfort_conn, table_name="station")
        assert str(min_date.date()) == "1998-11-02"
        assert str(max_date.date()) == "2005-06-10"

    # Use a non-default time column
    def test_custom_time_column(self, comfort_conn):
        comfort_conn.execute("ALTER TABLE station ADD COLUMN VISIT_DATE TEXT")
        comfort_conn.execute("UPDATE station SET VISIT_DATE = DATEANDTIME")
        comfort_conn.commit()
        min_date, _ = get_min_max_dates(comfort_conn, table_name="station", time_column="VISIT_DATE")
        assert str(min_date.date()) == "1998-11-02"


class TestGetNumSamples:
    # Count every row when no quality flags are given
    def test_counts_all_rows_without_flags(self, comfort_conn):
        assert get_num_samples(comfort_conn, "P_NITRATE") == 3

    # Drop the row that fails QC_GOOD
    def test_applies_quality_flags(self, comfort_conn):
        assert get_num_samples(comfort_conn, "P_NITRATE", quality_flags=QC_GOOD) == 2

    # Return 0 and log an error for a table that does not exist
    def test_nonexistent_table_returns_zero(self, comfort_conn, caplog):
        import logging
        with caplog.at_level(logging.ERROR):
            result = get_num_samples(comfort_conn, "P_DOESNOTEXIST")
        assert result == 0
        assert "P_DOESNOTEXIST" in caplog.text


class TestGetNamesOfAllParameterTables:
    # Match both parameter tables with the default LIKE pattern
    def test_matches_parameter_tables(self, comfort_conn):
        names = get_names_of_all_parameter_tables(comfort_conn)
        assert set(names) == {"P_NITRATE", "P_OXYGEN"}

    # Station is not a P_* table, must not be included
    def test_excludes_non_matching_tables(self, comfort_conn):
        names = get_names_of_all_parameter_tables(comfort_conn)
        assert "station" not in names

    # Grid-mapped tables (numeric suffix) are excluded by default
    def test_excludes_digit_suffixed_tables_by_default(self, comfort_conn):
        comfort_conn.execute(
            "CREATE TABLE P_NITRATE_123 (ID INTEGER, LEV_M REAL, VAL REAL)"
        )
        comfort_conn.commit()
        names = get_names_of_all_parameter_tables(comfort_conn)
        assert "P_NITRATE_123" not in names

    # include_digits=True brings grid-mapped tables back
    def test_includes_digit_suffixed_tables_when_requested(self, comfort_conn):
        comfort_conn.execute(
            "CREATE TABLE P_NITRATE_123 (ID INTEGER, LEV_M REAL, VAL REAL)"
        )
        comfort_conn.commit()
        names = get_names_of_all_parameter_tables(comfort_conn, include_digits=True)
        assert "P_NITRATE_123" in names

    # Same lookup works against views, not just tables
    def test_matches_views(self, comfort_conn):
        comfort_conn.execute("CREATE VIEW E_NITRATE AS SELECT * FROM P_NITRATE")
        comfort_conn.commit()
        names = get_names_of_all_parameter_tables(
            comfort_conn, like_pattern="E|_%", escape_char="|", table_type="view"
        )
        assert names == ["E_NITRATE"]


class TestGetMinMax:
    # Global min/max of VAL across two explicit parameter tables
    def test_global_min_max_across_tables(self, comfort_conn):
        lo, hi = get_minmax(comfort_conn, column="VAL", column_type="float",
                            param_tables=["P_NITRATE", "P_OXYGEN"])
        assert lo == pytest.approx(4.0)
        assert hi == pytest.approx(220.0)

    # param_tables=None auto-discovers all parameter tables
    def test_default_param_tables_uses_all_parameter_tables(self, comfort_conn):
        lo, hi = get_minmax(comfort_conn, column="VAL", column_type="float")
        assert lo == pytest.approx(4.0)
        assert hi == pytest.approx(220.0)

    # column_type="int" converts the result to Python int
    def test_int_column_type(self, comfort_conn):
        lo, hi = get_minmax(comfort_conn, column="ID", column_type="int",
                            param_tables=["P_NITRATE"])
        assert (lo, hi) == (1, 2)

    # Unsupported column_type logs an error and returns (None, None)
    def test_unsupported_column_type_returns_none(self, comfort_conn, caplog):
        import logging
        with caplog.at_level(logging.ERROR):
            result = get_minmax(comfort_conn, column="VAL", column_type="bogus",
                                param_tables=["P_NITRATE"])
        assert result == (None, None)
        assert "bogus" in caplog.text


class TestCountSamplesOverTime:
    # Every year between first and last is reported, years without samples count 0
    def test_year_fills_empty_years(self, comfort_conn):
        result = count_samples_over_time(comfort_conn, "NITRATE", time_mode="year")
        assert list(result.columns) == ["YEAR", "COUNT"]
        assert list(result["YEAR"]) == [2000, 2001, 2002, 2003, 2004, 2005]
        assert list(result["COUNT"]) == [2, 0, 0, 0, 0, 1]

    # Months always run from 1 to 12
    def test_month(self, comfort_conn):
        result = count_samples_over_time(comfort_conn, "NITRATE", time_mode="month")
        assert list(result["MONTH"]) == list(range(1, 13))
        assert result.set_index("MONTH").loc[1, "COUNT"] == 2
        assert result.set_index("MONTH").loc[6, "COUNT"] == 1
        assert result["COUNT"].sum() == 3

    # The SQL counts equal the pandas counts of the same dates
    def test_matches_pandas(self, comfort_conn):
        df = pd.read_sql("SELECT s.DATEANDTIME FROM P_NITRATE p JOIN station s ON p.ID = s.ID", comfort_conn)
        expected = pd.to_datetime(df["DATEANDTIME"]).dt.year.value_counts().sort_index()
        got = count_samples_over_time(comfort_conn, "NITRATE").set_index("YEAR")["COUNT"]
        assert (got[got > 0] == expected).all()

    # Quality flags filter the samples before counting
    def test_quality_flags(self, comfort_conn):
        result = count_samples_over_time(comfort_conn, "NITRATE", quality_flags=QC_GOOD)
        assert list(result["YEAR"]) == [2000]
        assert list(result["COUNT"]) == [2]

    # Dates with fractional seconds are counted, rows without a valid date are ignored
    def test_date_formats(self, comfort_conn):
        comfort_conn.execute("UPDATE station SET DATEANDTIME='2001-02-03 04:05:06.789' WHERE ID=1")
        comfort_conn.execute("UPDATE station SET DATEANDTIME=NULL WHERE ID=2")
        comfort_conn.commit()
        result = count_samples_over_time(comfort_conn, "NITRATE")
        assert list(result["YEAR"]) == [2001]
        assert list(result["COUNT"]) == [2]

    # No samples -> No year range to count over
    def test_year_empty(self, comfort_conn):
        comfort_conn.execute("CREATE TABLE P_EMPTY (ID INTEGER, VAL REAL)")
        result = count_samples_over_time(comfort_conn, "EMPTY", time_mode="year")
        assert result.empty
        assert list(result.columns) == ["YEAR", "COUNT"]

    # An unknown time_mode is rejected
    def test_bad_time_mode(self, comfort_conn):
        with pytest.raises(ValueError, match="time_mode"):
            count_samples_over_time(comfort_conn, "NITRATE", time_mode="week")


class TestCountSamplesPerParameter:
    # Counts are reported per requested parameter
    def test_requested_parameters(self, comfort_conn):
        result = count_samples_per_parameter(comfort_conn, ["NITRATE", "OXYGEN"])
        assert dict(zip(result["parameter"], result["count"])) == {"NITRATE": 3, "OXYGEN": 2}

    # parameters=None discovers and counts every parameter table
    def test_default_all(self, comfort_conn):
        result = count_samples_per_parameter(comfort_conn)
        assert set(result["parameter"]) == {"NITRATE", "OXYGEN"}

    # Quality flags filter the counted samples
    def test_quality_flags(self, comfort_conn):
        result = count_samples_per_parameter(comfort_conn, ["NITRATE"], quality_flags=QC_GOOD)
        assert list(result["count"]) == [2]


class TestCountNegativeSamples:
    # Proportion of negative samples per parameter, sorted ascending
    def test_proportions(self, comfort_conn):
        comfort_conn.execute("INSERT INTO P_NITRATE VALUES (3,0.0,0.0,-1.0,1,3,0,1,3,1,3,1)")
        comfort_conn.commit()
        result = count_negative_samples(comfort_conn, ["NITRATE", "OXYGEN"])
        assert list(result["parameter"]) == ["OXYGEN", "NITRATE"]
        assert list(result["number of negative samples"]) == [0, 1]
        assert list(result["proportion of negative samples"]) == [0.0, 25.0]

    # No parameters requested: Empty df with the documented columns
    def test_no_parameters(self, comfort_conn):
        result = count_negative_samples(comfort_conn, [])
        assert result.empty
        assert list(result.columns) == [
            "parameter", "number of samples", "number of negative samples", "proportion of negative samples",
        ]


class TestFindSpatiotemporalDuplicates:
    # No duplicated (ID, LEV_M) pairs in the clean fixture
    def test_none_found(self, comfort_conn):
        summary, details = find_spatiotemporal_duplicates(comfort_conn, ["NITRATE"])
        assert summary.empty
        assert details == {}

    # A repeated (ID, LEV_M) pair is one time/location duplicate, but not a row duplicate
    def test_time_loc_duplicate(self, comfort_conn):
        comfort_conn.execute("INSERT INTO P_NITRATE VALUES (1,0.0,0.0,6.0,1,3,0,9,1,1,3,1)")
        comfort_conn.commit()
        summary, details = find_spatiotemporal_duplicates(comfort_conn, ["NITRATE"])
        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["number of samples"] == 4
        assert row["number of time loc duplicates"] == 1
        assert row["number of row duplicates"] == 0
        assert row["proportion of time loc duplicates"] == 25.0
        assert set(details["NITRATE"]) == {"time_loc", "rows"}
        assert len(details["NITRATE"]["time_loc"]) == 1

    # An exactly repeated row counts as a row duplicate
    def test_row_duplicate(self, comfort_conn):
        comfort_conn.execute("INSERT INTO P_NITRATE VALUES (1,0.0,0.0,5.0,1,3,0,1,1,1,3,1)")
        comfort_conn.commit()
        summary, _ = find_spatiotemporal_duplicates(comfort_conn, ["NITRATE"])
        assert summary.iloc[0]["number of row duplicates"] == 1

    # No duplicates found: Empty summary with defined columns
    def test_summary_empty(self, comfort_conn):
        comfort_conn.execute("CREATE TABLE P_X (ID INTEGER, LEV_M REAL, VAL REAL)")
        summary, details = find_spatiotemporal_duplicates(comfort_conn, ["X"])
        assert summary.empty
        assert details == {}
        assert list(summary.columns) == [
            "parameter", "number of samples", "number of row duplicates", "number of time loc duplicates",
            "proportion of row duplicates", "proportion of time loc duplicates",
        ]
