"""Tests for comfort.database."""
import sqlite3
import pytest

from comfort.database.information import does_table_exist, get_table_as_df
from comfort.database.structure import (
    create_combined_parameter_table,
    remove_tables_like,
)


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
        (2, 0.0, 0.0, 8.0, 1, 3, 0, 1, 2, 1, 3, 2),
    ])
    conn.executemany("INSERT INTO P_OXYGEN VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        (1, 0.0, 0.0, 200.0, 1, 3, 0, 1, 1, 1, 3, 1),
        (2, 0.0, 0.0, 220.0, 1, 3, 0, 1, 2, 1, 3, 2),
    ])

    conn.commit()
    yield conn
    conn.close()


class TestRemoveTablesLike:
    @staticmethod
    def _create_views(conn):
        for param in ("NITRATE", "OXYGEN"):
            conn.execute(f"CREATE VIEW V_{param} AS SELECT * FROM P_{param}")

    # Drop every view matching the LIKE pattern
    def test_removes_views(self, comfort_conn):
        self._create_views(comfort_conn)
        assert does_table_exist(comfort_conn, "V_NITRATE", "view")

        remove_tables_like(comfort_conn, like_pattern="V|_%", escape_char="|", table_type="view")
        assert not does_table_exist(comfort_conn, "V_NITRATE", "view")
        assert not does_table_exist(comfort_conn, "V_OXYGEN", "view")

    # tables_except protects named views from removal
    def test_tables_except(self, comfort_conn):
        self._create_views(comfort_conn)
        remove_tables_like(
            comfort_conn, like_pattern="V|_%", escape_char="|",
            table_type="view", tables_except=["V_NITRATE"],
        )
        # V_NITRATE should survive, V_OXYGEN should be dropped
        assert does_table_exist(comfort_conn, "V_NITRATE", "view")
        assert not does_table_exist(comfort_conn, "V_OXYGEN", "view")

    # table_type="table" drops tables (as GridManager.remove_grid does) and leaves parameter tables alone
    def test_removes_tables(self, comfort_conn):
        comfort_conn.execute("CREATE TABLE grid_1 (idx INTEGER)")
        comfort_conn.execute("CREATE TABLE P_NITRATE_1 AS SELECT * FROM P_NITRATE")
        remove_tables_like(comfort_conn, like_pattern="%|_1", escape_char="|", table_type="table")
        assert not does_table_exist(comfort_conn, "grid_1", "table")
        assert not does_table_exist(comfort_conn, "P_NITRATE_1", "table")
        assert does_table_exist(comfort_conn, "P_NITRATE", "table")

    # The pattern is required: there is no default that could drop unrelated views
    def test_pattern_is_required(self, comfort_conn):
        with pytest.raises(TypeError):
            remove_tables_like(comfort_conn)


class TestCreateCombinedParameterTable:
    # Builds the UNION ALL view across all requested parameters
    def test_creates_combined_view(self, comfort_conn):
        create_combined_parameter_table(comfort_conn, ["NITRATE", "OXYGEN"])
        assert does_table_exist(comfort_conn, "P_COMBINED", "view")

    # Adds a PARAM_NAME column identifying each source parameter
    def test_combined_has_param_name_column(self, comfort_conn):
        create_combined_parameter_table(comfort_conn, ["NITRATE", "OXYGEN"])
        df = get_table_as_df(comfort_conn, "P_COMBINED")
        assert "PARAM_NAME" in df.columns

    # Row count is the sum of rows across all source tables
    def test_combined_row_count(self, comfort_conn):
        create_combined_parameter_table(comfort_conn, ["NITRATE", "OXYGEN"])
        df = get_table_as_df(comfort_conn, "P_COMBINED")
        # 3 nitrate + 2 oxygen = 5
        assert len(df) == 5

    # PARAM_NAME values match the requested parameter list
    def test_combined_param_names(self, comfort_conn):
        create_combined_parameter_table(comfort_conn, ["NITRATE", "OXYGEN"])
        df = get_table_as_df(comfort_conn, "P_COMBINED")
        assert set(df["PARAM_NAME"].unique()) == {"NITRATE", "OXYGEN"}

    # A nonexistent parameter is skipped with a warning, not a hard failure
    def test_nonexistent_param_skipped(self, comfort_conn, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            create_combined_parameter_table(comfort_conn, ["NITRATE", "DOESNOTEXIST"])
        assert "DOESNOTEXIST" in caplog.text
        df = get_table_as_df(comfort_conn, "P_COMBINED")
        # Only NITRATE rows should be present
        assert set(df["PARAM_NAME"].unique()) == {"NITRATE"}
