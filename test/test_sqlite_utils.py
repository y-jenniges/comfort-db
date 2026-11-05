"""Tests for comfort.util.sqlite_utils."""
import logging
import sqlite3

import pytest

from comfort.database import vacuum
from comfort.util.sqlite_utils import validate_identifier


class TestValidateIdentifier:
    # Accept a normal table/column name
    def test_accepts_valid_identifier(self):
        validate_identifier("P_NITRATE")  # Must not raise

    # Accept identifiers starting with an underscore
    def test_accepts_underscore_prefix(self):
        validate_identifier("_hidden")

    # Reject a name containing a SQL injection payload
    def test_rejects_sql_injection_attempt(self):
        with pytest.raises(ValueError):
            validate_identifier("foo; DROP TABLE bar;--")

    # Reject non-string input
    def test_rejects_non_string(self):
        with pytest.raises(ValueError):
            validate_identifier(123)

    # Reject identifiers that start with a digit
    def test_rejects_identifier_starting_with_digit(self):
        with pytest.raises(ValueError):
            validate_identifier("1table")


class TestVacuum:
    # VACUUM runs on a normal connection and is reachable from comfort.database
    def test_runs(self, tmp_path):
        conn = sqlite3.connect(tmp_path / "db.sqlite")
        conn.execute("CREATE TABLE t (a INTEGER)")
        conn.commit()
        vacuum(conn)  # must not raise
        conn.close()

    # A failing VACUUM (e.g. inside a transaction) only logs a warning
    def test_failure_logs_warning(self, tmp_path, caplog):
        conn = sqlite3.connect(tmp_path / "db.sqlite")
        conn.execute("CREATE TABLE t (a INTEGER)")
        conn.execute("BEGIN")
        conn.execute("INSERT INTO t VALUES (1)")
        with caplog.at_level(logging.WARNING):
            vacuum(conn)
        assert "VACUUM failed" in caplog.text
        conn.close()
