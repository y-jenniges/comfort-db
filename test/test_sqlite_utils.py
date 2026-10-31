"""Tests for comfort.util.sqlite_utils."""
import pytest

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