"""Quality-flag presets, SQL clause builders and DataFrame QC filters."""
from __future__ import annotations

import pandas as pd
import logging
import operator as op
import re
from collections import namedtuple

from .util import validate_identifier

logger = logging.getLogger(__name__)

# Map of comparison operator strings to their Python equivalents
_OP_MAP = {
    ">=": op.ge,
    "<=": op.le,
    ">": op.gt,
    "<": op.lt,
    "==": op.eq,
    "!=": op.ne,
}
QCFilter = namedtuple("QCFilter", ["column", "condition"])

# Accept all records regardless of quality flags.
QC_ALL = []

# Accept records that passed basic quality control (PQF1 checked, PQF2 acceptable).
QC_GOOD = [QCFilter("PQF1", ">0"), QCFilter("PQF2", ">2")]

# Operator followed by a decimal number, e.g. ">0", ">= 2", "!=-1"
_CONDITION_RE = re.compile(r"^\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)\s*$")


def _parse_flag(column: str, condition: str) -> tuple[str, str, str]:
    """Validate one quality flag and split its condition into operator and number.

    Args:
        column (str): Name of the flag column, e.g. ``"PQF1"``.
        condition (str): Operator followed by a number, e.g. ``">0"``.
            Supported operators: ``>=``, ``<=``, ``>``, ``<``, ``==``, ``!=``.
    Returns:
        tuple[str, str, str]: Column name, operator and number.
    Raises:
        ValueError: If *column* is not a safe SQL identifier or *condition*
            is not a supported operator followed by a number.
    """
    # Validate the column name
    validate_identifier(column)

    # Split the condition into operator and number
    match = _CONDITION_RE.match(condition) if isinstance(condition, str) else None
    if match is None:
        raise ValueError(
            f"Invalid quality flag condition {condition!r} for column {column!r}: "
            "expected one of >=, <=, >, <, ==, != followed by a number, e.g. '>0'."
        )
    return column, match.group(1), match.group(2)


def qc_conditions(quality_flags: list[QCFilter] | list[tuple[str, str]] | None,
                  table_alias: str | None = None) -> list[str]:
    """Return the SQL conditions for the given quality flags.

    Args:
        quality_flags: Quality flag criteria, e.g. ``QC_GOOD``.
        table_alias: Optional table alias to prefix column names.
    Returns:
        list[str]: One condition per flag, e.g. ``["p.PQF1>0", "p.PQF2>2"]``.
        Empty when *quality_flags* is ``None`` or empty.
    Raises:
        ValueError: If a flag or *table_alias* is invalid (see :func:`_parse_flag`).
    """
    # No quality flags: nothing to filter
    if not quality_flags:
        return []

    # Validate the table alias and build the column prefix
    if table_alias:
        validate_identifier(table_alias)
    prefix = f"{table_alias}." if table_alias else ""

    # Build each condition from the validated parts
    conditions = []
    for column, condition in quality_flags:
        column, operator, number = _parse_flag(column, condition)
        conditions.append(f"{prefix}{column}{operator}{number}")
    return conditions


def build_where_clause(quality_flags: list[QCFilter] | list[tuple[str, str]] | None,
                       table_alias: str | None = None,
                       keyword: str = "WHERE") -> str:
    """Return a SQL clause for the given quality flags.

    Args:
        quality_flags: Quality flag criteria, e.g. ``QC_GOOD``.
        table_alias: Optional table alias to prefix column names.
        keyword: SQL keyword to lead with (``"WHERE"`` or ``"AND"``).
    Returns:
        Clause string, or ``""`` when *quality_flags* is empty.
    """
    conditions = qc_conditions(quality_flags, table_alias)
    if not conditions:
        return ""
    return f"{keyword} " + " AND ".join(conditions)


def apply_qc_flags(df: pd.DataFrame,
                   quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Filter a DataFrame by quality flags.

    Same filtering as the SQL-level ``quality_flags`` on
    :func:`comfort.io.read_parameter`, but on an already-loaded DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame with QC flag columns (e.g. PQF1, PQF2, SQF).
        quality_flags (list[QCFilter] or list[tuple[str, str]]): Quality flag
            filters, e.g. :data:`QC_GOOD`. ``None`` or ``[]`` returns df unchanged.
    Returns:
        pandas.DataFrame: Filtered copy.
    """
    # Return plain df if no qc flags were provided
    if not quality_flags:
        return df.copy()

    # Start with all rows selected, then narrow down
    mask = pd.Series(True, index=df.index)
    for column, condition in quality_flags:
        # Parse and validate the quality flag
        column, operator, number = _parse_flag(column, condition)

        # A missing flag column would silently leave the data unfiltered
        if column not in df.columns:
            raise ValueError(f"apply_qc_flags: column {column!r} not in DataFrame")

        mask &= _OP_MAP[operator](df[column], float(number))

    return df[mask].copy()


def flag_salinity_like_oxygen(df: pd.DataFrame, oxygen_col: str = "OXYGEN",
                              salinity_col: str | None = None,
                              tol: float = 0.0) -> pd.Series:
    """Flag oxygen values that are identical to the co-located Practical Salinity
    (known COMFORT data issue, about 14% of oxygen rows are affected).
    See ``known_data_issues/known_data_issues.md``.

    Args:
        df (pandas.DataFrame): Must contain *oxygen_col* and a Practical
            Salinity column (``SALINITY``/``salinity``).
        oxygen_col (str): Oxygen value column. Default ``'OXYGEN'``.
        salinity_col (str): Practical Salinity column to compare against.
            ``None`` auto-detects ``SALINITY``/``salinity``.
        tol (float): Maximum absolute difference between oxygen and salinity
            for a row to be flagged. Default 0 flags exact copies only.
    Returns:
        pandas.Series: Bool mask aligned with df (True = suspect row).
    Raises:
        ValueError: If the oxygen or salinity column cannot be found, or if *tol* is negative.
    """
    if tol < 0:
        # Tolerance must be positive (or zero)
        raise ValueError(f"flag_salinity_like_oxygen: tol must be >= 0, got {tol}")

    # Resolve column names case-insensitively
    def _find(name):
        for col in df.columns:
            if col.lower() == name.lower():
                return col
        return None

    # Find the oxygen column
    oxy = _find(oxygen_col)
    if oxy is None:
        raise ValueError(f"flag_salinity_like_oxygen: column {oxygen_col!r} not found")

    # Auto-detect the salinity column when not given explicitly
    if salinity_col is None:
        sal = _find("SALINITY")
        if sal is None:
            raise ValueError("flag_salinity_like_oxygen: No SALINITY column found")
    else:
        sal = _find(salinity_col)
        if sal is None:
            raise ValueError(f"flag_salinity_like_oxygen: Column {salinity_col!r} not found")

    # Suspect rows: Oxygen equals the salinity value
    mask = (df[oxy] - df[sal]).abs() <= tol
    if mask.any():
        # Print how common the issue is in this DataFrame
        logger.warning(
            "flag_salinity_like_oxygen: %d of %d rows (%.2f%%) look like mislabelled salinity",
            int(mask.sum()), len(df), 100 * mask.mean(),
        )
    return mask
