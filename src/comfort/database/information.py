"""Functions to query metadata from the COMFORT database.

Adapted from Jenniges (2025), doi:10.5281/zenodo.15827777
"""
from __future__ import annotations

import logging
import sqlite3
import numpy as np
import pandas as pd

from ..qc import build_where_clause
from ..util.sqlite_utils import validate_identifier

logger = logging.getLogger(__name__)


def does_table_exist(conn: sqlite3.Connection, table_name: str, table_type: str = "table") -> bool:
    """Check if a table/view exists (case-sensitive).

    Args:
        conn (sqlite3.Connection): Connection to the database.
        table_name (str): Table name to check.
        table_type (str): 'table' or 'view'. Default is 'table'.
    Returns:
        bool: True if the table/view exists.
    """
    # Validate identifier
    validate_identifier(table_name)

    # Connect to and query db
    cur = conn.cursor()
    result = cur.execute(
        "SELECT name FROM sqlite_master WHERE type=? AND name=?;",
        (table_type, table_name),
    ).fetchall()
    return bool(result)


def get_min_max_dates(conn: sqlite3.Connection, table_name: str = "e_combined",
                      time_column: str = "DATEANDTIME") -> tuple[pd.Timestamp, pd.Timestamp]:
    """Fetch minimum and maximum dates from a table.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        table_name (str): Table or view to query.
        time_column (str): Column containing date/time values.
    Returns:
        tuple[pandas.Timestamp, pandas.Timestamp]: (min_date, max_date).
    """
    # Validate identifiers
    validate_identifier(table_name)
    validate_identifier(time_column)

    # Connect to db
    cur = conn.cursor()

    # Fetch min/max dates
    min_date = pd.Timestamp(cur.execute(f"SELECT MIN({time_column}) FROM {table_name};").fetchone()[0])
    max_date = pd.Timestamp(cur.execute(f"SELECT MAX({time_column}) FROM {table_name};").fetchone()[0])
    return min_date, max_date


def get_num_samples(conn: sqlite3.Connection, table_name: str, table_type: str = "table",
                    quality_flags: list | None = None) -> int:
    """Get the number of samples in the given table.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        table_name (str): Table or view name.
        table_type (str): 'table' or 'view'. Default is 'table'.
        quality_flags (list[QCFilter] or list[tuple[str, str]]): Optional quality flag filters.
    Returns:
        int: Number of samples, or 0 if the table does not exist.
    """
    # Validate identifier
    validate_identifier(table_name)

    # Build quality filter
    quality_statement = build_where_clause(quality_flags)

    # Check if table exists
    if not does_table_exist(conn, table_name, table_type):
        logger.error(f"No such {table_type}: {table_name}")
        return 0

    # Execute counting query
    result = conn.cursor().execute(f"SELECT COUNT(*) FROM {table_name} {quality_statement};").fetchone()
    return result[0]


def get_table_as_df(conn: sqlite3.Connection, table_name: str,
                    columns: list[str] | None = None) -> pd.DataFrame:
    """Fetch a table from the database as a DataFrame.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        table_name (str): Name of the table to fetch.
        columns (list[str]): Columns to query. ``None`` means all.
    Returns:
        pandas.DataFrame
    """
    # Validate identifiers
    validate_identifier(table_name)
    if columns:
        for col in columns:
            validate_identifier(col)

    # Connect to db
    cur = conn.cursor()

    # Fetch data
    col_selection = "*" if not columns else ", ".join(columns)
    ex = cur.execute(f"SELECT {col_selection} FROM {table_name};")
    cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(ex.fetchall(), columns=cols)


def get_names_of_all_parameter_tables(conn: sqlite3.Connection, like_pattern: str = "P|_%",
                                      escape_char: str = "|", include_digits: bool = False,
                                      table_type: str = "table") -> list[str]:
    """Get all table/view names matching the given LIKE pattern.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        like_pattern (str): SQLite LIKE pattern.
        escape_char (str): Escape character for the LIKE pattern.
        include_digits (bool): Include tables with numeric suffixes.
        table_type (str): ``'table'`` or ``'view'``. Default is 'table'.
    Returns:
        list[str]: Matching table/view names.
    """
    # Exclude tables with numeric suffixes (e.g. grid-mapped tables) unless requested
    digits_filter = "" if include_digits else "and name not glob '*_[0-9]*'"

    # Build the query
    if like_pattern and escape_char:
        query = (f"SELECT name FROM sqlite_master "
                 f"WHERE type='{table_type}' AND name LIKE '{like_pattern}' ESCAPE '{escape_char}' "
                 f"{digits_filter};")
    elif like_pattern:
        query = (f"SELECT name FROM sqlite_master "
                 f"WHERE type='{table_type}' AND name LIKE '{like_pattern}' "
                 f"{digits_filter};")
    else:
        query = f"SELECT name FROM sqlite_master WHERE type='{table_type}' {digits_filter};"

    logger.debug(query)

    # Execute query
    result = conn.cursor().execute(query).fetchall()
    return [entry[0] for entry in result]


def get_minmax(conn: sqlite3.Connection, column: str = "LATITUDE", column_type: str = "float",
               param_tables: list[str] | None = None) -> tuple:
    """Get the global min and max of a column across all parameter tables.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        column (str): Column name to aggregate.
        column_type (str): ``'float'``, ``'int'`` or ``'datetime64'``.
        param_tables (list[str]): Tables to query. ``None`` means all.
    Returns:
        tuple: (minimum, maximum), or (None, None) on error.
    """
    # Validate identifier
    validate_identifier(column)

    # Get parameter table names
    if not param_tables:
        param_tables = get_names_of_all_parameter_tables(conn)

    column_minimum = None
    column_maximum = None

    # Determine how to convert the raw SQL result to the requested type
    converters = {"float": float, "int": int, "datetime64": np.datetime64}
    if column_type not in converters:
        logger.error(f"get_minmax: conversion to {column_type} is not implemented")
        return None, None
    convert = converters[column_type]

    # Iterate over parameter tables
    for table in param_tables:
        # Execute query / get min and max
        cur = conn.execute(f"SELECT MIN({column}), MAX({column}) FROM {table};")
        raw = cur.fetchone()
        res = [convert(x) for x in raw]

        # Set the column minimum/maximum (use first when multiple entries are returned)
        if column_minimum is None or column_minimum > res[0]:
            column_minimum = res[0]
        if column_maximum is None or column_maximum < res[1]:
            column_maximum = res[1]

    return column_minimum, column_maximum
