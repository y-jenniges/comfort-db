"""Functions to query metadata from the COMFORT database.

Adapted from Jenniges (2025), doi:10.5281/zenodo.15827777
"""
from __future__ import annotations

__all__ = [
    "count_negative_samples",
    "count_samples_over_time",
    "count_samples_per_parameter",
    "does_table_exist",
    "find_spatiotemporal_duplicates",
    "get_min_max_dates",
    "get_minmax",
    "get_names_of_all_parameter_tables",
    "get_num_samples",
    "get_table_as_df",
    "param_label",
    "param_table_name",
]

import logging
import sqlite3
import numpy as np
import pandas as pd

from ..qc import build_where_clause
from ..util.sqlite_utils import validate_identifier

logger = logging.getLogger(__name__)


def param_label(name: str) -> str:
    """Return the canonical parameter name: Upper case, without ``P_`` prefix.

    ``'nitrate'``, ``'NITRATE'``, ``'p_nitrate'`` and ``'P_NITRATE'`` all give ``'NITRATE'``.
    Use it for dictionary keys, column names and plot labels.

    Args:
        name (str): Parameter name or ``P_*`` table name, any casing.
    Returns:
        str: Canonical parameter name.
    """
    name = name.strip().upper()
    return name[2:] if name.startswith("P_") else name


def param_table_name(name: str) -> str:
    """Return the ``P_*`` table name of a parameter.

    ``'nitrate'``, ``'NITRATE'``, ``'p_nitrate'`` and ``'P_NITRATE'``  all give ``'P_NITRATE'``.
    See :func:`param_label` for the inverse.

    Args:
        name (str): Parameter name or ``P_*`` table name, any casing.
    Returns:
        str: Table name.
    """
    return f"P_{param_label(name)}"


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


def count_samples_over_time(conn: sqlite3.Connection, parameter: str, *, time_mode: str = "year",
                            quality_flags: list | None = None) -> pd.DataFrame:
    """Count the number of samples of a parameter per year or per month.
    Counting is done in SQL and rows without a valid date are ignored.

    Plot the result with :func:`comfort.plot.plot_counts_bar` (``tick_style`` set to *time_mode*).

    Args:
        conn (sqlite3.Connection): Connection to the database.
        parameter (str): Parameter name. Insensitive to casing and ``P_`` prefix.
        time_mode (str): ``'year'`` (every year between first and last, empty years
            counted as 0) or ``'month'`` (1 to 12).
        quality_flags (list[QCFilter] or list[tuple[str, str]]): Optional quality flag filters.
    Returns:
        pandas.DataFrame: Columns ``[YEAR|MONTH, COUNT]``.
    """
    # Select the SQL date format
    formats = {"year": "%Y", "month": "%m"}
    if time_mode not in formats:
        raise ValueError(f"time_mode must be 'year' or 'month', got {time_mode!r}")
    column = time_mode.upper()

    # Validate identifier and build quality filter
    table = param_table_name(parameter)
    validate_identifier(table)
    quality_statement = build_where_clause(quality_flags)

    # Count in SQL, joining the station table for the date
    rows = conn.execute(
        f"SELECT CAST(strftime('{formats[time_mode]}', s.DATEANDTIME) AS INTEGER) AS T, COUNT(*) "
        f"FROM {table} p JOIN station s ON p.ID = s.ID {quality_statement} GROUP BY T;"
    ).fetchall()
    counts = pd.Series({t: n for t, n in rows if t is not None}, dtype="int64")

    # Fill unobserved years / months with 0
    if time_mode == "month":
        counts = counts.reindex(np.arange(1, 13), fill_value=0)
    elif not counts.empty:
        counts = counts.reindex(np.arange(counts.index.min(), counts.index.max() + 1), fill_value=0)

    out = counts.sort_index().reset_index()
    out.columns = [column, "COUNT"]
    return out


def count_samples_per_parameter(conn: sqlite3.Connection, parameters: list[str] | None = None,
                                *, quality_flags: list | None = None) -> pd.DataFrame:
    """Count the total number of samples per parameter.

    Plot the result with :func:`comfort.plot.plot_counts_bar` (``tick_style=None``).

    Args:
        conn (sqlite3.Connection): Connection to the database.
        parameters (list[str]): Parameter names. Insensitive to casing and ``P_`` prefix. ``None`` means all.
        quality_flags (list[QCFilter] or list[tuple[str, str]]): Optional quality flag filters.
    Returns:
        pandas.DataFrame: Columns ``[parameter, count]``.
    """
    # Default to all parameter tables in the database
    if not parameters:
        parameters = get_names_of_all_parameter_tables(conn)
    parameters = [param_label(p) for p in parameters]

    # Count samples per parameter via SQL COUNT(*)
    df_count = pd.DataFrame({"parameter": parameters, "count": 0})
    for i, name in enumerate(parameters):
        df_count.loc[i, "count"] = get_num_samples(
            conn, param_table_name(name), table_type="table", quality_flags=quality_flags,
        )
    return df_count


def count_negative_samples(conn: sqlite3.Connection, parameters: list[str], *,
                           quality_flags: list | None = None) -> pd.DataFrame:
    """Count the proportion of negative values per parameter.

    Plot the proportions with :func:`comfort.plot.plot_counts_bar` (``tick_style=None``)
    on ``df[["parameter", "proportion of negative samples"]]``.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        parameters (list[str]): Parameter names. Insensitive to casing and ``P_`` prefix.
        quality_flags (list[QCFilter] or list[tuple[str, str]]): Optional quality flag filters
            (applied to the negative count only).
    Returns:
        pandas.DataFrame: Columns ``[parameter, number of samples, number of negative samples,
        proportion of negative samples]``, sorted by the proportion.
    """
    cursor = conn.cursor()
    quality_statement = build_where_clause(quality_flags, keyword="AND")

    # Query negative and total counts for each parameter
    rows = []
    for parameter in map(param_label, parameters):
        table = param_table_name(parameter)
        validate_identifier(table)
        logger.info("count_negative_samples: %s", parameter)
        negative_count = cursor.execute(
            f"SELECT COUNT(*) FROM {table} WHERE VAL<0 {quality_statement};"
        ).fetchone()[0]
        count = cursor.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
        rows.append({
            "parameter": parameter,
            "number of samples": count,
            "number of negative samples": negative_count,
        })

    # Compute proportions and sort for display
    df = pd.DataFrame(rows, columns=["parameter", "number of samples", "number of negative samples"])
    df["proportion of negative samples"] = df["number of negative samples"] / df["number of samples"] * 100
    return df.sort_values("proportion of negative samples")


def find_spatiotemporal_duplicates(conn: sqlite3.Connection, parameters: list[str], *,
                                   quality_flags: list | None = None
                                   ) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    """Find spatio-temporal duplicates (same station and depth) per parameter.

    Plot the result with :func:`comfort.plot.plot_duplicates`.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        parameters (list[str]): Parameter names. Insensitive to casing and ``P_`` prefix.
        quality_flags (list[QCFilter] or list[tuple[str, str]]): Optional quality flag filters.
    Returns:
        tuple: ``(summary, details)``. *summary* is a DataFrame with columns ``[parameter,
        number of samples, number of row duplicates, number of time loc duplicates,
        proportion of row duplicates, proportion of time loc duplicates]``. *details* maps
        each parameter in *summary* to ``{"time_loc": ..., "rows": ...}``, the duplicated
        (ID, LEV_M) groups with ``size``, ``mean`` and ``std`` and the exactly duplicated
        rows with ``size``.
    """
    cursor = conn.cursor()
    quality_statement = build_where_clause(quality_flags)

    rows = []
    details = {}
    # Iterate over parameters
    for parameter in map(param_label, parameters):
        table = param_table_name(parameter)
        validate_identifier(table)

        # Fetch all rows for this parameter
        logger.info("find_spatiotemporal_duplicates: %s", parameter)
        ex = cursor.execute(f"SELECT * FROM {table} {quality_statement};")
        df_param = pd.DataFrame(ex.fetchall(), columns=np.array([x[0] for x in cursor.description]))

        # Spatiotemporal duplicates (same station and depth)
        dups_time_loc = (
            df_param.groupby(["ID", "LEV_M"], as_index=False, dropna=False)
            .agg(size=("VAL", "size"), mean=("VAL", "mean"), std=("VAL", "std"))
        )
        dups_time_loc = dups_time_loc[dups_time_loc["size"] > 1]

        if len(dups_time_loc) < 1:
            logger.info("  %s: no duplicates", parameter)
            continue

        # Exact row duplicates (identical across all columns)
        dups_row = (
            df_param.groupby(df_param.columns.tolist(), as_index=False, dropna=False)
            .size()
        )
        dups_row = dups_row[dups_row["size"] > 1]

        rows.append({
            "parameter": parameter,
            "number of samples": len(df_param),
            "number of row duplicates": len(dups_row),
            "number of time loc duplicates": len(dups_time_loc),
        })
        details[parameter] = {"time_loc": dups_time_loc, "rows": dups_row}

    # Compute duplicate proportions (an empty summary keeps the documented columns)
    columns = ["parameter", "number of samples", "number of row duplicates", "number of time loc duplicates"]
    summary = pd.DataFrame(rows, columns=columns)
    summary["proportion of row duplicates"] = summary["number of row duplicates"] / summary["number of samples"] * 100
    summary["proportion of time loc duplicates"] = (
        summary["number of time loc duplicates"] / summary["number of samples"] * 100
    )
    return summary, details
