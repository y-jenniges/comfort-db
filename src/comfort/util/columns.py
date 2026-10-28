"""Helpers to find columns in DataFrames, insensitive to casing."""
from __future__ import annotations

import pandas as pd


def find_col(df: pd.DataFrame, name_to_check: str | list[str] | None, *, required: bool = True):
    """Return the column name in *df* that matches *name*, insensitive to casing.

    An exact match wins. Otherwise, the one column that matches ignoring case is returned.

    Args:
        df (pandas.DataFrame): DataFrame to search.
        name_to_check (str or list[str]): Column name to look for. A list is resolved element by element.
            ``None`` is returned as ``None``, so optional column arguments can be passed through.
        required (bool): If ``True``, a missing column raises. Otherwise, ``None`` is returned.
    Returns:
        str or None: The column name as it is spelled in *df*.
    Raises:
        KeyError: If *name* is not found and *required* is ``True`` or if several columns
            match *name* only when ignoring case (e.g. ``Val`` and ``VAL`` for ``val``).
    """
    if name_to_check is None:
        return None
    if isinstance(name_to_check, (list, tuple)):
        return [find_col(df, n, required=required) for n in name_to_check]

    # Exact match
    if name_to_check in df.columns:
        return name_to_check

    # Match ignoring case
    matches = [c for c in df.columns if str(c).casefold() == str(name_to_check).casefold()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"Column {name_to_check!r} is ambiguous in DataFrame, it matches {matches}")

    # Column not found
    if required:
        raise KeyError(f"Column {name_to_check!r} not found in DataFrame. Available columns: {list(df.columns)}")
    return None


def resolve_param_columns(df: pd.DataFrame, parameters: list[str] | str) -> list[str]:
    """Return the columns of *df* that belong to the given parameters.

    A parameter is found under its spelling in *parameters*, as ``P_*`` table name or as
    canonical label, ignoring case. Thus, ``'nitrate'``, ``'NITRATE'``, ``p_nitrate`` and ``'P_NITRATE'`` all
    find the column ``P_NITRATE``.

    Args:
        df (pandas.DataFrame): DataFrame with one column per parameter.
        parameters (list[str] or str): Parameter names. Insensitive to casing and ``P_`` prefix.
    Returns:
        list[str]: The matching column names, in the order of *parameters*.
    Raises:
        KeyError: If a parameter has no column in *df*.
    """
    from ..database.information import param_label, param_table_name

    # Wrap single parameter into a list
    if isinstance(parameters, str):
        parameters = [parameters]

    columns = []
    # Iterate over parameter names
    for param_name in parameters:
        # Check alternative spellings
        for candidate in (param_name, param_table_name(param_name), param_label(param_name)):
            column = find_col(df, name_to_check=candidate, required=False)
            if column is not None:
                columns.append(column)
                break
        else:
            raise KeyError(f"No column for parameter {param_name!r} in the DataFrame")
    return columns
