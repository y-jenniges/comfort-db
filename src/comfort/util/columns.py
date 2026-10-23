"""Helper to find parameter columns in DataFrames."""
from __future__ import annotations

import pandas as pd

from ..database.information import param_label, param_table_name


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
    if isinstance(parameters, str):
        parameters = [parameters]

    # Case-insensitive lookup of the actual column names
    by_upper = {str(c).upper(): c for c in df.columns}

    columns = []
    # Iterate over parameter names
    for param_name in parameters:
        # Check alternative spellings
        for candidate in (param_name, param_table_name(param_name), param_label(param_name)):
            if candidate in df.columns:
                columns.append(candidate)
                break
            # Check casing
            if candidate.upper() in by_upper:
                columns.append(by_upper[candidate.upper()])
                break
        else:
            raise KeyError(f"No column for parameter {param_name!r} in the DataFrame")
    return columns
