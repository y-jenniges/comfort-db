"""Functions to manage views and tables in the COMFORT database."""
from __future__ import annotations

__all__ = [
    "create_combined_parameter_table",
]

import logging
import sqlite3

from .information import does_table_exist, param_label, param_table_name
from ..qc import QualityFlags, build_where_clause
from ..util.sqlite_utils import validate_identifier

logger = logging.getLogger(__name__)


def create_combined_parameter_table(conn: sqlite3.Connection, parameters: list[str],
                                    quality_flags: QualityFlags | None = None,
                                    columns: list[str] | None = None,
                                    table_type: str = "view") -> None:
    """Create a UNION ALL table/view combining multiple parameters.

    Adds a PARAM_NAME column. Assumes P_* tables exist.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        parameters (list[str]): Parameter names to combine.
        quality_flags (QualityFlags): Quality flag filters.
        columns (list[str]): Columns to include. Defaults to the standard COMFORT parameter columns.
        table_type (str): 'view' or 'table'. Default is 'view'.
    """
    # Define columns to include
    if columns is None:
        columns = ["ID", "LEV_DBAR", "LEV_M", "VAL", "PQF1", "PQF2", "SQF", "BOTTLE_NUMBER",
                   "PROFILE_NUMBER", "PROFILE_BEST", "UNITS_ID", "INSTRUMENT_ID"]

    # Validate identifiers (parameters in canonical spelling)
    parameters = [param_label(p) for p in parameters]
    for p in parameters:
        validate_identifier(p)
    for col in columns:
        validate_identifier(col)

    # Build quality filter
    quality_statement = build_where_clause(quality_flags)

    # Iterate over parameters
    select_params = []
    for p in parameters:
        # Check if view/table exist
        view_exists = does_table_exist(conn, param_table_name(p), "view")
        table_exists = does_table_exist(conn, param_table_name(p), "table")

        # Build data-fetching query
        if view_exists or table_exists:
            select_params.append(
                f"SELECT '{p}' AS PARAM_NAME, {', '.join(columns)} "
                f"FROM {param_table_name(p)} {quality_statement}"
            )
        else:
            logger.warning(f"create_combined_parameter_table: {param_table_name(p)} does not exist - skipping")

    # Build combined query and execute
    query = (f"CREATE {table_type} IF NOT EXISTS P_COMBINED AS "
             f"SELECT * FROM ({' UNION ALL '.join(select_params)});")
    conn.cursor().execute(query)
    conn.commit()
