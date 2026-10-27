"""Parameter value scaling for the COMFORT database."""
from __future__ import annotations

__all__ = [
    "ParamScaler",
]

import logging
import sqlite3
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

from .util.optional import import_optional
from .util.sqlite_utils import validate_identifier

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


class ParamScaler:
    """ Scale parameters using a scikit-learn scaler per parameter.

    Args:
        scaler_class: A scikit-learn scaler class (must support fit/transform).
            Default is MinMaxScaler.
        **scaler_kwargs: Keyword arguments forwarded to scaler_class().
    """

    def __init__(self, scaler_class: type[Any] | None = None, **scaler_kwargs):
        preprocessing = import_optional("sklearn.preprocessing", "scale", "scaling")

        # Set up scalers per parameter
        self.scaler_class = scaler_class or preprocessing.MinMaxScaler
        self.scaler_kwargs = scaler_kwargs
        self.scalers: dict = {}

    def _get_or_fit_scaler(self, parameter: str, vals: np.ndarray) -> Any:
        """Return a fitted scaler for parameter, creating one if needed."""
        if parameter in self.scalers:
            return self.scalers[parameter]

        # Fit and store new scaler
        scaler = self.scaler_class(**self.scaler_kwargs).fit(vals)
        self.scalers[parameter] = scaler
        return scaler

    def scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale the VAL column. df must have a single PARAM_NAME value.

        Args:
            df (pandas.DataFrame): Contains ``VAL`` and ``PARAM_NAME`` columns.
        """
        # Prepare for scaling
        temp = df.copy()
        param_name = temp["PARAM_NAME"].value_counts().index[0]
        vals = np.array(temp["VAL"]).reshape(-1, 1)

        # Scale values
        scaler = self._get_or_fit_scaler(param_name, vals)
        temp["VAL_SCALED"] = scaler.transform(vals)
        return temp

    def scale_columns(self, columns: list[str], df: pd.DataFrame) -> pd.DataFrame:
        """Scale the given columns (one parameter per column).

        Args:
            columns (list[str]): Columns to scale.
            df (pandas.DataFrame): Source data.
        """
        temp = df.copy()

        # Iterate over columns
        for col in columns:
            # Check if column is in the df
            if col not in df.columns:
                logger.warning(f"scale_columns: column {col!r} not found in dataframe")
                continue

            # Scale
            vals = np.array(df[col]).reshape(-1, 1)
            scaler = self._get_or_fit_scaler(col, vals)

            # Add scaled values to df
            temp[col + "_SCALED"] = scaler.transform(vals)
        return temp

    def scale_value(self, parameter: str, value: float) -> float:
        """Scale a single value using the already-fitted scaler for parameter.

        Args:
            parameter (str): Parameter name.
            value (float): Value to scale.
        Raises:
            KeyError: If no scaler has been fitted for parameter.
        """
        # Check if there is a scaler for the parameter
        if parameter not in self.scalers:
            raise KeyError(f"No scaler fitted for {parameter!r} - call scale_columns first")

        return self.scalers[parameter].transform([[value]])[0][0]


class DBParamScaler:
    """ Scale parameters to [lower, upper] by operating directly on the database via SQL.

    Args:
        connection (sqlite3.Connection): Must hold one table per parameter.
        lower (float): Minimum after scaling. Default is 0.
        upper (float): Maximum after scaling. Default is 1.
        value_column (str): Value column name. Default is ``'VAL'``.
        depth_column (str): Depth column name. Default is ``'LEV_M'``.
    """

    def __init__(self, connection: sqlite3.Connection, lower: float = 0, upper: float = 1,
                 value_column: str = "VAL", depth_column: str = "LEV_M") -> None:
        self.connection = connection
        self.lower_limit = lower
        self.upper_limit = upper
        self.value_column = value_column
        self.depth_column = depth_column
        self.scalers = pd.DataFrame(columns=["param_name", "min", "max"])

    def get_min_max(self, table_name: str, column: str) -> tuple[float, float]:
        """Return (min, max) of a column."""
        # Validate identifiers
        validate_identifier(table_name)
        validate_identifier(column)

        # Fetch min/max values
        res = self.connection.cursor().execute(
            f"SELECT MIN({column}), MAX({column}) FROM {table_name};"
        ).fetchone()
        return res[0], res[1]

    def get_min_max_depth(self, table_names: list[str]) -> tuple[float, float]:
        """Return global (min, max) of the depth column across all tables."""
        # Iterate over tables
        mins, maxs = [], []
        for table in table_names:
            # Get min/max depth
            tmin, tmax = self.get_min_max(table, self.depth_column)
            mins.append(tmin)
            maxs.append(tmax)
        return min(mins), max(maxs)

    def _min_max_formula(self, table: str, column: str, param_min: float,
                         param_max: float) -> str:
        """Return a SQL expression for MinMax scaling."""
        lo, hi = self.lower_limit, self.upper_limit
        return (f"({table}.{column} - {param_min}) / ({param_max} - {param_min}) "
                f"* ({hi} - {lo}) + {lo}")

    def scale(self, tables_params: dict[str, str]) -> dict[str, str]:
        """Create a ``scaled_<table>`` table with VAL and depth scaled to [lower, upper].

        Args:
            tables_params (dict): Maps table names to parameter names.
        """
        if not tables_params:
            return {}

        # Shared depth range across all parameter tables
        param_tables = list(tables_params.keys())
        min_depth, max_depth = self.get_min_max_depth(param_tables)
        self.scalers = pd.concat(
            [self.scalers,
             pd.DataFrame([{"param_name": self.depth_column, "min": min_depth, "max": max_depth}])],
            ignore_index=True,
        )

        new_tables_params = {}
        for table, param in tables_params.items():
            # Validate table name
            validate_identifier(table)
            logger.info(f"DBParamScaler: scaling {table} ({param})")

            # Get min/max values
            param_min, param_max = self.get_min_max(table, self.value_column)

            # Scale
            val_formula = self._min_max_formula(table, self.value_column, param_min, param_max)
            depth_formula = self._min_max_formula(table, self.depth_column, min_depth, max_depth)

            # Store scaler info
            self.scalers = pd.concat(
                [self.scalers,
                 pd.DataFrame([{"param_name": param, "min": param_min, "max": param_max}])],
                ignore_index=True,
            )

            # Get column names
            cur = self.connection.cursor()
            cur.execute(f"SELECT * FROM {table} LIMIT 1;")
            old_columns = [desc[0] for desc in cur.description]

            # Define new column names
            new_columns = [f"{table}.{x}" for x in old_columns
                           if x not in (self.value_column, "UNITS_ID", self.depth_column)]

            # Create new, scaled table
            new_table = f"scaled_{table}"
            cur.execute(
                f"CREATE TABLE {new_table} AS "
                f"SELECT {', '.join(new_columns)}, "
                f"{val_formula} AS {self.value_column}, "
                f"{depth_formula} AS {self.depth_column} "
                f"FROM {table};"
            )
            new_tables_params[new_table] = param

        return new_tables_params
