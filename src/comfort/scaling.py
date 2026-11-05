"""Parameter value scaling for COMFORT data (scikit-learn scalers on DataFrames)."""
from __future__ import annotations

__all__ = [
    "ParamScaler",
]

import logging
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

from .util.optional import import_optional

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


class ParamScaler:
    """Scale parameters using a scikit-learn scaler per parameter.

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
            df: Contains ``VAL`` and ``PARAM_NAME`` columns.
        Returns:
            pandas.DataFrame: Copy of *df* with the added column ``VAL_SCALED``.
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
            columns: Columns to scale.
            df: Source data.
        Returns:
            pandas.DataFrame: Copy of *df* with one added ``<column>_SCALED`` column per scaled column.
        """
        temp = df.copy()

        # Iterate over columns
        for col in columns:
            # Check if column is in the df
            if col not in df.columns:
                logger.warning(f"scale_columns: Column {col!r} not found in dataframe")
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
            parameter: Parameter name.
            value: Value to scale.
        Returns:
            float: The scaled value.
        Raises:
            KeyError: If no scaler has been fitted for parameter.
        """
        # Check if there is a scaler for the parameter
        if parameter not in self.scalers:
            raise KeyError(f"No scaler fitted for {parameter!r} - call scale_columns first")

        return self.scalers[parameter].transform([[value]])[0][0]
