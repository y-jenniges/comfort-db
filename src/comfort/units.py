"""Unit conversion for measured values in the COMFORT database.

Adapted from Jenniges (2025), doi:10.5281/zenodo.15827777
"""
from __future__ import annotations

__all__ = [
    "UnitsConverter",
]

import logging
import sqlite3
import gsw
import numpy as np
import pandas as pd

from .physics import convert_salinity, convert_temperature, resolve_pressure
from .util.sqlite_utils import validate_identifier

logger = logging.getLogger(__name__)

DENSITY_FALLBACKS = ("constant", "drop", "raise")


def _get_column(df: pd.DataFrame, name: str) -> str | None:
    """Return the column name matching name case-insensitively (or None)."""
    for col in df.columns:
        if col.lower() == name.lower():
            return col
    return None


class UnitsConverter:
    """Convert parameter units in the SQLite database to their default units."""

    def __init__(self, connection: sqlite3.Connection, default_units: pd.DataFrame,
                 units: pd.DataFrame) -> None:
        """
        Args:
            connection (sqlite3.Connection): Database connection.
            default_units (pandas.DataFrame): Must have columns NAME_TABLE and UNITS_ID_DEFAULT.
            units (pandas.DataFrame): Must have column ID (and NAME_SHORT for unit names).
        """
        self.connection = connection
        self.df_default_units = default_units
        self.df_units = units
        self.formulas = ConversionFormulas()
        logger.info("Initialised UnitsConverter")

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection) -> UnitsConverter:
        """Alternative constructor:
        Create a UnitsConverter by reading DATABASE_TABLES and UNITS automatically.

        Args:
            conn (sqlite3.Connection): COMFORT database connection.
        Returns:
            UnitsConverter
        Raises:
            ValueError: If required tables are not found.
        """
        from .database.information import get_table_as_df, does_table_exist

        # Check table existence
        if not does_table_exist(conn, "DATABASE_TABLES"):
            raise ValueError("DATABASE_TABLES table not found in database")
        if not does_table_exist(conn, "UNITS"):
            raise ValueError("UNITS table not found in database")

        # Fetch tables
        default_units = get_table_as_df(conn, "DATABASE_TABLES")
        units = get_table_as_df(conn, "UNITS")
        return cls(conn, default_units, units)

    def convert_dataframe(self, df: pd.DataFrame, param_table_name: str,
                          use_density: bool = False, on_missing: str = "drop",
                          density_fallback: str = "constant") -> pd.DataFrame:
        """Convert a DataFrame's values to the default unit.

        Rows that cannot be converted (no UNITS_ID, no registered conversion
        formula, or missing temperature/salinity inputs) are handled according
        to *on_missing*. Counts are always logged.

        Args:
            df (pandas.DataFrame): Must contain ``VAL`` and ``UNITS_ID``.
            param_table_name (str): Table name (e.g. ``'P_TEMPERATURE'``).
            use_density (bool): Use computed density instead of the
                constant 1.025 kg/L.
            on_missing (str): Policy for unconvertible rows: ``'drop'``
                (default) removes them, ``'keep'`` retains them with their
                original VAL and UNITS_ID, ``'raise'`` raises ValueError.
            density_fallback (str): With ``use_density=True``, policy for rows whose
                density cannot be computed (missing/NaN salinity, temperature or
                position): ``'constant'`` (default, use 1.025 kg/L), ``'drop'`` or
                ``'raise'``. Counts are logged.
        """
        # Ensure that only defined missingness policies are passed
        if on_missing not in ("drop", "keep", "raise"):
            raise ValueError(f"on_missing must be 'drop', 'keep' or 'raise', got {on_missing!r}")
        if density_fallback not in DENSITY_FALLBACKS:
            raise ValueError(f"density_fallback must be one of {DENSITY_FALLBACKS}, "
                             f"got {density_fallback!r}")

        # Ensure unit id info is given
        if "UNITS_ID" not in df.columns or df.empty:
            return df

        # Check if default unit exists
        target = self.default_unit_id(param_table_name)
        if target is None:
            logger.warning("convert_dataframe: no default unit for %s", param_table_name)
            return df

        # Get used unit IDs
        unit_ids = df["UNITS_ID"].dropna().unique()
        n_no_units_id = int(df["UNITS_ID"].isna().sum())

        # Check if conversion is required
        if n_no_units_id == 0 and len(unit_ids) == 1 and int(unit_ids[0]) == target:
            return df

        vc = "VAL"
        parts = []
        n_no_formula = 0
        n_missing_input = 0

        # Rows without a unit ID cannot be converted
        if n_no_units_id:
            if on_missing == "raise":
                raise ValueError(f"{param_table_name}: {n_no_units_id} rows have no UNITS_ID")
            if on_missing == "keep":
                parts.append(df[df["UNITS_ID"].isna()])

        # Iterate over unit IDs that need conversion
        for uid in unit_ids:
            chunk = df[df["UNITS_ID"] == uid]

            # Already in the target unit
            if int(uid) == target:
                parts.append(chunk)
                continue

            # Look up conversion formula
            formula = self.formulas.conversions.get((int(uid), target))
            if formula is not None and not self.formulas.is_allowed(int(uid), target, param_table_name):
                # If a formula was found for the conversion, but is not allowed
                # (e.g. only valid for a specific parameter), set to None
                formula = None
            if formula is None:
                # No formula found
                logger.warning(f"Conversion from unit {int(uid)} to {target} not implemented for {param_table_name}")
                # Missing formula policy
                if on_missing == "raise":
                    # Raise error
                    raise ValueError(f"{param_table_name}: no conversion from unit {int(uid)} to {target}")
                if on_missing == "keep":
                    # Keep the values
                    parts.append(chunk)
                n_no_formula += len(chunk)
                continue

            converted = formula(chunk, param_table_name, use_density=use_density, density_fallback=density_fallback)

            # NaN from a non-NaN input marks a failed conversion (missing T/S)
            failed = converted[vc].isna() & chunk[vc].notna()
            if failed.any():
                if on_missing == "raise":
                    raise ValueError(f"{param_table_name}: {int(failed.sum())} rows could not be "
                                     f"converted from unit {int(uid)} "
                                     f"(missing temperature/salinity)")
                if on_missing == "keep":
                    converted.loc[failed] = chunk.loc[failed]
                else:
                    converted = converted[~failed]
                n_missing_input += int(failed.sum())
            parts.append(converted)

        # Report unconvertible rows
        action = "kept unconverted" if on_missing == "keep" else "dropped"
        if n_no_units_id:
            logger.warning("%s: %d rows without UNITS_ID %s", param_table_name, n_no_units_id, action)
        if n_no_formula:
            logger.warning("%s: %d rows with no conversion formula to unit %d %s",
                            param_table_name, n_no_formula, target, action)
        if n_missing_input:
            logger.warning("%s: %d rows missing temperature/salinity inputs %s",
                            param_table_name, n_missing_input, action)

        if not parts:
            return df.iloc[0:0]

        # Restore the original row order
        return pd.concat(parts).sort_index()

    def default_unit_id(self, param_table_name: str) -> int | None:
        """Return the ID of a parameter's default unit.

        Args:
            param_table_name (str): Table name (e.g. ``'P_OXYGEN'``).
        Returns:
            int or None: Unit ID, or ``None`` when no default unit is
                registered for *param_table_name*.
        """
        mask = self.df_default_units["NAME_TABLE"] == param_table_name
        if not mask.any():
            return None
        return int(self.df_default_units.loc[mask, "UNITS_ID_DEFAULT"].iloc[0])

    def required_inputs(self, param_table_name: str, use_density: bool = False) -> tuple[bool, bool]:
        """Returns which co-located measurements a conversions potentially need.

        Combines requirements of all conversions into the parameter's default unit
        (see :meth:`ConversionFormulas.is_allowed`).

        Args:
            param_table_name (str): Table name (e.g. ``'P_OXYGEN'``).
            use_density (bool): If densities are computed (``True``) or the constant is used.
        Returns:
            tuple[bool, bool]: ``(needs_temperature, needs_salinity)``.
        """
        target = self.default_unit_id(param_table_name)  # default unit ID
        needs: set[str] = set()  # list of required co-located parameters

        # Iterate over all potential targets
        if target is not None:
            # Iterate over all conversions with default unit ID as target
            for source, conversion_target in self.formulas.conversions:
                # Check if the formula is allowed for the parameter
                if conversion_target == target and self.formulas.is_allowed(source, target, param_table_name):
                    # Check if the conversion requires temperature and/or salinity
                    needs |= self.formulas.t_s_inputs_needed(source, target, param_table_name, use_density)
        return "temperature" in needs, "salinity" in needs

    def default_unit_name(self, param_table_name: str) -> str | None:
        """Return the human-readable name of a parameter's default unit.

        Args:
            param_table_name (str): Table name (e.g. ``'P_OXYGEN'``).
        Returns:
            str or None: Unit name (e.g. ``'µmol/kg'``), or ``None`` when
                no default unit is registered for *param_table_name*.
        """
        # Get target unit ID
        target = self.default_unit_id(param_table_name)
        if target is None:
            return None

        # Get the short name of the unit ID
        row = self.df_units.loc[self.df_units["ID"] == target, "NAME_SHORT"]
        return row.iloc[0] if not row.empty else None

    def convert_units(self, tables: list[str], use_density: bool = False,
                      override_old_tables: bool = False,
                      on_missing: str = "drop",
                      density_fallback: str = "constant") -> dict[str, str]:
        """Convert units for the given parameter tables to their default unit.

        Args:
            tables (list[str]): Table names to convert.
            use_density (bool): Use in-situ density.
            override_old_tables (bool): Replace the original table.
            on_missing (str): Policy for unconvertible rows
                (``'drop'``, ``'keep'`` or ``'raise'``). Default ``'drop'``.
            density_fallback (str): Policy for rows without computable density
                (``'constant'``, ``'drop'`` or ``'raise'``). Default ``'constant'``.
        Returns:
            dict[str, str]: Maps each input table name to the table holding
                the converted data (the input name itself when nothing changed).
        """
        # Iterate over tables
        new_tables = {}
        for table in tables:
            validate_identifier(table)
            logger.info("UnitsConverter: converting %s", table)

            # Get data as df
            ex = self.connection.cursor().execute(f"SELECT * FROM {table};")
            df = pd.DataFrame(ex.fetchall(), columns=[x[0] for x in ex.description])
            old_len = len(df)

            # Convert units
            df_new = self.convert_dataframe(df, table, use_density=use_density,
                                            on_missing=on_missing, density_fallback=density_fallback)

            # Write converted data only when something changed
            changed = ("UNITS_ID" in df_new.columns
                       and (len(df_new) != old_len or not df_new["UNITS_ID"].equals(df["UNITS_ID"])))
            if changed:
                new_table_name = table if override_old_tables else f"converted_{table}"
                if_exists = "replace" if override_old_tables else "fail"
                df_new.to_sql(new_table_name, self.connection, if_exists=if_exists, index=False)
                logger.info("  rows removed during conversion: %d", old_len - len(df_new))
                new_tables[table] = new_table_name
            else:
                new_tables[table] = table
        return new_tables


class ConversionFormulas:
    """
    Registry of callables to convert between COMFORT units, keyed by
    (source_unit_id, target_unit_id). Based on COMFORT dataset v3 report, Appendix C.

    With ``use_density=True`` the report's density conventions apply:
    ``lab_dens`` (T = 22 °C) for most volumetric conversions, ``real_dens``
    (in-situ T) for chlorophyll and oxygen mL/L (both TEOS-10 at atmospheric
    pressure). Otherwise, the constant 1.025 kg/L is used.
    % saturation -> µmol/kg (oxygen) is a comfort-db extension beyond the
    report based on Benson & Krause 1984.
    """

    def __init__(self):
        # Molar masses for mass-to-molar conversions (COMFORT v3, Appendix C)
        self.molar_masses = pd.DataFrame(
            {"NAME_TABLE": ["P_BARIUM", "P_NITRATE", "P_NITRATENITRITE", "P_NITRITE",
                            "P_PHOSPHATE", "P_SILICATE"],
             "ELEMENT": ["BARIUM", "NITROGEN", "NITROGEN", "NITROGEN", "PHOSPHORUS", "SILICON"],
             "MOLAR_MASS": [137.327, 14.00672, 14.00672, 14.00672, 30.973762, 28.085530]}
        )

        # Known conversions: (source_unit_id, target_unit_id) -> formula
        self.conversions = {
            (5, 3): self.milliEquivalentPerLiter_micromolPerKilogram,
            (4, 14): self.microgramPerLiter_microgramPerKilogram,
            (7, 3): self.millimolPerLiter_micromolPerKilogram,
            (12, 19): self.nanomolPerKilogram_femtomolPerKilogram,
            (21, 3): self.milliliterPerLiter_micromolPerKilogram,
            (14, 3): self.microgramPerKilogram_micromolPerKilogram,
            (4, 3): self.microgramPerLiter_micromolPerKilogram,
            (26, 3): self.microgramAtomPerKilogram_micromolPerKilogram,
            (10, 3): self.percent_micromolPerKilogram,
        }

        # Parameter tables each conversion may be applied to (not-listed pairs have no restrictions)
        molar_mass_tables = frozenset(self.molar_masses["NAME_TABLE"])
        self.allowed_parameters = {
            (5, 3): frozenset({"P_ALKALINITY"}),
            (7, 3): frozenset({"P_DIC", "P_BARIUM"}),
            (21, 3): frozenset({"P_OXYGEN"}),
            (10, 3): frozenset({"P_OXYGEN"}),
            (4, 14): frozenset({"P_AMMONIUM", "P_DIN", "P_CHLOROPHYLL"}),
            (4, 3): molar_mass_tables,
            (14, 3): molar_mass_tables,
            (26, 3): molar_mass_tables,
        }

    def t_s_inputs_needed(self, source_unit_id: int, target_unit_id: int, param_table_name: str,
                          use_density: bool = False) -> set[str]:
        """Check if the conversion needs co-located temperature and/or salinity measurements.
        Depends on whether density should be used (lab_dens needs salinity, real_dens needs both)

        Args:
            source_unit_id (int): Unit ID of the values.
            target_unit_id (int): Unit ID to convert to.
            param_table_name (str): Table name (e.g. ``'P_CHLOROPHYLL'``), case-insensitive.
            use_density (bool): Whether densities are computed (``True``) or the constant is used.
        Returns:
            set[str]: Subset of ``{'temperature', 'salinity'}``.
        """
        pair = (source_unit_id, target_unit_id)
        if pair == (10, 3):
            # Requires T, S (independent of density-setting)
            return {"temperature", "salinity"}
        if not use_density:
            # No density for conversions -> No T, S needed
            return set()
        if pair == (21, 3) or (pair == (4, 14) and param_table_name.upper().endswith("CHLOROPHYLL")):
            # Conversions using real_dens
            return {"temperature", "salinity"}
        if pair in ((5, 3), (7, 3), (4, 3), (4, 14)):
            # Conversions using lab_dens
            return {"salinity"}
        return set()

    def is_allowed(self, source_unit_id: int, target_unit_id: int, param_table_name: str) -> bool:
        """Check if a registered conversion may be applied to a parameter.

        Args:
            source_unit_id (int): Unit ID of the values.
            target_unit_id (int): Unit ID to convert to.
            param_table_name (str): Table name (e.g. ``'P_OXYGEN'``), case-insensitive.
        Returns:
            bool: ``True`` if the conversion has no parameter restriction or lists *param_table_name*.
        """
        allowed = self.allowed_parameters.get((source_unit_id, target_unit_id))  # consult the valid-parameter map
        return allowed is None or param_table_name.upper() in allowed  # check if passed param is contained

    def get_molar_mass(self, param_table_name: str) -> float:
        """Return the molar mass for param_table_name."""
        mask = self.molar_masses["NAME_TABLE"] == param_table_name
        if not mask.any():
            raise ValueError(f"Molar mass not defined for {param_table_name}")
        return float(self.molar_masses.loc[mask, "MOLAR_MASS"].iloc[0])

    def milliEquivalentPerLiter_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                                    use_density: bool = False,
                                                    density_fallback: str = "constant") -> pd.DataFrame:
        """Convert meq/L to umol/kg (App. C: lab_dens). Valid for alkalinity."""
        logger.debug("milliEquivalentPerLiter -> micromolPerKilogram")
        temp, lab_dens = self._with_density(df, self.lab_density, use_density=use_density, density_fallback=density_fallback)
        return temp.assign(VAL=temp["VAL"] * 1000 / lab_dens, UNITS_ID=3)

    def microgramPerLiter_microgramPerKilogram(self, df: pd.DataFrame, param_name: str,
                                               use_density: bool = False,
                                               density_fallback: str = "constant") -> pd.DataFrame:
        """Convert ug/L to ug/kg (App. C: real_dens for chlorophyll, lab_dens otherwise)."""
        logger.debug("microgramPerLiter -> microgramPerKilogram")
        density_fn = (self.real_density if param_name.upper().endswith("CHLOROPHYLL") else self.lab_density)
        temp, dens = self._with_density(df, density_fn, use_density=use_density, density_fallback=density_fallback)
        return temp.assign(VAL=temp["VAL"] / dens, UNITS_ID=14)

    def millimolPerLiter_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                             use_density: bool = False,
                                             density_fallback: str = "constant") -> pd.DataFrame:
        """Convert mmol/L to umol/kg (App. C: lab_dens)."""
        logger.debug("millimolPerLiter -> micromolPerKilogram")
        temp, lab_dens = self._with_density(df, self.lab_density, use_density=use_density, density_fallback=density_fallback)
        return temp.assign(VAL=temp["VAL"] * 1000 / lab_dens, UNITS_ID=3)

    def nanomolPerKilogram_femtomolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                               use_density: bool = False,
                                               density_fallback: str = "constant") -> pd.DataFrame:
        """Convert nmol/kg to fmol/kg."""
        logger.debug("nanomolPerKilogram -> femtomolPerKilogram")
        return df.copy().assign(VAL=df["VAL"] * 1_000_000, UNITS_ID=19)

    def milliliterPerLiter_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                               use_density: bool = False,
                                               density_fallback: str = "constant") -> pd.DataFrame:
        """Convert mL/L to umol/kg (oxygen-specific, factor 44.661; App. C: real_dens)."""
        logger.debug("milliliterPerLiter -> micromolPerKilogram (oxygen-specific, factor 44.661)")
        temp, real_dens = self._with_density(df, self.real_density, use_density=use_density, density_fallback=density_fallback)
        return temp.assign(VAL=temp["VAL"] * 44.661 / real_dens, UNITS_ID=3)

    def microgramPerKilogram_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                                 use_density: bool = False,
                                                 density_fallback: str = "constant") -> pd.DataFrame:
        """Convert ug/kg to umol/kg using molar mass."""
        logger.debug("microgramPerKilogram -> micromolPerKilogram")
        temp = df.copy()
        molar_mass = self.get_molar_mass(param_name)
        return temp.assign(VAL=temp["VAL"] / molar_mass, UNITS_ID=3)

    def microgramPerLiter_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                              use_density: bool = False,
                                              density_fallback: str = "constant") -> pd.DataFrame:
        """Convert ug/L to umol/kg using molar mass (App. C: lab_dens)."""
        logger.debug("microgramPerLiter -> micromolPerKilogram")
        temp, lab_dens = self._with_density(df, self.lab_density, use_density=use_density, density_fallback=density_fallback)
        molar_mass = self.get_molar_mass(param_name)
        return temp.assign(VAL=temp["VAL"] / (molar_mass * lab_dens), UNITS_ID=3)

    def microgramAtomPerKilogram_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                                     use_density: bool = False,
                                                     density_fallback: str = "constant") -> pd.DataFrame:
        """Convert ugAtom/kg to umol/kg (1:1 equivalence)."""
        logger.debug("microgramAtomPerKilogram -> micromolPerKilogram")
        return df.copy().assign(UNITS_ID=3)

    def percent_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                    use_density: bool = False,
                                    density_fallback: str = "constant") -> pd.DataFrame:
        """Convert % saturation to umol/kg using TEOS-10 oxygen solubility.

        Requires co-located temperature and salinity columns (any casing).
        Potential temperature is computed from co-located pressure (or depth)/latitude/
        longitude when available (see :meth:`_potential_temperature_for_o2sol`), otherwise falls
        back to in-situ temperature as an approximation.
        """
        logger.debug("percent -> micromolPerKilogram (gsw.O2sol_SP_pt)")
        temp = df.copy()
        sal_col = _get_column(df, "SALINITY")
        temp_col = _get_column(df, "TEMPERATURE")

        # Check if salinity and temperature columns are given
        if sal_col is None or temp_col is None:
            logger.warning("percent -> micromolPerKilogram: temperature/salinity columns "
                            "missing - cannot compute oxygen saturation")
            return temp.assign(VAL=np.nan, UNITS_ID=3)

        # Unit conversion
        pt = self._potential_temperature_for_o2sol(df, sal_col, temp_col)
        o2_eq = gsw.O2sol_SP_pt(df[sal_col].to_numpy(dtype=float), pt)
        return temp.assign(VAL=temp["VAL"] * np.asarray(o2_eq) / 100, UNITS_ID=3)

    def _potential_temperature_for_o2sol(self, df: pd.DataFrame, sal_col: str,
                                         temp_col: str) -> np.ndarray:
        """Helper function to compute potential temperature (pt0) for oxygen solubility.

        If pressure is not given, it computes pressure from depth and latitude if both are available.
        Falls back to in-situ temperature when pressure/latitude/longitude are unavailable.
        """
        lat_col = _get_column(df, "LATITUDE")
        lon_col = _get_column(df, "LONGITUDE")

        # Resolve pressure (without latitude, the in-situ temperature fallback applies)
        pressure = resolve_pressure(df) if lat_col is not None else None

        if pressure is None or pressure.isna().all() or lon_col is None:
            # Pressure, lat or lon missing
            logger.warning("percent -> micromolPerKilogram: pressure/latitude/longitude "
                            "missing, using in-situ temperature as an approximation to "
                            "potential temperature")
            return df[temp_col].to_numpy(dtype=float)

        # Compute absolute salinity and potential temperature
        sa = convert_salinity(df[sal_col], pressure, df[lon_col], df[lat_col])
        pt = convert_temperature(df[temp_col], sa.to_numpy(), pressure, to="pt0")
        return pt.to_numpy()

    def _with_density(self, df: pd.DataFrame, density_fn, use_density: bool,
                      density_fallback: str) -> tuple[pd.DataFrame, float | pd.Series]:
        """Return (df, density) with rows lacking a density removed (``'drop'`` policy)."""
        dens = density_fn(df, use_density=use_density, density_fallback=density_fallback)
        if isinstance(dens, pd.Series):
            not_nan_dens = dens.notna()

            # Check if there are any not-nan densities
            if not not_nan_dens.all():
                return df[not_nan_dens].copy(), dens[not_nan_dens]

        # If no not-nan densities, return all data
        return df.copy(), dens

    def lab_density(self, df: pd.DataFrame, use_density: bool,
                    density_fallback: str = "constant") -> float | pd.Series:
        """Density at sample salinity, T = 22 °C, atmospheric pressure [kg/L],
        see ``lab_dens`` of COMFORT report Appendix C.
        """
        return self._density(df, use_density=use_density, lab_temperature=22.0, density_fallback=density_fallback)

    def real_density(self, df: pd.DataFrame, use_density: bool,
                     density_fallback: str = "constant") -> float | pd.Series:
        """Density at sample salinity, in-situ temperature, atmospheric pressure [kg/L],
        see ``real_dens`` of COMFORT report Appendix C.
        """
        return self._density(df, use_density=use_density, lab_temperature=None, density_fallback=density_fallback)

    def _density(self, df: pd.DataFrame, use_density: bool,
                 lab_temperature: float | None, density_fallback: str) -> float | pd.Series:
        """TEOS-10 density [kg/L] at sea pressure 0, column names resolved case-insensitively.

        With ``use_density=False`` the COMFORT report's fixed factor 1.025 kg/L is returned.
        Otherwise, densities are computed and for non-computable densities, the fallback
        policy (``density_fallback``) is applied.
        """
        if density_fallback not in DENSITY_FALLBACKS:
            # Ensure valid density fallback
            raise ValueError(f"density_fallback must be one of {DENSITY_FALLBACKS}, "
                             f"got {density_fallback!r}")
        if not use_density:
            return 1.025

        # Get column names
        sal_col = _get_column(df, "SALINITY")
        lat_col = _get_column(df, "LATITUDE")
        lon_col = _get_column(df, "LONGITUDE")
        needed = [("salinity", sal_col), ("latitude", lat_col), ("longitude", lon_col)]
        temp_col = None
        if lab_temperature is None:
            temp_col = _get_column(df, "TEMPERATURE")
            needed.append(("temperature", temp_col))

        # Check if any column required for density computations is missing
        missing = [needed_col_name for needed_col_name, col in needed if col is None]
        if missing:
            # If any column is missing, set densities to NaN
            reason = f"columns missing ({', '.join(missing)})"
            rho = pd.Series(np.nan, index=df.index)
        else:
            # If all columns are there, compute densities
            temperature = lab_temperature if lab_temperature is not None else df[temp_col]
            salinity_absolute = gsw.SA_from_SP(df[sal_col], 0.0, df[lon_col], df[lat_col])
            rho = pd.Series(np.asarray(gsw.density.rho_t_exact(salinity_absolute, temperature, 0.0)) / 1000, index=df.index)
            reason = "NaN salinity/temperature/position"  # NaN densities can arise for single rows

        # Check if any density is NaN/undefined
        nan_density_rows = rho.isna()
        if not nan_density_rows.any():
            return rho

        # If NaN densities exist, apply density_fallback policy
        n = int(nan_density_rows.sum())
        if density_fallback == "raise":
            # Raise error
            raise ValueError(f"density: {n} of {len(df)} rows undefined - {reason}")
        # Use constant density
        action = "using constant density 1.025 kg/L" if density_fallback == "constant" else "dropped"
        logger.warning("density: %d of %d rows undefined (%s) - %s", n, len(df), reason, action)
        return rho.fillna(1.025) if density_fallback == "constant" else rho
