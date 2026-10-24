"""TEOS-10 physical oceanography functions for COMFORT data.

Conversions are computed using the `gsw` library, the Python implementation of the
TEOS-10 Gibbs Seawater Oceanographic Toolbox (McDougall, T., & Barker, P. (2011).
Getting started with TEOS-10 and the Gibbs Seawater (GSW) Oceanographic Toolbox. SCOR/IAPSO WG127).

"""
from __future__ import annotations

__all__ = [
    "add_teos10_variables",
    "compute_aou",
    "compute_buoyancy_frequency",
    "compute_density",
    "compute_potential_vorticity",
    "compute_spiciness",
    "convert_salinity",
    "convert_temperature",
    "convert_to_potential_temperature",
    "resolve_pressure",
]

import logging
import gsw
import numpy as np
import pandas as pd

from .profile_analysis import _empty_profile_frame, _resolve_profile_col, _profile_key_dict
from .util.columns import find_col
from .util.types import DensityQuantity, TemperatureTarget, check_choice

logger = logging.getLogger(__name__)


def _to1d(x):
    """Convert x to a 1D float64 numpy array."""
    return np.atleast_1d(np.asarray(x, dtype=float))


def resolve_pressure(df: pd.DataFrame, pressure_col: str = "LEV_DBAR",
                     depth_col: str = "LEV_M", lat_col: str = "LATITUDE") -> pd.Series:
    """Resolve sea pressure [dbar] for every row using the best information available, in this order:

    1. The measured pressure (*pressure_col*, if not NaN),
    2. Pressure computed from depth and latitude (``gsw.p_from_z``),
    3. Depth in metres itself as a proxy for dbar (if no latitude is available),
    4. NaN.

    Depths more than 5 m above the sea surface are treated as invalid.

    Args:
        df (pandas.DataFrame): DataFrame with any of the pressure, depth and latitude columns (case-insensitive).
        pressure_col (str): Column with measured sea pressure [dbar].
        depth_col (str): Column with depth [m, positive down].
        lat_col (str): Column with latitude [°N].
    Returns:
        pandas.Series: Sea pressure [dbar].
    """
    # Get column names for pressure, depth and latitude
    p_col = find_col(df, pressure_col, required=False)
    d_col = find_col(df, depth_col, required=False)
    lat = find_col(df, lat_col, required=False)

    # Use pressure column if available
    pressure = df[p_col].astype(float) if p_col else pd.Series(np.nan, index=df.index)

    # Check if depth column is given
    if d_col is not None:
        # gsw.p_from_z needs height (negative down) and raises for z > 5 m
        depth = df[d_col].astype(float)
        depth = depth.where(depth >= -5)

        # Pressure from depth and latitude
        if lat is not None:
            from_depth = pd.Series(gsw.p_from_z(-depth.to_numpy(), df[lat].to_numpy(dtype=float)), index=df.index)
            pressure = pressure.fillna(from_depth)

        # Depth as proxy for pressure
        proxy = pressure.isna() & depth.notna()
        if proxy.any():
            logger.warning("resolve_pressure: %d of %d rows without pressure and latitude, "
                            "using depth [m] as a proxy for dbar", int(proxy.sum()), len(df))
            pressure = pressure.fillna(depth)

    # Count the remaining number of NaN pressure values
    n_nan = int(pressure.isna().sum())
    if n_nan:
        logger.warning("resolve_pressure: %d of %d rows have no pressure information (NaN)", n_nan, len(df))
    return pressure


def convert_salinity(sp: float | np.ndarray | pd.Series,
                     pressure: float | np.ndarray | pd.Series,
                     longitude: float | np.ndarray | pd.Series,
                     latitude: float | np.ndarray | pd.Series) -> pd.Series:
    """Convert Practical Salinity to Absolute Salinity (TEOS-10).

    Args:
        sp (array-like): Practical Salinity [PSS-78].
        pressure (array-like): Sea pressure [dbar] (0 dbar at the ocean surface).
        longitude (array-like): Longitude [°East].
        latitude (array-like): Latitude [°North].
    Returns:
        pandas.Series: Absolute Salinity SA [g/kg].
    """
    return pd.Series(gsw.SA_from_SP(_to1d(sp), _to1d(pressure),
                                    _to1d(longitude), _to1d(latitude)))


def convert_temperature(t_insitu: float | np.ndarray | pd.Series,
                        sa: float | np.ndarray | pd.Series,
                        pressure: float | np.ndarray | pd.Series,
                        to: TemperatureTarget = "CT") -> pd.Series:
    """Convert in-situ temperature to a TEOS-10 temperature variable.

    Args:
        t_insitu (array-like): In-situ temperature [°C, ITS-90].
        sa (array-like): Absolute Salinity [g/kg]
            (from :func:`convert_salinity`).
        pressure (array-like): Sea pressure [dbar].
        to (str): Target variable:
            ``"CT"`` - Conservative temperature [°C];
            ``"pt0"`` - Potential temperature ref. 0 dbar [°C].
    Returns:
        pandas.Series
    Raises:
        ValueError: If ``to`` is not ``'CT'`` or ``'pt0'``.
    """
    check_choice(to, TemperatureTarget, "to")

    # Prepare variables
    t = _to1d(t_insitu)
    sa_ = _to1d(sa)
    p = _to1d(pressure)

    # Convert
    if to == "CT":
        return pd.Series(gsw.CT_from_t(sa_, t, p))
    return pd.Series(gsw.pt0_from_t(sa_, t, p))


def compute_density(sa: float | np.ndarray | pd.Series,
                    ct: float | np.ndarray | pd.Series,
                    pressure: float | np.ndarray | pd.Series | None = None,
                    quantity: DensityQuantity = "sigma0") -> pd.Series:
    """Compute seawater density or potential density anomaly (TEOS-10).

    Args:
        sa (array-like): Absolute Salinity [g/kg].
        ct (array-like): Conservative Temperature [°C].
        pressure (array-like): Sea pressure [dbar]. Required only for
            ``quantity='rho'``, ignored otherwise.
        quantity (str): What to compute:
            ``"sigma0"`` - potential density anomaly [kg/m3], ref. 0 dbar;
            ``"sigma1"`` - [kg/m3], ref. 1000 dbar;
            ``"sigma2"`` - [kg/m3], ref. 2000 dbar;
            ``"rho"`` - in-situ density [kg/m3].
    Returns:
        pandas.Series
    Raises:
        ValueError: If ``quantity`` is unknown or pressure is missing
            for ``'rho'``.
    """
    check_choice(quantity, DensityQuantity, "quantity")

    # Prepare variables
    sa_ = _to1d(sa)
    ct_ = _to1d(ct)

    # Compute density based on sigma type
    if quantity == "sigma0":
        return pd.Series(gsw.sigma0(sa_, ct_))
    if quantity == "sigma1":
        return pd.Series(gsw.sigma1(sa_, ct_))
    if quantity == "sigma2":
        return pd.Series(gsw.sigma2(sa_, ct_))

    # Compute density from pressure
    if pressure is None:
        raise ValueError("pressure is required for quantity='rho'")
    return pd.Series(gsw.rho(sa_, ct_, _to1d(pressure)))


def compute_buoyancy_frequency(df: pd.DataFrame, sa_col: str, ct_col: str,
                               pressure_col: str = "LEV_DBAR",
                               lat_col: str = "LATITUDE",
                               profile_col: str | list[str] | None = None,
                               ) -> pd.DataFrame:
    """Compute Brunt-Vaisala (buoyancy) frequency N2 for each profile.

    N2 > 0 indicates stable stratification.
    N2 < 0 indicates gravitational instability.

    Args:
        df (pandas.DataFrame): Wide-format DataFrame with SA and CT columns.
        sa_col (str): Column with Absolute Salinity [g/kg].
        ct_col (str): Column with Conservative Temperature [°C].
        pressure_col (str): Column with sea pressure [dbar].
        profile_col (str | list[str] | None): Column(s) identifying individual profiles.
        lat_col (str): Latitude column [°N] for gravitational
            acceleration correction. Falls back to 0 if absent.
    Returns:
        pandas.DataFrame: One row per consecutive level pair per profile
            with the profile key columns (``ID`` and ``PROFILE_NUMBER``
            by default), ``'pressure_mid'`` and ``'N2'`` (N2 in 1/s2).
    """
    # Resolve column names (insensitive to casing)
    sa_col = find_col(df, sa_col)
    ct_col = find_col(df, ct_col)
    pressure_col = find_col(df, pressure_col)

    lat_col = find_col(df, lat_col, required=False)
    lat_available = lat_col is not None
    rows = []

    # Get group columns
    group_cols = _resolve_profile_col(df, profile_col=profile_col)

    # Iterate over profiles
    for pid, group in df.groupby(group_cols):
        # Check if profile has min. 2 levels
        cols = [pressure_col, sa_col, ct_col] + ([lat_col] if lat_available else [])
        valid = group[cols].dropna().sort_values(pressure_col)
        if len(valid) < 2:
            logger.debug("compute_buoyancy_frequency: profile %s has < 2 levels, skipped", pid)
            continue

        # Fall back to equatorial gravity if latitude not available
        lat = valid[lat_col].values if lat_available else np.zeros(len(valid))

        # N2 is computed at mid-depth pressure levels between consecutive pairs
        n2, p_mid = gsw.Nsquared(
            SA=valid[sa_col].values,
            CT=valid[ct_col].values,
            p=valid[pressure_col].values,
            lat=lat,
        )
        for val, pmid in zip(n2, p_mid):
            rows.append({**_profile_key_dict(profile_col=group_cols, pid=pid), "pressure_mid": float(pmid), "N2": float(val)})

    if not rows:
        # Return empty df
        return _empty_profile_frame(group_cols, ["pressure_mid", "N2"])
    return pd.DataFrame(rows)


def compute_potential_vorticity(n2: float | np.ndarray | pd.Series,
                                latitude: float | np.ndarray | pd.Series) -> pd.Series:
    """Compute the planetary (linear) potential vorticity PV = f * N2 / g.

    This is Ertel's potential vorticity theorem with relative vorticity
    neglected as e.g. used in Johnson, G. C., 2006: Generation and Initial
    Evolution of a Mode Water θ–S Anomaly. J. Phys. Oceanogr., 36, 739–751,
    https://doi.org/10.1175/JPO2895.1.

    Args:
        n2 (array-like): Buoyancy frequency squared [1/s2], e.g. from
            :func:`compute_buoyancy_frequency`.
        latitude (array-like): Latitude [°N]. Broadcast against *n2*,
            e.g. pass the same latitude for every level of one profile.
    Returns:
        pandas.Series: Potential vorticity [1/(m s)].
    """
    lat_ = _to1d(latitude)
    f = gsw.f(lat_)  # Coriolis parameter
    g = gsw.grav(lat_, np.zeros_like(lat_))  # Gravitational acceleration
    return pd.Series(f * _to1d(n2) / g)


def compute_spiciness(sa: float | np.ndarray | pd.Series,
                      ct: float | np.ndarray | pd.Series,
                      reference_pressure: int = 0) -> pd.Series:
    """Compute oceanographic spiciness as a water-mass tracer (TEOS-10).

    Spiciness increases with temperature and salinity along isopycnals:
    high = warm and salty; low = cold and fresh at the same density.
    Use alongside sigma0 in T-S space to identify water masses.

    Args:
        sa (array-like): Absolute Salinity [g/kg].
        ct (array-like): Conservative Temperature [°C].
        reference_pressure (int): Reference pressure in dbar:
            ``0`` - upper ocean (spiciness0);
            ``2000`` - deep ocean (spiciness2).
    Returns:
        pandas.Series: Spiciness [kg/m3].
    Raises:
        ValueError: If ``reference_pressure`` is not 0 or 2000.
    """
    # Prepare variables
    sa_ = _to1d(sa)
    ct_ = _to1d(ct)

    # Return spiciness according to reference pressure
    if reference_pressure == 0:
        return pd.Series(gsw.spiciness0(sa_, ct_))
    if reference_pressure == 2000:
        return pd.Series(gsw.spiciness2(sa_, ct_))
    raise ValueError(
        f"reference_pressure must be 0 or 2000; got {reference_pressure}"
    )


def compute_aou(oxygen: float | np.ndarray | pd.Series,
                sp: float | np.ndarray | pd.Series,
                pt: float | np.ndarray | pd.Series) -> pd.Series:
    """Compute Apparent Oxygen Utilization (AOU), i.e. the difference between
    how much oxygen the water could hold and the actual oxygen concentration.

    Args:
        oxygen (array-like): Observed dissolved oxygen concentration [umol/kg].
        sp (array-like): Practical Salinity.
        pt (array-like): Potential temperature referenced to 0 dbar [°C].
    Returns:
        pandas.Series: AOU [umol/kg].
    """
    o2_eq = gsw.O2sol_SP_pt(_to1d(sp), _to1d(pt))
    return pd.Series(o2_eq - _to1d(oxygen))


def add_teos10_variables(df: pd.DataFrame, sp_col: str, t_col: str,
                         pressure_col: str = "LEV_DBAR",
                         lon_col: str = "LONGITUDE",
                         lat_col: str = "LATITUDE") -> pd.DataFrame:
    """Add SA, CT and sigma0 columns to a DataFrame in a single call.

    Convenience wrapper around :func:`convert_salinity`,
    :func:`convert_temperature` and :func:`compute_density`.

    Args:
        df (pandas.DataFrame): DataFrame with Practical Salinity, in-situ
            temperature, sea pressure, longitude and latitude columns.
        sp_col (str): Column with Practical Salinity [PSS-78].
        t_col (str): Column with in-situ temperature [°C, ITS-90].
        pressure_col (str): Column with sea pressure [dbar].
        lon_col (str): Column with longitude [°East].
        lat_col (str): Column with latitude [°North].
    Returns:
        pandas.DataFrame: Copy of *df* with three added columns:
            ``SA`` [g/kg], ``CT`` [°C], ``sigma0`` [kg/m3].
    """
    out = df.copy()

    # Resolve column names (case-insensitive matching)
    sp = find_col(out, sp_col)
    t = find_col(out, t_col)
    p = find_col(out, pressure_col)
    lon = find_col(out, lon_col)
    lat = find_col(out, lat_col)

    # Convert step by step: SP -> SA -> CT -> sigma0
    out["SA"] = convert_salinity(out[sp], out[p], out[lon], out[lat]).values
    out["CT"] = convert_temperature(out[t], out["SA"], out[p], to="CT").values
    out["sigma0"] = compute_density(out["SA"], out["CT"], quantity="sigma0").values
    return out


def convert_to_potential_temperature(
    averaged: dict[str, pd.DataFrame],
    temperature_col: str = "TEMPERATURE",
    salinity_col: str = "SALINITY",
    loc_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Convert in-situ temperature to potential temperature (gsw).

    Needs co-located salinity and pressure (``LEV_DBAR``).

    Args:
        averaged (dict[str, pandas.DataFrame]): Per-parameter DataFrames. Must
            contain *temperature_col* (with a ``LEV_DBAR`` column) and
            *salinity_col*.
        temperature_col (str): Key for temperature. Default ``"TEMPERATURE"``.
        salinity_col (str): Key for salinity. Default ``"SALINITY"``.
        loc_cols (list[str]): Columns identifying a unique location. Default
            ``["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"]``.
    Returns:
        dict[str, pandas.DataFrame]: Copy of *averaged* with *temperature_col*
            replaced by potential temperature.
    Raises:
        KeyError: If *temperature_col* or *salinity_col* is missing from *averaged*.
    """
    # Define location columns
    if loc_cols is None:
        loc_cols = ["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"]

    # Check if temperature and salinity DataFrames are passed
    if temperature_col not in averaged or salinity_col not in averaged:
        raise KeyError(
            f"convert_to_potential_temperature requires both {temperature_col!r} "
            f"and {salinity_col!r} in `averaged`"
        )

    # Merge temperature and salinity DataFrames
    df_t = averaged[temperature_col].merge(
        averaged[salinity_col][loc_cols + [salinity_col]], on=loc_cols, how="left",
    )

    # Convert to absolute salinity
    pressure = df_t["LEV_DBAR"].values
    sa = convert_salinity(df_t[salinity_col], pressure, df_t["LONGITUDE"], df_t["LATITUDE"])

    # Convert to potential temperature
    df_t[temperature_col] = convert_temperature(df_t[temperature_col], sa, pressure, to="pt0").values

    # Assemble and return results
    result = dict(averaged)
    result[temperature_col] = df_t[loc_cols + ["LEV_DBAR", temperature_col]]
    return result
