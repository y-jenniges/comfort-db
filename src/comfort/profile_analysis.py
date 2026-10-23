"""Profile analysis functions for COMFORT data."""
from __future__ import annotations

__all__ = [
    "average_duplicate_records_per_profile",
    "detect_depth_col",
    "interpolate_depth_levels",
    "mixed_layer_depth",
    "profile_completeness",
    "pycnocline_depth",
    "vertical_gradient",
]

import logging
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

# Columns that are never the measured-value column
_KNOWN_NON_VALUE_COLS = frozenset({
    "PROFILE_NUMBER", "LEV_DBAR", "LEV_M", "LATITUDE", "LONGITUDE",
    "DATEANDTIME", "ID", "PQF1", "PQF2", "SQF", "BOTTLE_NUMBER",
    "PROFILE_BEST", "UNITS_ID", "INSTRUMENT_ID", "SA", "CT", "sigma0",
})


def _resolve_profile_col(df: pd.DataFrame, profile_col: str | list[str] | None) -> str | list[str]:
    """Return profile grouping column(s). Defaults to [ID, PROFILE_NUMBER] when available."""
    # Use given profile_col
    if profile_col is not None:
        return profile_col

    # Use [ID, PROFILE_NUMBER] if available
    if "ID" in df.columns and "PROFILE_NUMBER" in df.columns:
        return ["ID", "PROFILE_NUMBER"]

    # Fallback
    return "PROFILE_NUMBER"


def _profile_key_dict(profile_col: str | list[str], pid) -> dict:
    """Map profile column name(s) to group key value(s)."""
    if isinstance(profile_col, list):
        return dict(zip(profile_col, pid))
    return {profile_col: pid}


def _empty_profile_frame(profile_col: str | list[str], columns: list[str]) -> pd.DataFrame:
    """Empty result df with the profile key column(s) followed by *columns*."""
    keys = profile_col if isinstance(profile_col, list) else [profile_col]
    return pd.DataFrame(columns=[*keys, *columns])


def _profile_levels(group: pd.DataFrame, depth_col: str, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the sorted unique depths of one profile and the mean value at each depth."""
    # Drop NaN rows
    levels = group[[depth_col, value_col]].dropna()

    # Average per (and sort by) depth
    levels = levels.groupby(depth_col)[value_col].mean()

    # Return depth-sorted arrays (depth, values)
    return levels.index.values, levels.values


def _detect_param_col(df, param_col):
    """Return param_col if it exists, otherwise auto-detect the value column."""
    # Use param_col
    if param_col in df.columns:
        return param_col

    # Infer potential candidates
    candidates = [c for c in df.columns if c not in _KNOWN_NON_VALUE_COLS]
    if len(candidates) == 1:
        logger.debug("_detect_param_col: auto-detected %r", candidates[0])
        return candidates[0]

    # Fallback
    return param_col


def detect_depth_col(df: pd.DataFrame, depth_col: str = "LEV_M") -> str:
    """Return a valid depth column name, falling back between LEV_M and LEV_DBAR.

    Args:
        df (pandas.DataFrame): DataFrame to inspect.
        depth_col (str): Preferred depth column. Default is ``"LEV_M"``.
    Returns:
        str: The depth column name that exists in *df*.
    """
    # Check if depth_col exists
    if depth_col in df.columns:
        return depth_col

    # Use fallback (if depth_col does not exist)
    fallback = "LEV_DBAR" if depth_col == "LEV_M" else "LEV_M"
    if fallback in df.columns:
        logger.debug("detect_depth_col: %r not found, using %r", depth_col, fallback)
        return fallback

    # Raise error if no depth column was found
    raise KeyError(
        f"Depth column {depth_col!r} not found in DataFrame. "
        f"Available columns: {list(df.columns)}"
    )


def average_duplicate_records_per_profile(
    raw: dict[str, pd.DataFrame],
    depth_col: str = "LEV_M",
    profile_col: str | list[str] | None = None,
    station_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Average duplicate records within the same profile at the same depth
    (e.g. repeated sensor readings within a single cast).

    Depths are matched by exact equality (no tolerance).

    Args:
        raw (dict[str, pandas.DataFrame]): Per-parameter DataFrames, as
            returned by ``load_comfort(as_xarray=False)``.
        depth_col (str): Column with depth. Default ``"LEV_M"``.
        profile_col (str or list<str>): Column(s) identifying a profile.
            Defaults to ``["ID", "PROFILE_NUMBER"]`` when both are present
            in the first DataFrame, else ``"PROFILE_NUMBER"``.
        station_cols (list[str]): Station-level metadata columns to keep.
            Defaults to whichever of ``["LATITUDE", "LONGITUDE", "DATEANDTIME", "LEV_DBAR"]``
            are present in the first DataFrame.
    Returns:
        dict[str, pandas.DataFrame]: One DataFrame per parameter, one row
            per (profile, depth).
    """
    if not raw:
        return {}

    # Resolve grouping columns from the first parameter's DataFrame
    first_df = next(iter(raw.values()))
    profile_col = _resolve_profile_col(first_df, profile_col)
    group_cols = (profile_col if isinstance(profile_col, list) else [profile_col]) + [depth_col]

    # Station metadata is constant within a group, keep it via "first"
    if station_cols is None:
        station_cols = [c for c in ("LATITUDE", "LONGITUDE", "DATEANDTIME", "LEV_DBAR")
                        if c in first_df.columns]

    # Average each parameter separately
    averaged = {}
    for param, df in raw.items():
        agg = {param: "mean"}
        for c in station_cols:
            if c in df.columns and c not in group_cols:
                agg[c] = "first"
        avg = df.groupby(group_cols, as_index=False).agg(agg)
        logger.info("average_duplicate_records_per_profile: %s %d -> %d", param, len(df), len(avg))
        averaged[param] = avg

    return averaged


def vertical_gradient(df: pd.DataFrame,
                      param_col: str = "VAL",
                      depth_col: str = "LEV_M",
                      profile_col: str | list[str] | None = None
                      ) -> pd.DataFrame:
    """Compute the vertical gradient of a parameter within each profile.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        param_col (str): Column with measured values.
        depth_col (str): Column with depth [m].
        profile_col (str or list<str>): Column(s) identifying individual profiles.
            Defaults to ``["ID", "PROFILE_NUMBER"]`` when both exist, else
            ``"PROFILE_NUMBER"``.
    Returns:
        pandas.DataFrame: Copy of *df* with added column
            ``<param_col>_gradient``. Repeated depths within a profile are
            averaged before computing the gradient.
    """
    profile_col = _resolve_profile_col(df, profile_col)
    param_col = _detect_param_col(df, param_col)
    depth_col = detect_depth_col(df, depth_col)
    sort_cols = (profile_col if isinstance(profile_col, list) else [profile_col]) + [depth_col]
    df_out = df.copy().sort_values(sort_cols)
    grad_col = f"{param_col}_gradient"
    df_out[grad_col] = np.nan

    # Compute gradient per profile using central differences
    for _, group in df_out.groupby(profile_col, sort=False):
        # Average duplicate depths, drop NaNs
        depths, vals = _profile_levels(group, depth_col, param_col)
        if len(depths) < 2:
            continue

        # Compute gradient and map it back to every row at that depth
        dv = pd.Series(np.gradient(vals, depths), index=depths)
        valid = group[[depth_col, param_col]].dropna()
        df_out.loc[valid.index, grad_col] = valid[depth_col].map(dv).values

    return df_out


def profile_completeness(df: pd.DataFrame,
                         depth_col: str = "LEV_M",
                         profile_col: str | list[str] | None = None,
                         reference_depths: ArrayLike | None = None,
                         tolerance: float = 5.0
                         ) -> pd.DataFrame:
    """Assess vertical sampling completeness of each profile.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        depth_col (str): Column with depth [m].
        profile_col (str or list<str>): Column(s) identifying individual profiles.
            Defaults to ``["ID", "PROFILE_NUMBER"]`` when both exist, else
            ``"PROFILE_NUMBER"``.
        reference_depths (array-like): Standard depth levels to measure
            coverage against. When provided, a ``coverage_fraction``
            column (0-1) is added.
        tolerance (float): Tolerance [m] for matching observed
            depths to reference levels. Default is 5m.
    Returns:
        pandas.DataFrame: One row per profile with columns
            [profile_col, n_levels, depth_min, depth_max, depth_range,
            coverage_fraction (optional)].
    """
    profile_col = _resolve_profile_col(df, profile_col)
    depth_col = detect_depth_col(df, depth_col)

    # Iterate over profiles
    rows = []
    for pid, group in df.groupby(profile_col):
        # Drop nans
        depths = group[depth_col].dropna().values

        # Assemble output row
        row = {
            **_profile_key_dict(profile_col, pid),
            "n_levels": len(depths),
            "depth_min": float(depths.min()) if len(depths) else np.nan,
            "depth_max": float(depths.max()) if len(depths) else np.nan,
            "depth_range": float(depths.max() - depths.min()) if len(depths) > 1 else 0.0,
        }

        # Check how many reference levels are covered within tolerance
        if reference_depths is not None:
            ref = np.asarray(reference_depths, dtype=float)
            covered = sum(
                np.any(np.abs(depths - r) <= tolerance) for r in ref
            )
            row["coverage_fraction"] = covered / len(ref) if len(ref) else np.nan
        rows.append(row)
    if not rows:
        # Return empty df
        return _empty_profile_frame(
            profile_col,
            ["n_levels", "depth_min", "depth_max", "depth_range"]
            + (["coverage_fraction"] if reference_depths is not None else []),
        )
    return pd.DataFrame(rows)


def mixed_layer_depth(df: pd.DataFrame,
                      density_col: str | None = None,
                      temp_col: str | None = None,
                      depth_col: str = "LEV_M",
                      profile_col: str | list[str] | None = None,
                      delta_density: float = 0.03,
                      delta_temp: float = 0.2,
                      reference_depth: float = 10.0,
                      max_ref_distance: float | None = 5.0
                      ) -> pd.DataFrame:
    """Estimate mixed layer depth (MLD) using the threshold method.
    If the threshold is not exceeded within the profile or no level lies close enough to the reference depth,
    there is no valid MLD and ``MLD_m`` is ``NaN``.

    Reference:
    De Boyer Montégut, C., Madec, G., Fischer, A. S., Lazar, A., & Iudicone, D. (2004).
    Mixed layer depth over the global ocean: An examination of profile data and
    a profile-based climatology. Journal of Geophysical Research: Oceans,
    109, C12003. https://doi.org/10.1029/2004JC002378

    Args:
        df (pandas.DataFrame): Input DataFrame with density or
            temperature and depth columns.
        density_col (str): Column with potential density anomaly [kg/m3].
        temp_col (str): Column with temperature [°C]. Used only
            when ``density_col`` is ``None``.
        depth_col (str): Column with depth [m].
        profile_col (str or list<str>): Column(s) identifying individual profiles.
            Defaults to ``["ID", "PROFILE_NUMBER"]`` when both exist, else
            ``"PROFILE_NUMBER"``.
        delta_density (float): Density threshold [kg/m3]. Default 0.03.
        delta_temp (float): Temperature threshold [°C]. Default 0.2.
        reference_depth (float): Reference depth [m]. The nearest
            measured depth is used when no exact match exists.
            Default 10.0m.
        max_ref_distance (float or None): Maximum allowed distance [m] between
            ``reference_depth`` and the nearest measured depth. ``None`` disables
            the check. Default 5.0m.
    Returns:
        pandas.DataFrame: One row per profile with columns
            [profile_col, ``'MLD_m'``] (``NaN`` when the threshold is never
            exceeded or the reference level is too far from ``reference_depth``).
    """
    # Get column names
    if density_col is None and temp_col is None:
        raise ValueError("Provide at least one of density_col or temp_col")

    profile_col = _resolve_profile_col(df, profile_col)
    depth_col = detect_depth_col(df, depth_col)

    # Select variable and threshold (density preferred over temperature)
    col = density_col if density_col is not None else temp_col
    delta = delta_density if density_col is not None else delta_temp

    # Density only counts an increase with depth, temperature either direction
    one_sided = density_col is not None

    # Iterate over profiles
    rows = []
    for pid, group in df.groupby(profile_col):
        # Average duplicate depths, drop NaNs
        depths, vals = _profile_levels(group, depth_col, col)
        if len(depths) < 2:
            logger.debug("mixed_layer_depth: profile %s has < 2 valid levels, skipped", pid)
            continue

        # Find reference value at nearest depth
        ref_idx = int(np.argmin(np.abs(depths - reference_depth)))
        v_ref = vals[ref_idx]

        # No usable reference level near the reference depth
        if max_ref_distance is not None and abs(depths[ref_idx] - reference_depth) > max_ref_distance:
            logger.debug("mixed_layer_depth: Profile %s has no level within %s m of %s m",
                          pid, max_ref_distance, reference_depth)
            rows.append({**_profile_key_dict(profile_col, pid), "MLD_m": np.nan})
            continue

        # Walk downward until threshold is exceeded
        mld = np.nan
        for i in range(ref_idx + 1, len(depths)):
            # Threshold test for density/temperature
            diff = vals[i] - v_ref
            if (diff if one_sided else abs(diff)) > delta:
                # Determine the crossed threshold (upper or lower)
                target = v_ref + delta if vals[i] > v_ref else v_ref - delta

                # Linear interpolation to estimate exact crossing depth
                mld = float(depths[i - 1] + (target - vals[i - 1]) * (depths[i] - depths[i - 1]) / (vals[i] - vals[i - 1]))
                break

        rows.append({**_profile_key_dict(profile_col, pid), "MLD_m": mld})

    if not rows:
        # Return empty df
        return _empty_profile_frame(profile_col, ["MLD_m"])
    return pd.DataFrame(rows)


def pycnocline_depth(df: pd.DataFrame,
                     density_col: str,
                     depth_col: str = "LEV_M",
                     profile_col: str | list[str] | None = None,
                     min_gradient: float = 0.0
                     ) -> pd.DataFrame:
    """Find the depth of maximum density increase with depth (pycnocline depth).

    Profiles without any positive gradient or whose maximum gradient is
    smaller than ``min_gradient`` do not have a valid pycnocline and get ``pycnocline_depth_m = NaN``.

    Args:
        df (pandas.DataFrame): Input DataFrame with density and depth columns.
        density_col (str): Column with potential density anomaly [kg/m3].
        depth_col (str): Column with depth [m].
        profile_col (str or list<str>): Column(s) identifying individual profiles.
            Defaults to ``["ID", "PROFILE_NUMBER"]`` when both exist, else
            ``"PROFILE_NUMBER"``.
        min_gradient (float): Minimum gradient [kg/m3 per metre] to be
            considered a pycnocline. Default 0.0 (no filtering).
    Returns:
        pandas.DataFrame: One row per profile with columns
            [profile_col, ``'pycnocline_depth_m'``, ``'max_gradient'``].
            ``pycnocline_depth_m`` is ``NaN`` when the maximum gradient is
            not positive or below ``min_gradient``.
    """
    profile_col = _resolve_profile_col(df, profile_col)
    depth_col = detect_depth_col(df, depth_col)

    # Iterate over profiles
    rows = []
    for pid, group in df.groupby(profile_col):
        # Average duplicate depths, drop NaNs
        depths, rho = _profile_levels(group, depth_col, density_col)
        if len(depths) < 2:
            logger.debug("pycnocline_depth: profile %s has < 2 valid levels, skipped", pid)
            continue

        # Pycnocline = depth of maximum d(density)/dz (strongest increase with depth)
        grad = np.gradient(rho, depths)
        idx = int(np.argmax(grad))

        significant = grad[idx] > 0 and grad[idx] >= min_gradient
        rows.append({
            **_profile_key_dict(profile_col, pid),
            "pycnocline_depth_m": float(depths[idx]) if significant else np.nan,
            "max_gradient": float(grad[idx]),
        })

    if not rows:
        # Return empty df
        return _empty_profile_frame(profile_col, ["pycnocline_depth_m", "max_gradient"])
    return pd.DataFrame(rows)


def _bracket_gap(z: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Distance between the two measured depths bracketing each target depth."""
    # Get upper bracket/depth level index (max. is the last depth level)
    idx = np.clip(np.searchsorted(z, targets, side="left"), 0, len(z) - 1)

    # Target depths that coincide with a depth level
    exact = np.isclose(z[idx], targets)

    # Avoid idx_hi going negative (if idx is 0)
    idx_hi = np.maximum(idx, 1)

    # Bracket widths/distance between the two depth levels
    gap = z[idx_hi] - z[idx_hi - 1]

    # Return bracket widths (or zero for exact matches)
    return np.where(exact, 0.0, gap)


def interpolate_depth_levels(df: pd.DataFrame,
                             target_depths: ArrayLike,
                             param_col: str = "VAL",
                             depth_col: str = "LEV_M",
                             profile_col: str | list[str] | None = None,
                             max_gap: float | None = None,
                             method: str = "pchip",
                             ) -> pd.DataFrame:
    """Interpolate profiles onto standardised depth levels. Only depth
    levels within the measured range are returned. Profiles with <2 valid
    measurements are skipped.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        target_depths (array-like): Target depth levels [m].
        param_col (str): Column with measured values.
        depth_col (str): Column with depth [m].
        profile_col (str or list<str>): Column(s) identifying individual profiles.
            Defaults to ``["ID", "PROFILE_NUMBER"]`` when both exist, else
            ``"PROFILE_NUMBER"``.
        max_gap (float): Maximum allowed distance [m] between the two
            measured depths bracketing a target depth. Target depths whose
            bracket is wider than this are dropped.
            ``None`` (default) disables the check.
        method (str): Interpolation method: ``"pchip"`` (default) is a
            piecewise cubic Hermite interpolant (smoother
            than linear, no overshoot between measured levels), ``"linear"``
            is piecewise-linear interpolation.
    Returns:
        pandas.DataFrame: DataFrame with one row per profile and target depth.
    Raises:
        ValueError: If ``method`` is not ``'pchip'`` or ``'linear'``.
    """
    if method not in ("pchip", "linear"):
        raise ValueError(f"Unknown method {method!r}; choose 'pchip' or 'linear'")

    profile_col = _resolve_profile_col(df, profile_col)
    param_col = _detect_param_col(df, param_col)
    depth_col = detect_depth_col(df, depth_col)
    target_depths = np.asarray(target_depths, dtype=float)

    # Identify profile keys and metadata
    profile_keys = profile_col if isinstance(profile_col, list) else [profile_col]
    meta_cols = [c for c in ("LATITUDE", "LONGITUDE", "DATEANDTIME", "ID")
                 if c in df.columns and c not in profile_keys]

    # Interpolate each profile
    chunks = []
    for pid, group in df.groupby(profile_col):
        # Average duplicate depths, drop nans (both interpolants need strictly increasing x)
        depths, vals = _profile_levels(group, depth_col, param_col)
        if len(depths) < 2:
            logger.debug("interpolate_depth_levels: profile %s has < 2 valid levels, skipped", pid)
            continue

        # Only work with depths in the given range
        in_range = (target_depths >= depths.min()) & (target_depths <= depths.max())
        if not in_range.any():
            continue

        # Interpolate depths
        depths_sel = target_depths[in_range]
        if method == "pchip":
            from scipy.interpolate import PchipInterpolator
            vals_interp = PchipInterpolator(depths, vals, extrapolate=False)(depths_sel)
        else:
            vals_interp = np.interp(depths_sel, depths, vals)

        # Drop targets whose measurements are too far apart
        if max_gap is not None:
            keep = _bracket_gap(depths, depths_sel) <= max_gap
            depths_sel = depths_sel[keep]
            vals_interp = vals_interp[keep]
            if len(depths_sel) == 0:
                continue

        # Build a DataFrame chunk for this profile
        chunk = pd.DataFrame({depth_col: depths_sel, param_col: vals_interp})
        for c in profile_keys + meta_cols:
            chunk[c] = group[c].iloc[0]
        chunks.append(chunk)

    if not chunks:
        # Return empty df
        return pd.DataFrame(columns=[depth_col, param_col, *profile_keys, *meta_cols])
    return pd.concat(chunks, ignore_index=True)
