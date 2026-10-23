"""Profile analysis: Gradients, completeness, MLD, pycnocline, interpolation.

Loads and analyse real profiles from the COMFORT database.
(Duplicates in (station, profile, depth) are averaged initially.)

Requires:
    COMFORT_DB_PATH - path to the COMFORT SQLite database

Install extras:
    pip install "comfort-db[all]"
"""
from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

import comfort
from comfort.profile_analysis import (
    average_duplicate_records_per_profile,
    interpolate_depth_levels,
    mixed_layer_depth,
    profile_completeness,
    pycnocline_depth,
    vertical_gradient,
)
from comfort.physics import add_teos10_variables, resolve_pressure

load_dotenv()

if __name__ == "__main__":
    # Get DB path from .env file
    db_path = os.environ.get("COMFORT_DB_PATH", "")
    if not db_path:
        sys.exit("COMFORT_DB_PATH is not set. Point it to your COMFORT SQLite database.")

    # Load temperature profiles from the North Atlantic
    print("Loading TEMPERATURE and SALINITY (North Atlantic, QC_GOOD)...")
    with comfort.connect(db_path) as conn:
        dfs = comfort.load_comfort(
            conn, parameters=["TEMPERATURE", "SALINITY"],
            quality_flags=comfort.QC_GOOD,
            lat_min=45, lat_max=65, lon_min=-30, lon_max=0,
            as_xarray=False,
        )

    # Average duplicates
    averaged = average_duplicate_records_per_profile(dfs)
    df_t, df_s = averaged["TEMPERATURE"], averaged["SALINITY"]

    # Count profiles (a profile is identified by (ID, PROFILE_NUMBER))
    profile_key = ["ID", "PROFILE_NUMBER"]
    print(f"  Temperature: {df_t[profile_key].drop_duplicates().shape[0]} profiles, {len(df_t)} rows")
    print(f"  Salinity:    {df_s[profile_key].drop_duplicates().shape[0]} profiles, {len(df_s)} rows\n")

    # Pick a few profiles for demonstration
    sample_ids = sorted(df_t["ID"].unique())[:5]
    df_t_sample = df_t[df_t["ID"].isin(sample_ids)]

    # Vertical gradient (auto-detects 'TEMPERATURE' as value column)
    print("=== vertical_gradient ===")
    grad = vertical_gradient(df_t_sample)
    print(grad[["ID", "PROFILE_NUMBER", "LEV_M", "TEMPERATURE", "TEMPERATURE_gradient"]]
          .head(5).to_string(index=False), "\n")

    # Profile completeness against WOD standard depths
    print("=== profile_completeness ===")
    comp = profile_completeness(
        df_t_sample, reference_depths=[0, 50, 100, 200, 500, 1000, 2000],
    )
    print(comp.to_string(index=False), "\n")

    # Interpolate onto target depth levels
    print("=== interpolate_depth_levels ===")
    interp = interpolate_depth_levels(
        df_t_sample, target_depths=[0, 50, 100, 200, 500, 1000],
    )
    print(interp[["ID", "PROFILE_NUMBER", "LEV_M", "TEMPERATURE"]]
          .head(5).to_string(index=False), "\n")

    # MLD and pycnocline require density - compute sigma0 from T + S
    # (sigma0 = potential density anomaly)
    join_key = ["ID", "PROFILE_NUMBER", "LEV_M"]
    df_ts = df_t.merge(df_s[join_key + ["SALINITY"]], on=join_key)
    df_ts["LEV_DBAR"] = resolve_pressure(df_ts)
    df_ts = add_teos10_variables(df_ts, sp_col="SALINITY", t_col="TEMPERATURE")

    # Pick profiles with enough depth levels for MLD detection
    deep_profiles = (df_ts.groupby(profile_key)
                     .filter(lambda g: g["LEV_M"].max() >= 200))
    sample_deep = sorted(deep_profiles["ID"].unique())[:5]
    df_mld = deep_profiles[deep_profiles["ID"].isin(sample_deep)]

    print("=== mixed_layer_depth ===")
    mld = mixed_layer_depth(df_mld, density_col="sigma0")
    print(mld.head(5).to_string(index=False), "\n")

    print("=== pycnocline_depth ===")
    pyc = pycnocline_depth(df_mld, density_col="sigma0", min_gradient=0.001)
    print(pyc.head(5).to_string(index=False))
