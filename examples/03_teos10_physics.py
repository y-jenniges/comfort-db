"""TEOS-10 conversions on COMFORT data.

Demonstrates computation of Absolute Salinity, Conservative Temperature,
density, buoyancy frequency and spiciness via ``comfort.physics``.

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
from comfort.physics import (
    add_teos10_variables,
    compute_buoyancy_frequency,
    compute_density,
    compute_spiciness,
    convert_salinity,
    convert_temperature,
    resolve_pressure,
)

load_dotenv()

LOC_COLS = ["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"]

if __name__ == "__main__":
    # Get DB path from .env
    db_path = os.environ.get("COMFORT_DB_PATH", "")
    if not db_path:
        sys.exit("COMFORT_DB_PATH is not set. Point it to your COMFORT SQLite database.")

    # Load T and S from a small region
    print("Loading TEMPERATURE and SALINITY (North Atlantic, QC_GOOD)...")
    with comfort.connect(db_path) as conn:
        dfs = comfort.load_comfort(
            conn, parameters=["TEMPERATURE", "SALINITY"],
            quality_flags=comfort.QC_GOOD,
            lon_min=-30, lon_max=-20, lat_min=45, lat_max=55,
            output="dict",
        )

    # Average duplicates and merge T + S on (station, profile, depth)
    join_key = ["ID", "PROFILE_NUMBER", "LEV_M"]
    df_t = dfs["TEMPERATURE"].groupby(join_key, as_index=False).agg(
        {"TEMPERATURE": "mean", "LEV_DBAR": "mean", "LATITUDE": "first", "LONGITUDE": "first"})
    df_s = dfs["SALINITY"].groupby(join_key, as_index=False).agg(
        {"SALINITY": "mean"})
    df = df_t.merge(df_s, on=join_key)
    df["LEV_DBAR"] = resolve_pressure(df)  # get or compute available pressure
    n_profiles = df.groupby(["ID", "PROFILE_NUMBER"]).ngroups
    print(f"  Merged T+S: {n_profiles} profiles, {len(df)} rows\n")

    # Individual conversions
    print("=== convert_salinity (SP -> SA) ===")
    df["SA"] = convert_salinity(df["SALINITY"], df["LEV_DBAR"], df["LONGITUDE"], df["LATITUDE"])
    print(df[["LEV_M", "SALINITY", "SA"]].head(5).to_string(index=False), "\n")

    print("=== convert_temperature (in-situ -> CT) ===")
    df["CT"] = convert_temperature(df["TEMPERATURE"], df["SA"], df["LEV_DBAR"], to="CT")
    print(df[["LEV_M", "TEMPERATURE", "CT"]].head(5).to_string(index=False), "\n")

    print("=== convert_temperature (in-situ -> pt0) ===")
    df["pt0"] = convert_temperature(df["TEMPERATURE"], df["SA"], df["LEV_DBAR"], to="pt0")
    print(df[["LEV_M", "TEMPERATURE", "pt0"]].head(5).to_string(index=False), "\n")

    print("=== compute_density (sigma0) ===")
    df["sigma0"] = compute_density(df["SA"], df["CT"])
    print(df[["LEV_M", "SA", "CT", "sigma0"]].head(5).to_string(index=False), "\n")

    # Add SA, CT, sigma0 in one call
    print("=== add_teos10_variables (one call) ===")
    df_in = dfs["TEMPERATURE"].head(20).merge(dfs["SALINITY"].head(20)[join_key + ["SALINITY"]], on=join_key)
    df_in["LEV_DBAR"] = resolve_pressure(df_in)  # get or compute available pressure
    df_auto = add_teos10_variables(df_in, sp_col="SALINITY", t_col="TEMPERATURE")
    print(df_auto[["LEV_M", "TEMPERATURE", "SALINITY", "SA", "CT", "sigma0"]]
          .head(5).to_string(index=False), "\n")

    # Buoyancy frequency (N^2), computed per (station, profile)
    print("=== compute_buoyancy_frequency ===")
    n2 = compute_buoyancy_frequency(df, sa_col="SA", ct_col="CT")
    print(n2.head(8).to_string(index=False), "\n")

    # Spiciness
    print("=== compute_spiciness ===")
    spice = compute_spiciness(df["SA"].head(5), df["CT"].head(5))
    print(f"  Spiciness values: {spice.values.round(4)}")
