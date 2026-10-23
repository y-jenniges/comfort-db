"""Quick-start: load, explore and filter COMFORT data.

Requires:
    COMFORT_DB_PATH - path to the COMFORT SQLite database
    (set this in .env file)
"""
from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

import comfort

load_dotenv()

if __name__ == "__main__":
    # Get db path from .env file
    db_path = os.environ.get("COMFORT_DB_PATH", "")
    if not db_path:
        sys.exit("COMFORT_DB_PATH is not set. Point it to your COMFORT SQLite database.")

    with comfort.connect(db_path) as conn:
        # Database overview (taking a few minutes)
        print("=== info ===")
        comfort.info(conn)
        print()

        # Detailed variable summary with QC filtering
        print("=== describe_variables ===")
        summary = comfort.describe_variables(conn,
                                             parameters=["NITRATE", "DIC"],
                                             quality_flags=comfort.QC_GOOD
                                             )
        print(summary.to_string(index=False), "\n")

        # Load as dict of DataFrames
        print("=== load_comfort (dict output) ===")
        dfs = comfort.load_comfort(
            conn,
            parameters=["NITRATE", "DIC"],
            quality_flags=comfort.QC_GOOD,
            as_xarray=False,
        )
        for name, df in dfs.items():
            print(f"  {name}: {len(df)} rows, columns {list(df.columns)}")
        print()

        # Load as xarray Dataset
        print("=== load_comfort (xarray output) ===")
        ds = comfort.load_comfort(
            conn,
            parameters=["NITRATE", "DIC"],
            quality_flags=comfort.QC_GOOD,
            as_xarray=True,
        )
        print(ds, "\n")

        # Spatial and temporal filtering in SQL
        print("=== load_comfort with spatial filter ===")
        dfs_filtered = comfort.load_comfort(
            conn,
            parameters=["NITRATE"],
            quality_flags=comfort.QC_GOOD,
            lat_min=40, lat_max=65,
            lon_min=-30, lon_max=0,
            as_xarray=False,
        )
        df_nit = dfs_filtered.get("NITRATE")
        if df_nit is not None:
            print(f"  Profiles: {df_nit['PROFILE_NUMBER'].nunique()}, rows: {len(df_nit)}")

        # Post-load subsetting on an already-loaded DataFrame
        print("\n=== subset_region ===")
        if df_nit is not None and not df_nit.empty:
            shallow = comfort.subset_region(df_nit, depth_max=200)
            print(f"  Before: {len(df_nit)} rows, after depth_max=200: {len(shallow)} rows")

        # Lower-level read: single parameter table
        print("\n=== read_parameter ===")
        df_oxy = comfort.read_parameter(conn,
                                        parameter="OXYGEN",
                                        quality_flags=comfort.QC_GOOD,
                                        limit=1000
                                        )
        print(f"  OXYGEN (first 1000 QC-good rows): {len(df_oxy)} rows")
