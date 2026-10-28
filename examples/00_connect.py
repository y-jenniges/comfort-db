"""Connecting to the COMFORT database.

This example shows how to point the library at your local copy of the
COMFORT SQLite file and (optionally) a GEBCO bathymetry NetCDF for gridding.

Setup (once):
    1. Copy ``.env.example`` to ``.env`` in the project root.
    2. Fill in the paths for your machine.
    3. Run this script::
        python examples/00_connect.py
"""
from __future__ import annotations
import os
import sys
from dotenv import load_dotenv

import comfort
from comfort.qc import QC_GOOD

load_dotenv()

DB_PATH = os.environ.get("COMFORT_DB_PATH", "")
BATHYMETRY_PATH = os.environ.get("BATHYMETRY_PATH", "")

# Check if path to COMFORT DB is set
if not DB_PATH:
    sys.exit(
        "COMFORT_DB_PATH is not set.\n"
        "Copy .env.example to .env and fill in the path to your COMFORT SQLite file."
    )

# --- 1. Context-managed connection ---------------------------------------------- #
with comfort.connect(DB_PATH) as conn:
    # List parameters in the database
    params = comfort.list_parameters(conn)
    print(f"Available parameters: {params}\n")

    # Get basic info about given parameters
    summary = comfort.describe_variables(
        conn, parameters=["NITRATE", "DIC", "C13"],
        quality_flags=QC_GOOD,
    )
    print("--- Variable summary (QC_GOOD) ---")
    print(summary.to_string(index=False), "\n")

# --- 2. load_comfort with a file path (manages connection automatically)  --------- #
# Load parameters of interest (filtered for quality, spatial and temporal extent)
dfs = comfort.load_comfort(
    DB_PATH,
    parameters=["NITRATE", "DIC"],
    quality_flags=QC_GOOD,
    lat_min=30, lat_max=70,
    date_min="2000-01-01",
    output="dict",
)
for parameter, df in dfs.items():
    print(
        f"{parameter}: {len(df)} rows, profiles {df['PROFILE_NUMBER'].nunique()}, "
        f"n instruments {df['INSTRUMENT_ID'].nunique()}")
print()

# --- 3. xarray output with depth interpolation ------------------------------------ #
# Load nitrate data as xarray, mapped given depth levels
ds = comfort.load_comfort(
    DB_PATH,
    parameters=["NITRATE"],
    quality_flags=QC_GOOD,
    target_depths=[0, 50, 100, 200, 500, 1000],
    output="xarray",
)
print("--- xarray Dataset ---")
print(ds, "\n")

# --- 4. Gridding with bathymetry (optional) ---------------------------------------- #
if BATHYMETRY_PATH:
    from comfort import Grid

    # Create grid with 5-year time steps
    grid = Grid(
        lon_min=-80, lon_max=0, dlon=5,
        lat_min=30, lat_max=70, dlat=5,
        bathymetry_grid_path=BATHYMETRY_PATH,
        z_array=[0, 100, 500, 1000],
        time_min="2000-01-01", time_max="2020-12-31",
        mode="Y", dtime=5,
    )
    print("--- Grid ---")
    print(f"Grid ID:    {grid.grid_id}")
    print(f"Cells:      {grid.grid.shape}")
    print(f"Time steps: {len(grid.time_array)}")
else:
    print(
        "Skipping gridding example - set COMFORT_BATHYMETRY_PATH in .env to enable."
    )
