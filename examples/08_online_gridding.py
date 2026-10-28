"""Illustrates database-backed gridding at the example of data in the Mediterranean Sea:
The data is filtered and joined in SQL, binned in memory and the gridded tables are stored
in the database, where the wide table is then built by a pure SQL join.
Gridding of data already loaded in memory is shown in example 9.

Requires:
    COMFORT_DB_PATH - path to the COMFORT SQLite database
    BATHYMETRY_PATH - path to the GEBCO bathymetry NetCDF

Install extras:
    pip install "comfort-db[grid]"
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
from dotenv import load_dotenv

from comfort.database.information import get_names_of_all_parameter_tables
from comfort.gridding import (
    GridManager,
    create_wide_table_online,
    get_missing_value_info_per_param,
    load_wide_table,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

# --- Configuration ---
PARAMETERS = ["TEMPERATURE", "SALINITY", "OXYGEN", "NITRATE", "SILICATE", "PHOSPHATE"]

# Mediterranean Sea bounding box
LAT_MIN, LAT_MAX = 30, 46
LON_MIN, LON_MAX = -6, 37

# Grid resolution
DEPTH_INTERVALS = np.array([0, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 4000, 5000])
GRID_DEG = 1
START_YEAR = 1900
END_YEAR = 2020
DTIME = 20


def main():
    db_path = os.environ.get("COMFORT_DB_PATH", "")
    bathy_path = os.environ.get("BATHYMETRY_PATH", "")
    if not db_path:
        sys.exit("COMFORT_DB_PATH is not set.")
    if not bathy_path:
        sys.exit("BATHYMETRY_PATH is not set.")

    # Create GridManager
    with GridManager(db_path) as gm:
        # -------------------------------------------------------------------- #
        # Step 1: Create the grid template                                     #
        # -------------------------------------------------------------------- #
        log.info(
            "Step 1: Creating Mediterranean grid (%d x %d deg, %d depth layers, %d-year steps)...",
            GRID_DEG, GRID_DEG, len(DEPTH_INTERVALS) - 1, DTIME,
        )
        grid = gm.create_grid(
            lon_min=LON_MIN, lon_max=LON_MAX, dlon=GRID_DEG,
            lat_min=LAT_MIN, lat_max=LAT_MAX, dlat=GRID_DEG,
            z_array=DEPTH_INTERVALS,
            time_min=f"{START_YEAR}-01-01", time_max=f"{END_YEAR}-12-31",
            mode="Y", dtime=DTIME,
            bathymetry_grid_path=bathy_path,
        )
        log.info("  Grid ID: %d, %d cells", grid.grid_id, len(grid.grid))

        # -------------------------------------------------------------------- #
        # Step 2: Bin each raw P_* table onto the grid                         #
        # -------------------------------------------------------------------- #
        log.info("Step 2: Mapping parameter tables onto the grid (mean/median/std/count)...")
        mapped = grid.map_tables(gm.connection, parameters=PARAMETERS, replace_existing=True)
        for table in mapped:
            log.info("  %s", table)

        # -------------------------------------------------------------------- #
        # Step 3: Left-join grid_<grid_id> with every mapped P_*_<grid_id>     #
        # to produce one wide table                                            #
        # -------------------------------------------------------------------- #
        log.info("Step 3: Joining mapped tables into a wide table...")
        existing_wide = get_names_of_all_parameter_tables(
            gm.connection, like_pattern=f"wide|_{grid.grid_id}|_%",
            escape_char="|", include_digits=True, table_type="table",
        )
        for old_name in existing_wide:
            gm.connection.execute(f"DROP TABLE {old_name}")
            log.info("  Dropped stale wide table: %s", old_name)

        wide_name = create_wide_table_online(gm.connection, grid.grid_id, parameters=PARAMETERS)
        gm.connection.commit()
        log.info("  Created wide table: %s", wide_name)

        # -------------------------------------------------------------------- #
        # Step 4: Read the wide table back and summarise per-parameter         #
        # coverage                                                             #
        # -------------------------------------------------------------------- #
        log.info("Step 4: Loading the wide table back and summarising coverage...")
        df_wide = load_wide_table(gm.connection, wide_name)
        log.info("  %d cells (land-only cells with no data dropped)", len(df_wide))

        coverage = get_missing_value_info_per_param(gm.connection, wide_name, PARAMETERS)
        print("\n--- Per-parameter missing values (%) ---")
        print(coverage.to_string(index=False))

        log.info("Done. Wide table: %s", wide_name)
        log.info("Select it in the Dash app's Grid tab (python -m comfort.app.explorer).")


if __name__ == "__main__":
    main()
