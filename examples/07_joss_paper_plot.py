"""Generates the global T-S/oxygen figure used in the JOSS paper.

Demonstrates the comfort-db workflow in a single plot: QC filtering,
TEOS-10 computations (SA, CT, isopycnals), oxygen unit harmonisation
and hexbin aggregation to keep a global ~40 million rows dataset readable.

Requires:
    COMFORT_DB_PATH - path to the COMFORT SQLite database

Install extras:
    pip install "comfort-db[plot]"
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import comfort
from comfort.database import get_num_samples
from comfort.gridding import average_duplicate_locations
from comfort.physics import add_teos10_variables
from comfort.plot import plot_ts_diagram
from comfort.qc import flag_salinity_like_oxygen

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def main():
    # Get DB path from .env
    db_path = os.environ.get("COMFORT_DB_PATH", "")
    if not db_path:
        sys.exit("COMFORT_DB_PATH is not set. Point it to your COMFORT SQLite database.")

    with comfort.connect(db_path) as conn:
        # Get total oxygen count
        num_observations = get_num_samples(conn=conn, table_name="P_OXYGEN")
        log.info(f"Total number of oxygen observations: {num_observations}")

        # Load global data with SQL-level QC filtering, oxygen is converted
        # to umol/kg on load (SQL-joined co-located T/S, App. C densities)
        log.info("Loading TEMPERATURE, SALINITY, OXYGEN (global, QC_GOOD)...")
        raw = comfort.load_comfort(
            conn, parameters=["TEMPERATURE", "SALINITY", "OXYGEN"],
            quality_flags=comfort.QC_GOOD,
            convert_units=True, use_density=True, as_xarray=False,
        )
        for param, df_param in raw.items():
            log.info("  %-12s %10d rows", param, len(df_param))

        # Define plotting label (oxygen is converted to umol/kg on load; unit parts separated by a space like the SA label)
        oxygen_label = r"Oxygen [µmol kg$^{-1}$]"

    # Drop oxygen rows that are suspected mislabelled salinity values
    suspect = flag_salinity_like_oxygen(raw["OXYGEN"])
    log.info("  Dropping %d/%d OXYGEN rows with salinity-like values",
             suspect.sum(), len(raw["OXYGEN"]))
    raw["OXYGEN"] = raw["OXYGEN"][~suspect].copy()

    # Average duplicate (lat, lon, depth, time) records
    log.info("Averaging duplicate locations...")
    averaged = average_duplicate_locations(raw)
    for param, avg in averaged.items():
        log.info("  %-12s %10d -> %10d unique locations", param, len(raw[param]), len(avg))

    # Merge averaged T, S, O2 into a single wide DataFrame
    loc_cols = ["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"]
    df = (
        averaged["TEMPERATURE"]
        .merge(averaged["SALINITY"], on=loc_cols)
        .merge(averaged["OXYGEN"], on=loc_cols)
    )
    log.info("Merged T+S+O2: %d rows", len(df))

    # Add absolute salinity and conservative temperature (needed for isopycnals)
    df = add_teos10_variables(df, sp_col="SALINITY", t_col="TEMPERATURE", pressure_col="LEV_DBAR")

    # Get count of remaining oxygen observations
    num_remaining_observations =len(df.dropna())
    log.info(f"Remaining number of oxygen observations: {num_remaining_observations}")

    # Global T-S diagram with isopycnal contours, coloured by oxygen concentration
    ax = plot_ts_diagram(
        df,
        temp_col="CT",
        sal_col="SA",
        temp_type="CT",
        sal_type="SA",
        color_col="OXYGEN",
        colorbar_label=oxygen_label,
        kind="hexbin",
        gridsize=60,
    )

    # Annotate low-oxygen blobs
    ax.annotate(
        "Baltic Sea",
        xy=(10.5, 5.5), xytext=(8, 10),
        arrowprops=dict(arrowstyle="->", color="white"),
        fontsize=9, ha="center", color="white",
    )
    ax.annotate(
        "Black Sea",
        xy=(22, 9), xytext=(22, 15),
        arrowprops=dict(arrowstyle="->", color="white"),
        fontsize=9, ha="center", color="white",
    )
    ax.annotate(
        "Arabian Sea",
        xy=(36, 3.5), xytext=(31, 8),
        arrowprops=dict(arrowstyle="->", color="white"),
        fontsize=9, ha="center", color="white",
    )

    # Save
    save_as = Path("output/ts_oxygen_global.png")
    save_as.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(save_as, dpi=1000, bbox_inches="tight")
    log.info("Saved figure to %s", save_as)


if __name__ == "__main__":
    main()
