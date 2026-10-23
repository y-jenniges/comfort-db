"""Reproduces the data preparation of:
Jenniges et al. (2027, in prep.)

Builds a 1° x 1° x 12 depth layers x 13 20-year time steps gridded
product from the COMFORT database with quality-controlled temperature,
salinity, oxygen, nitrate, silicate and phosphate in unified
units (°C / PSU / umol/kg).

Steps:
  1. Load global parameters with QC_GOOD, O2/NO3/SiO4/PO4 are converted to
     umol/kg on load.
  2. Plot annual coverage per parameter (to decide on START_YEAR)
  3. Filter to START_YEAR onwards
  4. Average params over identical (lat, lon, depth, time)
  5. Convert in-situ T to potential temperature
  6. Grid: 1° x 1° x 12 depth layers (0-5000 m) x 13 20-year time means
  7. Remove empty land cells (using GEBCO)
  8. Diagnostic plots

Requires:
    COMFORT_DB_PATH - path to COMFORT SQLite database
    BATHYMETRY_PATH - path to GEBCO bathymetry NetCDF

Install extras:
    pip install "comfort-db[all]"
"""
from __future__ import annotations

import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from comfort import connect
from comfort.gridding import Grid, average_duplicate_locations
from comfort.io import load_comfort
from comfort.physics import convert_to_potential_temperature
from comfort.plot import (
    plot_annual_coverage,
    plot_correlation,
    plot_depth_coverage,
    plot_missing_value_info,
    plot_missing_value_info_map_joint,
    plot_ts_diagram,
)
from comfort.qc import QC_GOOD

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

# --- Configuration ---
PARAMETERS = ["TEMPERATURE", "SALINITY", "OXYGEN", "NITRATE", "SILICATE", "PHOSPHATE"]
DEPTH_MAX = 5000
START_YEAR = 1772
END_YEAR = 2020
DTIME = 20

LAT_MIN, LAT_MAX = 0, 70
LON_MIN, LON_MAX = -77, 30

# Depth levels
DEPTH_INTERVALS = np.array([0, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 4000, 5000])
GRID_DEG = 1


def _save_path(output_dir: str, name: str) -> str | None:
    """Build a save path, or return None if no output directory is set."""
    if not output_dir:
        return None
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, name)


def main():
    db_path = os.environ.get("COMFORT_DB_PATH", "")
    bathy_path = os.environ.get("BATHYMETRY_PATH", "")
    output_dir = "output/reproduced_jenniges2027/"

    if not db_path:
        sys.exit("COMFORT_DB_PATH is not set.")
    if not bathy_path:
        sys.exit("BATHYMETRY_PATH is not set.")

    with connect(db_path) as conn:
        # ------------------------------------------------------------------ #
        # Step 1: Load parameters with QC; convert params are converted to   #
        # umol/kg on load (SQL-joined co-located T/S, App. C densities)      #
        # ------------------------------------------------------------------ #
        log.info("Step 1: Loading parameters (QC_GOOD, depth <= %d m)...", DEPTH_MAX)
        raw = load_comfort(
            conn, parameters=PARAMETERS, quality_flags=QC_GOOD,
            lon_min=LON_MIN, lon_max=LON_MAX,
            lat_min=LAT_MIN, lat_max=LAT_MAX,
            depth_max=DEPTH_MAX,
            convert_units=True, use_density=True, as_xarray=False,
        )
        for param in PARAMETERS:
            log.info("  %-15s %10d rows", param, len(raw[param]))
        total_qc = sum(len(df) for df in raw.values())
        log.info("  Total after QC + region: %d", total_qc)

        # ------------------------------------------------------------------ #
        # Step 2: Annual coverage plot                                       #
        # ------------------------------------------------------------------ #
        log.info("Step 2: Plotting annual coverage...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        plot_annual_coverage(raw, ax=ax1)
        plot_annual_coverage(raw, normalise=True, ax=ax2)
        ax2.axvline(
            START_YEAR, color="red", linestyle="--",
            label=f"START_YEAR = {START_YEAR}",
        )
        ax2.legend()
        plt.tight_layout()
        sp = _save_path(output_dir, "annual_coverage.png")
        if sp:
            plt.savefig(sp, dpi=150, bbox_inches="tight")
        plt.close()

        # ------------------------------------------------------------------ #
        # Step 3: Filter to START_YEAR onwards                               #
        # ------------------------------------------------------------------ #
        log.info("Step 3: Filtering to >= %d...", START_YEAR)
        for param in PARAMETERS:
            dates = pd.to_datetime(raw[param]["DATEANDTIME"])
            n_before = len(raw[param])
            raw[param] = raw[param][dates.dt.year >= START_YEAR].reset_index(drop=True)
            log.info("  %-15s %10d -> %10d", param, n_before, len(raw[param]))

        # ------------------------------------------------------------------ #
        # Step 4: Average duplicate (lat, lon, depth, time) records          #
        # ------------------------------------------------------------------ #
        log.info("Step 4: Averaging duplicate locations...")
        averaged = average_duplicate_locations(raw)
        for param in PARAMETERS:
            log.info("  %-15s %10d -> %10d", param, len(raw[param]), len(averaged[param]))
        total = sum(len(df) for df in averaged.values())
        log.info("  Total after processing: %d values", total)

        # ------------------------------------------------------------------ #
        # Step 5: Convert averaged in-situ temperature to potential          #
        # temperature (co-located salinity + pressure, gsw)                  #
        # ------------------------------------------------------------------ #
        log.info("Step 5: Converting T to potential temperature...")
        averaged = convert_to_potential_temperature(averaged)

        # ------------------------------------------------------------------ #
        # Step 6: Spatio-temporal grid                                       #
        # ------------------------------------------------------------------ #
        log.info(
            "Step 6: Gridding (1 deg x %d depth layers x annual)...",
            len(DEPTH_INTERVALS) - 1,
        )

        # Define grid
        grid = Grid(
            lon_min=LON_MIN, lon_max=LON_MAX, dlon=GRID_DEG,
            lat_min=LAT_MIN, lat_max=LAT_MAX, dlat=GRID_DEG,
            z_array=DEPTH_INTERVALS,
            bathymetry_grid_path=bathy_path,
            time_min=f"{START_YEAR}-01-01",
            time_max=f"{END_YEAR}-12-31",
            mode="Y", dtime=DTIME,
        )
        log.info("  Grid template: %d cells", len(grid.grid))

        # Mapping parameter tables to grid
        param_cols = [f"P_{p}" for p in PARAMETERS]
        df_wide = grid.map_dataframes(averaged, agg="mean", dropping_land_cells=True)
        for param, col in zip(PARAMETERS, param_cols):
            log.info("  %-15s %6d cell-years with data", param, df_wide[col].notna().sum())

        n_years = df_wide["DATEANDTIME"].nunique()
        log.info(
            "Done. Final grid: %d rows, %d years, %d parameters",
            len(df_wide), n_years, len(param_cols),
        )

        # ------------------------------------------------------------------ #
        # Step 8: Diagnostic plots                                           #
        # ------------------------------------------------------------------ #
        log.info("Step 8: Generating diagnostic plots...")

        # Histograms
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, param in zip(axes.flat, PARAMETERS):
            col = f"P_{param}"
            vals = df_wide[col].dropna()
            ax.hist(vals, bins=50, edgecolor="none", alpha=0.8)
            ax.set_title(param)
            ax.set_ylabel("Count")
        plt.suptitle("Parameter distributions (gridded)")
        plt.tight_layout()
        sp = _save_path(output_dir, "histograms.png")
        if sp:
            plt.savefig(sp, dpi=150, bbox_inches="tight")
        plt.close()

        # T-S diagram
        ts_data = df_wide[["P_TEMPERATURE", "P_SALINITY", "LEV_M"]].dropna()
        if not ts_data.empty:
            plot_ts_diagram(
                ts_data,
                temperature_col="P_TEMPERATURE",
                salinity_col="P_SALINITY",
                depth_col="LEV_M",
                save_as=_save_path(output_dir, "ts_diagram.png"),
            )
            plt.close()

        # Correlation matrix heatmap
        corr = df_wide[param_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = [c.replace("P_", "") for c in param_cols]
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            xticklabels=labels, yticklabels=labels, ax=ax,
            vmin=-1, vmax=1,
        )
        ax.set_title("Parameter correlation matrix")
        plt.tight_layout()
        sp = _save_path(output_dir, "correlation_matrix.png")
        if sp:
            plt.savefig(sp, dpi=150, bbox_inches="tight")
        plt.close()

        # Pairwise correlations (T-O2, NO3-PO4)
        for x_col, y_col in [
            ("P_TEMPERATURE", "P_OXYGEN"),
            ("P_NITRATE", "P_PHOSPHATE"),
        ]:
            name = f"corr_{x_col.replace('P_', '')}_{y_col.replace('P_', '')}"
            plot_correlation(
                df_wide, x_col=x_col, y_col=y_col, colour_col="LEV_M",
                save_as=_save_path(output_dir, f"{name}.png"),
            )
            plt.close()

        # Missingness bar chart
        n_total = len(df_wide)
        miss = pd.DataFrame({
            "parameter": [c.replace("P_", "") for c in param_cols],
            "relative": [df_wide[c].isna().sum() / n_total * 100 for c in param_cols],
        })
        plot_missing_value_info(miss, save_as=_save_path(output_dir, "missingness_bar.png"))
        plt.close()

        # Missingness map (% of rows where every parameter is missing at
        # once, one overview map across all lat/lon cells)
        plot_missing_value_info_map_joint(
            df_wide, param_cols, save_as=_save_path(output_dir, "missingness_map.png"),
        )
        plt.close()

        # Depth coverage profile
        plot_depth_coverage(df_wide, param_cols, save_as=_save_path(output_dir, "depth_coverage.png"))
        plt.close()

        # ------------------------------------------------------------------ #
        # Output                                                             #
        # ------------------------------------------------------------------ #
        years = pd.to_datetime(df_wide["DATEANDTIME"]).dt.year
        print("\n--- Gridded dataset summary ---")
        print(f"Rows: {len(df_wide)}, Years: {n_years}, Parameters: {len(param_cols)}")
        print(f"Year range: {years.min()} - {years.max()}")
        print(
            df_wide[["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"] + param_cols]
            .describe()
            .to_string()
        )

        # Store
        out_path = _save_path(output_dir, "na_20y_dataset.csv")
        if out_path:
            df_wide.to_csv(out_path, index=False)
            log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
