"""Reproduces the data preparation described in the supplement of:
Jenniges et al. (2025), doi:10.5281/zenodo.15827777

Steps:
  1. Select North Atlantic (0-70 °N, 77 °W - 30 °E, 0-5000 m) x 6 parameters
  2. QC filter: PQF1 > 0, PQF2 > 2
  3. Average T/S over identical (lat, lon, depth, time)
  4.-5. Convert O2/NO3/SiO4/PO4 to µmol/kg (Korablev et al. 2021; Benson &
        Krause 1984) - handled by load_comfort(convert_units=True), which
        joins co-located salinity/temperature in SQL (averaged per station
        and depth level)
  6. Average O2/NO3/SiO4/PO4 over identical (lat, lon, depth, time)
        - steps 3, 6 are handled by
          comfort.gridding.average_duplicate_locations
  7. Convert in-situ temperature to potential temperature (gsw) via
     comfort.physics.convert_to_potential_temperature
  8. Grid onto 1 deg x 12 depth intervals, averaging over all available times
  9. Remove empty land cells using GEBCO bathymetry
 10. KNN imputation (k=5, distance-weighted, 9 features)

Note: this reproduces the paper's *methodology*, not its exact published
numbers. Unit conversion follows the COMFORT report Appendix C density
conventions (lab density at 22 °C for nutrients, in-situ-temperature
density for oxygen mL/L, both at atmospheric pressure), while the
published dataset was built with in-situ density including pressure.
Validated against the published wide_table_knn.csv (2026-09-04, 49030
cells): Directly-observed cells match closely (salinity/temperature to
float precision; nitrate/oxygen/phosphate/silicate to a mean deviation
of <0.1% of range, from the density difference alone). In the ~40% of
cells with no direct observation, KNN imputation (step 10) amplifies
that same small input difference into larger per-cell deviations (up to
~90 umol/kg for oxygen). The imputation mask itself matches the
published `imputed` flag exactly, only the imputed values drift.

Requires:
    COMFORT_DB_PATH - path to the COMFORT SQLite database
    BATHYMETRY_PATH - path to the GEBCO bathymetry NetCDF

Install extras:
    pip install "comfort-db[all]"
"""
from __future__ import annotations

import logging
import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

from comfort import connect
from comfort.gridding import Grid, average_duplicate_locations
from comfort.io import load_comfort
from comfort.physics import convert_to_potential_temperature
from comfort.qc import QC_GOOD
from comfort.scaling import ParamScaler

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

# --- Constants from the paper ---
PARAMETERS = ["TEMPERATURE", "SALINITY", "OXYGEN", "NITRATE", "SILICATE", "PHOSPHATE"]

LAT_MIN, LAT_MAX = 0, 70
LON_MIN, LON_MAX = -77, 30
DEPTH_MIN, DEPTH_MAX = 0, 5000

DEPTH_INTERVALS = np.array([
    0, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 4000, 5000,
])
GRID_DEG = 1
KNN_K = 5


def main():
    # Get DB and bathymetry pathes
    db_path = os.environ.get("COMFORT_DB_PATH", "")
    bathy_path = os.environ.get("BATHYMETRY_PATH", "")
    if not db_path:
        sys.exit(
            "COMFORT_DB_PATH is not set.\n"
            "Point it to your COMFORT SQLite database."
        )
    if not bathy_path:
        sys.exit(
            "BATHYMETRY_PATH is not set.\n"
            "Point it to the GEBCO bathymetry NetCDF file."
        )

    with connect(db_path) as conn:
        # ------------------------------------------------------------------ #
        # Step 1-2 (+4-5): Load all parameters with QC and spatial filter.   #
        # convert_units=True converts O2/NO3/SiO4/PO4 to umol/kg on load:    #
        # Co-located salinity/temperature are joined in SQL (averaged per    #
        # station and depth level)                                           #
        # ------------------------------------------------------------------ #
        log.info("Step 1-2 (+4-5): Loading parameters, converting units on load...")
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
        # Steps 3, 6: Average duplicate (lat, lon, depth, time) records      #
        # ------------------------------------------------------------------ #
        log.info("Steps 3, 6: Averaging duplicate locations...")
        averaged = average_duplicate_locations(raw)
        for param in PARAMETERS:
            log.info("  %-15s %10d -> %10d", param, len(raw[param]), len(averaged[param]))
        total_final = sum(len(df) for df in averaged.values())
        log.info("  Total after processing: %d values", total_final)

        # ------------------------------------------------------------------ #
        # Step 7: Convert averaged in-situ temperature to potential          #
        # temperature                                                        #
        # ------------------------------------------------------------------ #
        log.info("Step 7: Converting T to potential temperature...")
        averaged = convert_to_potential_temperature(averaged)

        # ------------------------------------------------------------------ #
        # Step 8: Gridding (1 deg x 12 depth intervals, time-averaged)       #
        # ------------------------------------------------------------------ #
        log.info("Step 8: Creating spatial grid and mapping parameters...")
        grid = Grid(
            lon_min=LON_MIN, lon_max=LON_MAX, dlon=GRID_DEG,
            lat_min=LAT_MIN, lat_max=LAT_MAX, dlat=GRID_DEG,
            z_array=DEPTH_INTERVALS,
            bathymetry_grid_path=bathy_path,
        )
        log.info("  Grid template: %d cells", len(grid.grid))

        # Map parameters to grid and drop land cells with not measurements
        param_cols = [f"P_{p}" for p in PARAMETERS]
        df_wide = grid.map_dataframes(averaged, dropping_land_cells=True)
        for param, col in zip(PARAMETERS, param_cols):
            log.info("  %-15s %6d cells with data", param, df_wide[col].notna().sum())

        # ------------------------------------------------------------------ #
        # Step 10: KNN imputation (k=5, weights=distance)                   #
        # ------------------------------------------------------------------ #
        log.info("Step 10: KNN imputation (k=%d, weights=distance)...", KNN_K)
        from sklearn.impute import KNNImputer

        # Print some stats
        n_total = len(df_wide)
        n_empty = df_wide[param_cols].isna().all(axis=1).sum()
        n_partial = (
                df_wide[param_cols].isna().any(axis=1)
                & ~df_wide[param_cols].isna().all(axis=1)
        ).sum()
        log.info("  Completely empty: %d (%.1f%%)", n_empty, 100 * n_empty / n_total)
        log.info("  Partially empty:  %d (%.1f%%)", n_partial, 100 * n_partial / n_total)
        for col in param_cols:
            n_miss = df_wide[col].isna().sum()
            log.info("    %-20s %6d missing (%.1f%%)", col, n_miss, 100 * n_miss / n_total)
        n_all_miss = df_wide[param_cols].isna().sum().sum()
        log.info(
            "  Total missing: %d / %d (%.1f%%)",
            n_all_miss, n_total * len(param_cols), 100 * n_all_miss / (n_total * len(param_cols)),
        )

        # MinMax scaling of parameters
        scaler = ParamScaler()
        scaled = scaler.scale_columns(param_cols, df_wide[param_cols])
        features = scaled[[f"{col}_SCALED" for col in param_cols]].rename(
            columns=lambda c: c.removesuffix("_SCALED")
        )

        # MinMax scaling of lat/lon/depth (scaled to fixed domains)
        features["LATITUDE"] = (df_wide["LATITUDE"] - LAT_MIN) / (LAT_MAX - LAT_MIN)
        features["LONGITUDE"] = (df_wide["LONGITUDE"] - LON_MIN) / (LON_MAX - LON_MIN)
        features["LEV_M"] = (df_wide["LEV_M"] - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)

        # Imputation
        imputer = KNNImputer(n_neighbors=KNN_K, weights="distance")
        imputed = pd.DataFrame(
            imputer.fit_transform(features),
            columns=features.columns,
            index=features.index,
        )

        # Track which cells were imputed (% of params missing before KNN)
        pct_imputed = df_wide[param_cols].isna().mean(axis=1) * 100

        # Inverse-transform parameters back to original units
        df_result = df_wide[["LATITUDE", "LONGITUDE", "LEV_M"]].copy()
        for col in param_cols:
            df_result[col] = scaler.scalers[col].inverse_transform(imputed[[col]])[:, 0]
        df_result["pct_imputed"] = pct_imputed.values

        remaining_nan = df_result[param_cols].isna().sum().sum()
        log.info("  Missing after imputation: %d", remaining_nan)
        log.info(
            "Done. Final grid: %d cells x %d parameters",
            len(df_result), len(param_cols),
        )

        # ------------------------------------------------------------------ #
        # Output                                                             #
        # ------------------------------------------------------------------ #
        print("\n--- Preprocessed grid (first 10 rows) ---")
        display_cols = ["LATITUDE", "LONGITUDE", "LEV_M"] + param_cols + ["pct_imputed"]
        print(df_result[display_cols].head(10).to_string(index=False))

        out_path = Path("output/reproduced_jenniges2025/reproduced_jenniges2025.csv")
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)  # Create dir if necessary
            df_result.to_csv(out_path, index=False)
            log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
