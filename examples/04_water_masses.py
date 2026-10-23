"""Water-mass classification and characterisation, distance to coast and province assignment from ``comfort.geo``.

Requires:
    COMFORT_DB_PATH - path to the COMFORT SQLite database

Optional:
    LONGHURST_SHP_PATH - path to the Longhurst provinces shapefile (marineregions.org)
    JENNIGES_CSV_PATH  - path to cluster_set.csv (Zenodo record 15201767)
    BASINS_SHP_PATH    - path to the Global Oceans and Seas shapefile (marineregions.org)
    FAY_NC_PATH        - path to the Fay & McKinley (2014) biomes NetCDF file (https://essd.copernicus.org/articles/6/273/2014/)

Install extras:
    pip install "comfort-db[geo,grid]" (add [plot] too for the basin-filter map check)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

import comfort
from comfort.geo import (
    classify_from_grid,
    distance_to_coast,
    load_biomes,
    load_jenniges_provinces,
    load_longhurst,
    load_basins,
    select_regions,
    water_mass_masks,
    water_mass_statistics,
)
from comfort.gridding import average_duplicate_locations
from comfort.physics import add_teos10_variables, compute_spiciness

load_dotenv()

if __name__ == "__main__":
    # Get DB path from .env
    db_path = os.environ.get("COMFORT_DB_PATH", "")
    if not db_path:
        sys.exit("COMFORT_DB_PATH is not set. Point it to your COMFORT SQLite database.")

    # Load nitrate, temperature and salinity with spatial metadata
    print("Loading NITRATE, TEMPERATURE, SALINITY (QC_GOOD, North Atlantic)...")
    with comfort.connect(db_path) as conn:
        dfs = comfort.load_comfort(
            conn, parameters=["NITRATE", "TEMPERATURE", "SALINITY"],
            quality_flags=comfort.QC_GOOD,
            lon_min=-80, lon_max=0,
            lat_min=20, lat_max=65,
            depth_max=100,
            as_xarray=False,
            convert_units=True,
        )
    df = dfs["NITRATE"]
    print(f"  {len(df)} rows, {df['PROFILE_NUMBER'].nunique()} profiles\n")

    # --- Adding density (sigma0) and spiciness info ---
    print("=== add_teos10_variables + compute_spiciness ===")
    averaged = average_duplicate_locations(dfs)
    loc_cols = ["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"]
    df_wm = (averaged["TEMPERATURE"]
             .merge(averaged["SALINITY"], on=loc_cols)
             .merge(averaged["NITRATE"], on=loc_cols))
    df_wm = add_teos10_variables(
        df_wm, sp_col="SALINITY", t_col="TEMPERATURE", pressure_col="LEV_DBAR",
    )  # adds sigma0
    df_wm["spiciness0"] = compute_spiciness(df_wm["SA"], df_wm["CT"])  # adds spiciness
    print(df_wm[["SALINITY", "TEMPERATURE", "sigma0", "spiciness0"]]
          .head(5).to_string(index=False), "\n")

    # --- Water-mass classification with user-defined regions ---
    print("=== water_mass_masks (user-defined regions) ===")
    regions = {
        "Subpolar": [(-60, 45), (0, 45), (0, 65), (-60, 65)],
        "Subtropical": [(-80, 20), (0, 20), (0, 44), (-80, 44)],
    }
    df_m = water_mass_masks(df_wm, regions)
    print(df_m["region"].value_counts().to_string(), "\n")

    # Per-region statistics
    print("=== water_mass_statistics ===")
    stats = water_mass_statistics(
        df_m, parameters=["NITRATE", "TEMPERATURE", "SALINITY", "sigma0", "spiciness0"],
    )
    print(stats.to_string(), "\n")

    # --- Distance to coast ---
    print("=== distance_to_coast ===")
    dist = distance_to_coast(df)
    df_d = df.copy()
    df_d["dist_km"] = dist.round(1)
    print(df_d[["LATITUDE", "LONGITUDE", "dist_km"]]
          .drop_duplicates(subset=["LATITUDE", "LONGITUDE"])
          .head(10).to_string(index=False))

    # --- Longhurst provinces (requires shapefile from marineregions.org) ---
    print("\n=== Longhurst provinces ===")
    longhurst_path = os.environ.get("LONGHURST_SHP_PATH")
    if longhurst_path and Path(longhurst_path).exists():
        gdf = load_longhurst(longhurst_path)
        print(f"Loaded {len(gdf)} Longhurst provinces")
        print(gdf[["province_code", "province_name"]].head(10)
              .to_string(index=False), "\n")

        df_lh = water_mass_masks(df_wm, gdf, region_name_col="province_code")
        print(df_lh["region"].value_counts().head(10).to_string(), "\n")

        stats_lh = water_mass_statistics(
            df_lh, parameters=["NITRATE", "TEMPERATURE", "SALINITY", "sigma0", "spiciness0"],
        )
        print(stats_lh.head(10).to_string())
    else:
        print("Skipped - set LONGHURST_SHP_PATH to the Longhurst .shp file to run.")

    # --- Jenniges provinces (requires cluster_set.csv from Zenodo 15201767) ---
    print("\n=== Jenniges provinces ===")
    jenniges_path = os.environ.get("JENNIGES_CSV_PATH")
    if jenniges_path and Path(jenniges_path).exists():
        grid = load_jenniges_provinces(jenniges_path, depth=0)
        print(f"Loaded {len(grid)} grid cells at depth "
              f"{grid['LEV_M'].iloc[0]:.0f} m\n")

        df_j = classify_from_grid(df_wm, grid)
        print(df_j["label"].value_counts().head(10).to_string(), "\n")

        stats_j = water_mass_statistics(
            df_j, parameters=["NITRATE", "TEMPERATURE", "SALINITY", "sigma0", "spiciness0"],
            region_col="label",
        )
        print(stats_j.head(10).to_string())
    else:
        print("Skipped - set JENNIGES_CSV_PATH to cluster_set.csv to run.")

    # --- Ocean basins (requires GOaS shapefile from marineregions.org) ---
    print("\n=== Ocean basins ===")
    basins_path = os.environ.get("BASINS_SHP_PATH")
    if basins_path and Path(basins_path).exists():
        gdf_basins = load_basins(basins_path)  # load basins
        print(f"Loaded {len(gdf_basins)} basins")
        print(gdf_basins[["basin_name"]].head(10).to_string(index=False), "\n")

        # Map data to basins
        df_basins = water_mass_masks(df_wm, gdf_basins, region_name_col="basin_name")
        print(df_basins["region"].value_counts().head(10).to_string(), "\n")

        # Per-basin statistics
        stats_basins = water_mass_statistics(
            df_basins, parameters=["NITRATE", "TEMPERATURE", "SALINITY", "sigma0", "spiciness0"],
        )
        print(stats_basins.head(10).to_string())
    else:
        print("Skipped - set BASINS_SHP_PATH to the Global Oceans and Seas .shp file to run.")

    # --- Loading data from one named basin, no lat/lon box needed (requires GOaS shapefile) ---
    print("\n=== load_comfort(region=...) - basin filtering without a lat/lon box ===")
    if basins_path and Path(basins_path).exists():
        basin_name = "North Atlantic Ocean"

        # Load one basin
        with comfort.connect(db_path) as conn:
            basin_dfs = comfort.load_comfort(
                conn,
                parameters=["NITRATE"],
                quality_flags=comfort.QC_GOOD,
                region=select_regions(basins_path, basin_name),
                as_xarray=False,
            )
        print(f"{basin_name}: {len(basin_dfs['NITRATE'])} NITRATE rows")

        # Visual check: The filtered samples should stay within the basin
        try:
            from comfort.plot import plot_spatial_distribution
        except ImportError:
            print("Install comfort-db[plot] to save a spatial-distribution map of the filtered data.")
        else:
            out_path = f"output/{basin_name.replace(' ', '_').lower()}_nitrate_map.png"
            plot_spatial_distribution(basin_dfs["NITRATE"], colourbar_label=f"NITRATE samples ({basin_name})",
                                      save_as=out_path)
            print(f"Saved spatial check to {out_path}")
    else:
        print("Skipped - set BASINS_SHP_PATH to the Global Oceans and Seas .shp file to run.")

    # --- Fay and McKinley (2014) biomes (requires their gridded biomes NetCDF file) ---
    print("\n=== Fay and McKinley biomes ===")
    biomes_path = os.environ.get("FAY_NC_PATH")
    if biomes_path and Path(biomes_path).exists():
        # Load biomes
        grid_b = load_biomes(biomes_path, biome_var="MeanBiomes")
        print(f"Loaded {len(grid_b)} grid cells\n")

        # Map data to biomes
        df_b = classify_from_grid(df_wm, grid_b)
        print(df_b["label"].value_counts().head(10).to_string(), "\n")

        # Per-biome statistics
        stats_b = water_mass_statistics(
            df_b, parameters=["NITRATE", "TEMPERATURE", "SALINITY", "sigma0", "spiciness0"],
            region_col="label",
        )
        print(stats_b.head(10).to_string())

        # Time-resolved variant
        # (for outside-grid-range years, observations are still matched to nearest available year)
        try:
            # Load time-varying biomes
            grid_tv = load_biomes(biomes_path, biome_var="TimeVaryingBiomes")
        except ValueError:
            print("\nNo time-varying biome variable in this file - skipping time-resolved demo.")
        else:
            print(f"\nTime-resolved: {len(grid_tv)} grid cells across {grid_tv['YEAR'].nunique()} years")
            # Extract the year info of the data
            df_wm_y = df_wm.assign(YEAR=pd.to_datetime(df_wm["DATEANDTIME"]).dt.year)

            # Map data to time-varying biomes
            df_tv = classify_from_grid(df_wm_y, grid_tv, time_col="YEAR")
            print(df_tv["label"].value_counts().head(10).to_string())
    else:
        print("Skipped - set FAY_NC_PATH to the Fay and McKinley biomes NetCDF file to run.")
