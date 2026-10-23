"""Plotting: Profiles, T-S diagram, correlations, distributions, coverage
(``comfort.plot``). Duplicate observations at the same
(station, profile, depth) are averaged before plotting.

Requires:
    COMFORT_DB_PATH - path to the COMFORT SQLite database

Install extras:
    pip install "comfort-db[plot]"
"""
from __future__ import annotations

import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dotenv import load_dotenv

import comfort
from comfort.plot import (
    boxplot,
    compare_stats,
    plot_annual_coverage,
    plot_correlation,
    plot_histogram,
    plot_profile,
    plot_section,
    plot_ts_diagram,
)
from comfort.geo import along_track_distance
from comfort.physics import add_teos10_variables
from comfort.profile_analysis import average_duplicate_records_per_profile, mixed_layer_depth, pycnocline_depth
from comfort.qc import flag_salinity_like_oxygen
from comfort.units import UnitsConverter

load_dotenv()

# A profile is identified by (ID, PROFILE_NUMBER)
# LEV_M adds the depth
JOIN_KEY = ["ID", "PROFILE_NUMBER", "LEV_M"]


if __name__ == "__main__":
    # Get path to DB from .env
    db_path = os.environ.get("COMFORT_DB_PATH", "")
    if not db_path:
        sys.exit("COMFORT_DB_PATH is not set. Point it to your COMFORT SQLite database.")

    # Load T, S, O2 from a North Atlantic subregion
    print("Loading TEMPERATURE, SALINITY, OXYGEN (North Atlantic, QC_GOOD)...")
    with comfort.connect(db_path) as conn:
        dfs = comfort.load_comfort(
            conn, parameters=["TEMPERATURE", "SALINITY", "OXYGEN"],
            quality_flags=comfort.QC_GOOD,
            lat_min=45, lat_max=65, lon_min=-30, lon_max=0,
            as_xarray=False, convert_units=True
        )
        # Check oxygen default unit (for plotting)
        oxygen_unit = UnitsConverter.from_connection(conn).default_unit_name("P_OXYGEN")
    oxygen_label = f"Oxygen [{oxygen_unit}]" if oxygen_unit else "Oxygen"

    # Drop P_OXYGEN rows that are likely mislabelled salinity values
    # (suspected COMFORT bug)
    suspect = flag_salinity_like_oxygen(dfs["OXYGEN"])
    print(f"  Dropping {suspect.sum()}/{len(dfs['OXYGEN'])} OXYGEN rows with salinity-like values")
    dfs["OXYGEN"] = dfs["OXYGEN"][~suspect].copy()

    # Average duplicate records at the same (station, profile, depth) per
    # parameter - e.g. repeated sensor readings within a single cast
    averaged = average_duplicate_records_per_profile(dfs)
    df_t, df_s, df_o = averaged["TEMPERATURE"], averaged["SALINITY"], averaged["OXYGEN"]

    # Print profile stats
    for name, df in [("TEMPERATURE", df_t), ("SALINITY", df_s), ("OXYGEN", df_o)]:
        n_prof = df[["ID", "PROFILE_NUMBER"]].drop_duplicates().shape[0]
        print(f"  {name}: {len(df)} rows, {n_prof} profiles")
    print()

    # Merge T + S and T + O for multi-parameter plots
    df_ts = df_t.merge(df_s[JOIN_KEY + ["SALINITY"]], on=JOIN_KEY)
    df_to = df_t.merge(df_o[JOIN_KEY + ["OXYGEN"]], on=JOIN_KEY)

    # --- Figure 1: Profiles, section, T-S, correlation (2x2 grid) ---
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1a. Vertical temperature profiles (first 5 stations) with mixed layer depth
    sample_ids = sorted(df_t["ID"].unique())[:5]
    df_t_sample = df_t[df_t["ID"].isin(sample_ids)]
    mld = mixed_layer_depth(df_t_sample, temperature_col="TEMPERATURE", depth_col="LEV_M")
    plot_profile(df_t_sample,
                 value_col="TEMPERATURE", xlabel="Temperature [°C]",
                 depth_markers=mld, depth_marker_col="MLD_m",
                 ax=axes[0, 0])
    axes[0, 0].set_title("Temperature profiles (dashed = MLD)")

    # 1b. Oxygen section - binned
    plot_section(df_o, value_col="OXYGEN", colourbar_label=oxygen_label, binned=True, ax=axes[0, 1])
    axes[0, 1].set_title("Oxygen section (binned mean)")

    # 1c. T-S diagram with isopycnal contours and depth colouring
    plot_ts_diagram(df_ts, temperature_col="TEMPERATURE", salinity_col="SALINITY",
                    ylabel="Temperature [°C]", xlabel="Salinity [PSU]",
                    ax=axes[1, 0])

    # 1d. Correlation: Temperature vs. oxygen - hexbin
    plot_correlation(df_to, x_col="TEMPERATURE", y_col="OXYGEN",
                     xlabel="Temperature [°C]", ylabel=oxygen_label,
                     kind="hexbin", ax=axes[1, 1])

    fig.suptitle("COMFORT - North Atlantic overview", fontsize=14)
    fig.tight_layout()

    # --- Figure 2: Density profile with pycnocline depth (same 5 stations) ---
    df_dens_sample = add_teos10_variables(
        df_t_sample.merge(df_s[JOIN_KEY + ["SALINITY"]], on=JOIN_KEY),
        sp_col="SALINITY", t_col="TEMPERATURE", pressure_col="LEV_DBAR",
    )
    pyc = pycnocline_depth(df_dens_sample, density_col="sigma0", depth_col="LEV_M")
    ax_pyc = plot_profile(df_dens_sample, value_col="sigma0",
                          xlabel=r"$\sigma_0$ [kg/m$^3$]",
                          depth_markers=pyc, depth_marker_col="pycnocline_depth_m")
    ax_pyc.set_title("Density profiles (dashed = pycnocline depth)")

    # --- Figure 3: Oxygen section along an arbitrary transect
    df_o_transect = df_o.copy()
    # Compute cumulative distances along track
    df_o_transect["DISTANCE_KM"] = along_track_distance(df_o_transect, order_col="DATEANDTIME")
    # Plot section along the track
    ax_transect = plot_section(df_o_transect, value_col="OXYGEN", colourbar_label=oxygen_label,
                               along_col="DISTANCE_KM", xlabel="Distance along track [km]",
                               binned=True)
    ax_transect.set_title("Oxygen section along cruise track")

    # --- Figure 4: Distributions (2x1) ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    boxplot(df_o, value_col="OXYGEN", title=oxygen_label, ax=axes2[0])
    plot_histogram(df_t, value_col="TEMPERATURE", title="Temperature [°C]",
                   ax=axes2[1])

    fig2.suptitle("Distributions", fontsize=14)
    fig2.tight_layout()

    # --- Figure 5: Comparative boxplots across depth groups ---
    df_combined = df_t[JOIN_KEY + ["TEMPERATURE"]].merge(
        df_s[JOIN_KEY + ["SALINITY"]], on=JOIN_KEY,
    ).merge(
        df_o[JOIN_KEY + ["OXYGEN"]], on=JOIN_KEY,
    )
    df_combined["depth_bin"] = pd.cut(
        df_combined["LEV_M"], bins=[0, 100, 500, 2000],
        labels=["Shallow", "Intermediate", "Deep"],
    )
    df_combined = df_combined.dropna(subset=["depth_bin"])

    compare_stats(
        df_combined, group_col="depth_bin",
        value_cols=["TEMPERATURE", "SALINITY", "OXYGEN"],
        value_labels={
            "TEMPERATURE": "Temperature",
            "SALINITY": "Salinity",
            "OXYGEN": "Oxygen",
        },
    )

    # --- Figure 6: Annual observation coverage ---
    plot_annual_coverage({"TEMPERATURE": df_t, "SALINITY": df_s, "OXYGEN": df_o})

    plt.show()
