"""Quality control workflows for COMFORT data,
demonstrated on SQL level and DataFrame level.

Requires:
    COMFORT_DB_PATH - path to the COMFORT SQLite database
"""
from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

import comfort
from comfort.qc import (QC_ALL, QC_GOOD, QCFilter, apply_qc_flags,
                        build_where_clause, flag_salinity_like_oxygen)

load_dotenv()

if __name__ == "__main__":
    # Get path to DB from .env
    db_path = os.environ.get("COMFORT_DB_PATH", "")
    if not db_path:
        sys.exit("COMFORT_DB_PATH is not set. Point it to your COMFORT SQLite database.")

    with comfort.connect(db_path) as conn:
        # --- Built-in presets ---
        print("=== Built-in QC presets ===")
        print(f"QC_ALL  = {QC_ALL}  (no filtering)")
        print(f"QC_GOOD = {QC_GOOD}  (PQF1 > 0 AND PQF2 > 2)\n")

        # --- SQL-level filtering (most efficient - filters before loading) ---
        print("=== SQL-level filtering (applied during load) ===")
        df_all = comfort.read_parameter(conn, "NITRATE", quality_flags=QC_ALL)
        df_good = comfort.read_parameter(conn, "NITRATE", quality_flags=QC_GOOD)
        print(f"  QC_ALL:  {len(df_all)} rows")
        print(f"  QC_GOOD: {len(df_good)} rows "
              f"({100 * len(df_good) / max(len(df_all), 1):.1f}% retained)\n")

        # --- DataFrame-level filtering (when data is already loaded) ---
        print("=== DataFrame-level filtering (post-load) ===")
        df_filtered = apply_qc_flags(df_all, QC_GOOD)
        print(f"  apply_qc_flags(df, QC_GOOD): {len(df_all)} -> {len(df_filtered)} rows\n")

        # --- Custom QC filters ---
        print("=== Custom QC filters ===")
        strict = [QCFilter("PQF1", ">0"), QCFilter("PQF2", ">=4")]
        df_strict = apply_qc_flags(df_all, strict)
        print(f"  Strict (PQF1>0, PQF2>=4): {len(df_strict)} rows")

        lenient = [QCFilter("PQF2", ">=0")]
        df_lenient = apply_qc_flags(df_all, lenient)
        print(f"  Lenient (PQF2>=0):        {len(df_lenient)} rows\n")

        # --- SQL clause building (for custom queries) ---
        print("=== SQL clause building ===")
        print(f"  build_where_clause(QC_GOOD)        = {build_where_clause(QC_GOOD)!r}")
        print(f"  build_where_clause(QC_GOOD, 't')   = {build_where_clause(QC_GOOD, 't')!r}")
        print(f"  build_where_clause(QC_GOOD, 'q',")
        print(f"    keyword='AND')                    = "
              f"{build_where_clause(QC_GOOD, 'q', keyword='AND')!r}\n")

        # --- QC flag distribution ---
        print("=== QC flag distribution (NITRATE) ===")
        if "PQF1" in df_all.columns:
            print("  PQF1:")
            print(df_all["PQF1"].value_counts().sort_index()
                  .to_string(header=False), "\n")
        if "PQF2" in df_all.columns:
            print("  PQF2:")
            print(df_all["PQF2"].value_counts().sort_index()
                  .to_string(header=False))

    # --- Content sanity check: Salinity potentially mislabelled as oxygen ---
    # Likely a bug in COMFORT
    print("\n=== Salinity-like oxygen check ===")
    data = comfort.load_comfort(db_path, parameters=["OXYGEN"], quality_flags=QC_GOOD,
                                convert_units=True, output="dict", limit=100_000)
    df_oxygen = data.get("OXYGEN")
    if df_oxygen is not None and "salinity" in df_oxygen.columns:
        suspect = flag_salinity_like_oxygen(df_oxygen)
        print(f"  {suspect.sum()} of {len(df_oxygen)} oxygen rows look like mislabelled salinity")
    else:
        print("  co-located salinity not available - skipped")
