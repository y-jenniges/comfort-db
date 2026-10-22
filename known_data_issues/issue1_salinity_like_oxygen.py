"""Analysis of the salinity-like-oxygen issue.

Every P_OXYGEN row is paired with the P_SALINITY row of the same sample (same ID, LEV_M,
BOTTLE_NUMBER, PROFILE_NUMBER). The script then analyses
  1. how many oxygen values are identical to the salinity (histogram of OXYGEN - SP),
  2. how many of these rows QC_GOOD removes,
  3. how many low oxygen values have no salinity to be checked against.

The first run reads all oxygen and salinity data from the database and caches the pairs.

Requires:
    COMFORT_DB_PATH - path to the COMFORT SQLite database
"""
import os
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from comfort.qc import QC_GOOD, apply_qc_flags, flag_salinity_like_oxygen

load_dotenv()

# Settings
CACHE_FILE =  Path("output") / "flag_threshold" / "oxygen_salinity_pairs.pkl"
FIGURE_FILE = Path("output") / "flag_threshold" / "salinity_like_oxygen_histogram.png"
CHUNK = 5_000_000
KEYS = ["ID", "LEV_M", "BOTTLE_NUMBER", "PROFILE_NUMBER"]
FLAGS = ["UNITS_ID", "PQF1", "PQF2"]
LOW_OXYGEN = 40


def iter_chunks(conn, table, columns):
    """Yield a table in rowid order, CHUNK rows at a time."""
    last_rowid = 0
    n_read = 0
    while True:
        # Query current chunk
        query = (f"SELECT rowid AS rid, {columns} FROM {table} "
                 f"WHERE rowid > ? AND VAL IS NOT NULL AND LEV_M IS NOT NULL "
                 f"ORDER BY rowid LIMIT {CHUNK}")
        chunk = pd.read_sql_query(sql=query, con=conn, params=(last_rowid,))
        if chunk.empty:
            return

        # Update row id (as int) and number of read rows
        last_rowid = int(chunk["rid"].iloc[-1])
        n_read += len(chunk)
        print(f"  {table}: {n_read / 1e6:.0f}M rows read")

        # Same dtype in every chunk, NULL bottle numbers become NaN, drop rowid
        yield chunk.drop(columns="rid").astype({key: "float64" for key in KEYS})


def build_cache(db_path):
    """Build the oxygen with co-located salinity values cache."""
    conn = sqlite3.connect(db_path)
    keys = ", ".join(KEYS)

    # Load oxygen data chunk-wise
    ox = pd.concat(iter_chunks(conn, "P_OXYGEN", f"{keys}, VAL AS OXYGEN, {', '.join(FLAGS)}"), ignore_index=True)
    ox[FLAGS] = ox[FLAGS].astype("float32")  # ensure QC flags are float
    ox_ids = ox["ID"].unique()  # get station IDs

    # Low oxygen (umol/kg, hence UNITS_ID 3) is checked against any salinity at the same (ID, LEV_M)
    low = ox.loc[(ox["OXYGEN"] < LOW_OXYGEN) & (ox["UNITS_ID"] == 3), ["ID", "LEV_M"]]
    low_index = pd.MultiIndex.from_frame(low)
    low_has_salinity = np.zeros(len(low), dtype=bool)

    parts = []
    # Iterate over salinity  data
    for sal in iter_chunks(conn, "P_SALINITY", f"{keys}, VAL AS SP"):
        # Fill the low_has_salinity matrix
        low_has_salinity |= low_index.isin(pd.MultiIndex.from_frame(sal[["ID", "LEV_M"]]))
        sal = sal[sal["ID"].isin(ox_ids)]  # ensure salinity station is in oxygen stations
        pair = sal.merge(ox, on=KEYS, how="inner")  # merge salinity and oxygen
        # A hash of the key represents the four key columns to save memory
        pair["KEY"] = pd.util.hash_pandas_object(pair[KEYS], index=False)
        parts.append(pair[["KEY", "ID", "OXYGEN", "SP"] + FLAGS])
    conn.close()

    # Clean up
    n_oxygen = len(ox)
    del ox
    pairs = pd.concat(parts, ignore_index=True)
    del parts

    # Compute statistics
    stats = {"n_oxygen": n_oxygen,
             "n_shared_keys": int(pairs.duplicated("KEY", keep=False).sum()),
             "n_low": len(low),
             "n_low_without_salinity": int((~low_has_salinity).sum())}
    pairs = pairs.drop(columns="KEY")  # remove hash of the key
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)  # create the cash dir
    pd.to_pickle({"pairs": pairs, "stats": stats}, CACHE_FILE)  # save results
    return pairs, stats


if __name__ == "__main__":
    if CACHE_FILE.exists():
        # Load from cache
        cache = pd.read_pickle(CACHE_FILE)
        pairs, stats = cache["pairs"], cache["stats"]
    else:
        if "COMFORT_DB_PATH" not in os.environ:
            # DB path must be set
            raise SystemExit("COMFORT_DB_PATH is not set and no cache exists.")
        # Build cache (chunk-wise)
        pairs, stats = build_cache(os.environ["COMFORT_DB_PATH"])

    n = len(pairs)
    print(f"{stats['n_oxygen']} oxygen rows, {n} ({100 * n / stats['n_oxygen']:.1f}%) paired with a "
          f"salinity; {stats['n_shared_keys']} paired rows share their key with another one")

    # 1. A spike at exactly 0 on a flat background, flagged by the library function
    suspect = flag_salinity_like_oxygen(pairs, salinity_col="SP")  # flag suspects
    residual = pairs["OXYGEN"] - pairs["SP"]  # compute difference between O2 and SP values

    # Histogram plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residual, bins=np.arange(-50, 50.1, 0.1), log=True)
    ax.annotate(f"{suspect.sum():,} rows with OXYGEN = SP", xy=(0, 0.95),
                xycoords=("data", "axes fraction"), xytext=(8, 0), textcoords="offset points", va="top")
    ax.set_xlabel("OXYGEN - SP [umol/kg]")
    ax.set_ylabel("rows per 0.1 umol/kg")
    fig.tight_layout()
    fig.savefig(FIGURE_FILE, dpi=200)

    # 2. QC: Do the QC-flags catch the salinity-like-oxygen values?
    suspects = pairs[suspect]
    n_good = len(apply_qc_flags(suspects, QC_GOOD))  # apply QC_GOOD
    print(f"{n_good} of {len(suspects)} flagged rows ({100 * n_good / len(suspects):.1f}%) pass QC_GOOD; "
          f"most common (PQF1, PQF2):")
    print(suspects[["PQF1", "PQF2"]].value_counts().head(5))

    # 3. Low oxygen without salinity (cannot be checked)
    print(f"\n{stats['n_low_without_salinity']} of {stats['n_low']} oxygen values below {LOW_OXYGEN} "
          f"umol/kg have no salinity at the same (ID, LEV_M)")

    plt.show()
