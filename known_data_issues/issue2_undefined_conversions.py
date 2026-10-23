import sqlite3
import os
import pandas as pd
from dotenv import load_dotenv

from comfort import UnitsConverter
from comfort.database.information import get_names_of_all_parameter_tables

load_dotenv()


if __name__ == "__main__":
    # Connect to DB
    conn = sqlite3.connect(os.environ["COMFORT_DB_PATH"])

    # Units converter and all implemented conversions
    converter = UnitsConverter.from_connection(conn)
    conversions = converter.formulas.conversions
    unit_names = converter.df_units.set_index("ID")["NAME_SHORT"]

    # Iterate over all parameter tables
    for table in get_names_of_all_parameter_tables(conn):
        print(table)

        # Get default unit
        default_unit_id = converter.default_unit_id(table)
        if default_unit_id is None:
            print(f"  No default unit ID defined for {table}. Skipping.")
            continue

        # Get all units the parameter was measured in
        query = f"SELECT UNITS_ID, COUNT(*) AS N FROM {table} GROUP BY UNITS_ID;"
        results = pd.read_sql_query(query, conn)

        # Check if the conversion is implemented and allowed for this parameter
        for unit_id, n in zip(results["UNITS_ID"], results["N"]):
            # No unit ID
            if pd.isna(unit_id):
                print(f"  {n} rows have no unit (UNITS_ID is NULL), they cannot be converted.")
                continue

            unit_id = int(unit_id)

            # Identical ID does not require conversion
            if unit_id == default_unit_id:
                continue

            # Check if conversion is listed and allowed
            if not ((unit_id, default_unit_id) in conversions
                    and converter.formulas.is_allowed(unit_id, default_unit_id, table)):
                print(f"  {n} rows in {unit_names.get(unit_id)} cannot be converted to "
                      f"{unit_names.get(default_unit_id)}: No allowed conversion {unit_id} -> {default_unit_id}.")

    # Close DB connection
    conn.close()
