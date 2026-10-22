# Known data issues
This file describes known data issues in the COMFORT database. 

## 1. Salinity-like oxygen values
There are 9,492,323 oxygen values (14.34% of the oxygen values with a co-located salinity) that are an exact copy of their 
co-located (same station, depth, bottle and profile) practical salinity values. 
All but three carry PQF2 = 4 (passed primary QC). 96.6% of them have PQF1 = 0 and are removed by the `comfort.qc.QC_GOOD` preset, leaving 321,087 values (3.4%) with PQF1 = 4.
These values can be flagged by the `comfort.qc.flag_salinity_like_oxygen` function.
The numbers come from `issue1_salinity_like_oxygen.py`.

There are 54,470 oxygen values below 40 µmol/kg (0.081% of all oxygen values) that do not have salinity values at the same station and depth, so 
`flag_salinity_like_oxygen` cannot check them. 

## 2. No unit conversion formulas for PAR, FLUORESCENCE, TURBIDITY
Some samples were recorded in a unit that cannot be converted to the default parameter unit:
- FLUORESCENCE: 26,677 rows in (m-1 sr-1)10^4 cannot be converted to μg·m-3: No allowed conversion 22 -> 9.
- PAR: 28,209 rows in ml/l cannot be converted to µmol m-2 s-1: No allowed conversion 21 -> 23.
- TURBIDITY: 111,421 rows in NTU cannot be converted to (m-1 sr-1)10^4: No allowed conversion 25 -> 22.
With `convert_units=True`, `load_comfort` drops these rows by default (`on_missing="drop"`, logged); `on_missing="keep"` keeps them with their original unit.
