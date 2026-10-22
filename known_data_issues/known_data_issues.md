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
- PAR: 28,209 samples are recorded in ml/l (volume concentration cannot be converted to default µmol m^-2 s^-1).
- FLUORESCENCE: 26,677 samples are recorded in (m^-1 sr^-1)10^4 (default µg/m^3, conversion not implemented, would require instrument-specific calibration).
- TURBIDITY: 111,421 samples are recorded in NTU (default (m^-1 sr^-1)10^4, conversion not implemented).
With `convert_units=True`, `load_comfort` drops these rows by default (`on_missing="drop"`, logged); `on_missing="keep"` keeps them with their original unit.
