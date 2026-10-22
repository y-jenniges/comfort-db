# Known data issues
This file describes known data issues in the COMFORT database. 

## 1. Salinity-like oxygen values
There are 9,507,497 oxygen values (14.7%) that resemble their co-located salinity values with a Pearson correlation of 1.0. 
After filtering with the `comfort.qc.QC_GOOD` preset, this number reduces to 329,731 samples (0.8%). 
The values can be flagged by the `comfort.qc.flag_salinity_like_oxygen` function. 

There are 54,758 oxygen values (0.135%) below 40 umol/kg that do not have co-located salinity values, so 
`flag_salinity_like_oxygen` cannot check them. Most fall within known oxygen minimum zones or are close by
(open-ocean OMZs, Black Sea, Baltic Sea, etc.), though a small fraction of unphysical values could not be ruled out.

## 2. No unit conversion formulas for PAR, FLUORESCENCE, TURBIDITY
Some samples were recorded in a unit that cannot be converted to the default parameter unit:
- PAR: 28,209 samples are recorded in ml/l (volume concentration cannot be converted to default µmol m^-2 s^-1).
- FLUORESCENCE: 26,677 samples are recorded in (m^-1 sr^-1)10^4 (default µg/m^3, conversion not implemented, would require instrument-specific calibration).
- TURBIDITY: 111,421 samples are recorded in NTU (default (m^-1 sr^-1)10^4, conversion not implemented).
With `convert_units=True`, `load_comfort` drops these rows by default (`on_missing="drop"`, logged); `on_missing="keep"` keeps them with their original unit.
