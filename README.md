# comfort-db

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Documentation Status](https://img.shields.io/readthedocs/comfort-db)](https://comfort-db.readthedocs.io/en/latest/)

Python library for working with the [COMFORT](https://www.doi.org/10.11582/2022.00039)[1] oceanographic SQLite dataset 
with [code documentation](https://comfort-db.readthedocs.io/en/latest/). 

`comfort-db` wraps database access, preprocessing and basic analysis utilities into one library. 
This includes filtering for quality and space-time ranges, unit conversions, spatiotemporal gridding, profile and water-mass analysis, 
geospatial utilities, scaling, plotting and visualisation using a dashboard.

The data structure broadly follows oceanographic ship measurement practice: 
At a latitude, longitude and time, a station defines where measurements were taken.
At a station, profiles are measured, i.e. a parameter like temperature at different depths with a specific instrument.
There can be multiple profiles at the same station, though in the COMFORT dataset, it is mostly one profile per station. 

This library bases on and extends code developed in [2], including parts of the gridding, unit conversion, analysis and plotting 
routines.

[1] Korablev, A., Olsen, A., (2022) Geophysical Institute, University of Bergen, Bjerknes Centre for Climate Research. COMFORT Dataset [Data set]. NIRD RDA. https://doi.org/10.11582/2022.00039

[2] Jenniges, Y., Sonnewald, M., Maneth, S., Olsen, A., Koch, Boris P., (2025) Unveiling 3D ocean biogeochemical provinces in the North Atlantic: 
A systematic comparison and validation of clustering methods, Ecological Informatics, Volume 91, 103390, ISSN 1574-9541,
https://doi.org/10.1016/j.ecoinf.2025.103390.

---

## Installation

```bash
pip install comfort-db
```

Optional extras add more specialised functionality:

| Extra | Adds | Enables                                               |
|---|---|-------------------------------------------------------|
| `plot` | matplotlib, seaborn | `comfort.plot`                                        |
| `geo` | shapely, geopandas, cartopy | Water masses, Longhurst provinces, coastline distance |
| `grid` | xarray, netcdf4 | Spatio-temporal gridding, bathymetry masking, xarray output |
| `scale` | scikit-learn | `ParamScaler`                                         |
| `dash` | dash, plotly | Interactive Dash explorer app                         |
| `all` | everything above |                                                       |

Install via e.g.

```bash
pip install "comfort-db[geo,plot]"
```

**From source (editable, for development):**

```bash
git clone https://github.com/y-jenniges/comfort-db.git
cd comfort-db
pip install -e ".[dev]"
```

To build the documentation locally:

```bash
sphinx-build documentation/source documentation/build/html
```

Requires Python $\ge$ 3.10.

---

## Quick start

```python
import comfort

# Overview of the database (no data loaded, takes some minutes)
comfort.info("/path/to/comfort.sqlite")

# Filter at SQL level, get back an xarray Dataset
ds = comfort.load_comfort(
    "/path/to/comfort.sqlite",
    parameters=["NITRATE", "DIC"],
    quality_flags=comfort.QC_GOOD,  # quality filter
    lat_min=30, lat_max=70,  # latitude filter
    date_min="2000-01-01",  # date filter
    target_depths=[0, 50, 100, 200, 500, 1000],  # depth interpolation
    limit=500_000,  # safeguard for large tables
)
# ds.NITRATE -> xr.DataArray with dims (profile, depth)

# Table exploration
with comfort.connect("/path/to/comfort.sqlite") as conn:
    summary = comfort.describe_variables(conn, parameters=["NITRATE", "DIC"], quality_flags=comfort.QC_GOOD)
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest

# Include the bathymetry test (needs a GEBCO NetCDF file)
export BATHYMETRY_PATH=/path/to/gebco_bathymetry.nc
pytest
```

---

## Citing

If you use `comfort-db` in published work, please cite the COMFORT dataset [1] and this library: 
```bibtex
@article{jenniges_inprep,
    author = {Yvonne Jenniges and Are Olsen and Boris Peter Koch and Sebastian Maneth},
    title = {{comfort-db}: A Python toolkit to access, preprocess and explore the oceanographic COMFORT dataset},
    journal = {Journal of Open Source Software},
    year = {in prep.},
}
```

## License

MIT
