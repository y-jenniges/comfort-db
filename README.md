# comfort-db

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/y-jenniges/comfort-db/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/comfort-db)](https://pypi.org/project/comfort-db/)
[![Python](https://img.shields.io/pypi/pyversions/comfort-db)](https://pypi.org/project/comfort-db/)
[![Documentation Status](https://img.shields.io/readthedocs/comfort-db)](https://comfort-db.readthedocs.io/en/stable/)

Python library for working with the [COMFORT](https://www.doi.org/10.11582/2022.00039)[1] oceanographic SQLite dataset 
with [code documentation](https://comfort-db.readthedocs.io/en/stable/). 

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
| `geo` | shapely, geopandas, cartopy | Water masses, Longhurst provinces, coastline distance (`distance_to_coast`) |
| `grid` | xarray, netcdf4 | Spatio-temporal gridding, bathymetry masking, biome classification (`load_biomes`), xarray output |
| `scale` | scikit-learn | `ParamScaler`                                         |
| `dash` | dash, plotly | Interactive Dash explorer app                         |
| `docs` | sphinx, sphinx-rtd-theme | Building the documentation locally                    |
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

Requires Python 3.10 or newer.

---

## Quick start

The functions below find the database through the environment variable `COMFORT_DB_PATH` and the 
bathymetry file through `BATHYMETRY_PATH` 
(set it in your shell or in a `.env` file in the working directory).
Alternatively, pass the paths as first argument, e.g. `comfort.info("/path/to/comfort.sqlite")`.

```python
import logging
import matplotlib.pyplot as plt
import comfort
from comfort.plot import plot_parameter_map

# Set up logging
logging.basicConfig(level=logging.INFO)

# List all parameters from the dataset
print(comfort.list_parameters())

# Overview of the database (no data loaded, takes some minutes)
comfort.info()

# Table exploration (with quality filter)
print(comfort.describe_variables(parameters=["NITRATE", "DIC"], quality_flags=comfort.QC_GOOD))

# Filter at SQL level, get back one DataFrame per parameter
# (output="xarray" returns an xarray Dataset and needs the `grid` extra)
data = comfort.load_comfort(
    parameters=["NITRATE", "DIC"],
    quality_flags=comfort.QC_GOOD,  # quality filter
    lat_min=30, lat_max=70,  # latitude filter
    date_min="2000-01-01 00:00:00",  # date filter, format YYYY-MM-DD HH:MM:SS
    target_depths=[0, 50, 100, 200, 500, 1000],  # depth interpolation
    convert_units=True,  # convert units
    use_density=True,  # use computed densities for conversions
    limit=500_000,  # first 500,000 rows per parameter (a warning is logged if rows are cut)
    output="dict",
)

# Grid the data (needs the `grid` extra and a GEBCO bathymetry file)
grid = comfort.Grid(z_array=[0, 50, 100, 200, 500, 1000], dlon=1, dlat=1)
df_wide = grid.map_dataframes(data)  # mean per grid cell, one P_<parameter> column each

# Surface map (needs the `plot` and `geo` extras)
plot_parameter_map(df_wide, ["NITRATE"], depths=[0])
plt.show()
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```
Some gridding tests are skipped if the environmental variable `BATHYMETRY_PATH` is not set (e.g. via `export` or the `.env` file).

---

## Support and contributing
For questions, bug reports and feature ideas, please open an
[issue](https://github.com/y-jenniges/comfort-db/issues/new/choose). 
I usually respond within a week or two. 
If you would like to contribute, please see
[CONTRIBUTING.md](https://github.com/y-jenniges/comfort-db/blob/main/CONTRIBUTING.md).
The library is pre-1.0.0, so the API may still change between minor versions.

## Citing

If you use `comfort-db` in published work, please cite the COMFORT dataset [1] and this library.
The software is archived on Zenodo (DOI [10.5281/zenodo.24001495](https://doi.org/10.5281/zenodo.24001495), 
which always points to the latest version). The accompanying paper is available as an
[arXiv preprint](https://arxiv.org/abs/2611.09353):
```bibtex
@misc{jenniges2026_comfortdb,
      title = {comfort-db: A Python toolkit to access, preprocess and explore the oceanographic {COMFORT} dataset},
      author = {Yvonne Jenniges and Are Olsen and Boris Peter Koch and Sebastian Maneth},
      year = {2026},
      eprint = {2611.09353},
      url = {https://arxiv.org/abs/2611.09353},
}
```

## License

MIT
