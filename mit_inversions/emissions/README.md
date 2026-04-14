# Emissions

## Distribution

```python
from mit_inversions.emissions import generate_emissions_distribution

ds = generate_emissions_distribution(
    total_Gg=100.0,
    method="population",  # "nightlights", "population", "uniform", "uniform_over_land"
)

ds.flux                # g/m2/s emissions flux
ds.grid_cell_area_m2   # cell area in m²
```

Custom grid:

```python
import numpy as np
ds = generate_emissions_distribution(
    total_Gg=10.0, method="population", year=2020,
    lats=np.arange(35.0, 72.0, 0.5),
    lons=np.arange(-10.0, 40.0, 0.5),
)
```

## Aggregation

```python
from mit_inversions.emissions import get_country_emissions, get_region_emissions

by_country = get_country_emissions(ds)   # DataFrame: code, name, emissions_Gg, share_of_global
by_region  = get_region_emissions(ds)    # DataFrame: region, emissions_Gg, share_of_global
```

Country codes are ADM0_A3 (e.g. CHN, USA, GBR, OCN for ocean).
Default region = CONTINENT from world_countries.gpkg.

Custom region mapping (CSV with columns `code, region`):

```python
by_region = get_region_emissions(ds, csv_path="path/to/my_regions.csv")
```

See `emissions/data/country_region_sample.csv` for the expected format.
