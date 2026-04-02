# MIT Inversions

Atmospheric inversion framework for estimating regional greenhouse gas (GHG) and ozone-depleting substance (ODS) emissions using Bayesian modeling techniques.

## Overview

MIT Inversions implements the **ARTEMIS** (Atmospheric Regional Trace gas Emissions Modeling Inverse System) framework, which combines atmospheric observations, transport models, and Bayesian inference to estimate surface fluxes of trace gases. The package provides tools for:

- **Forward simulation** of atmospheric transport using footprint-based methods
- **Bayesian inversion** with multiple inversion frameworks 
- **Basis function optimization** for spatial flux representation
- **Emissions apriori distribution** using population, nightlights, or custom spatial proxies
- **Data processing** for observations, boundary conditions, and atmospheric footprints

This is an active research project with ongoing development.

## Installation

### Prerequisites

- Python ≥ 3.8
- pip or conda package manager

### From Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/your-org/mit_inversions.git
cd mit_inversions
pip install -e .
```

### Dependencies

The package automatically installs the following dependencies:

- **Core**: numpy, pandas, xarray, scipy
- **Geospatial**: geopandas, shapely, scitools-iris
- **Units**: pint, pint-xarray
- **Machine Learning**: scikit-learn, scikit-image
- **Bayesian Inference**: pymc, arviz

## Quick Start

### Running ARTEMIS

```python
from mit_inversions.artemis import artemis

# Prepare input data dictionary with observations, footprints, and configuration
data_dict_inputs = {
    'observations': obs_data,
    'footprints': fp_data,
    'model_error_method': 'your_method',
    # ... additional configuration
}

# Run the ARTEMIS framework
model_data_dict, flux_grid_prior = artemis(data_dict_inputs)
```

### Generating Emissions Distributions

```python
from mit_inversions.emissions import generate_emissions_distribution

# Create spatially-distributed emissions using population proxy
ds = generate_emissions_distribution(
    total_Gg=100.0,
    method="population",  # Options: "nightlights", "population", "uniform", "uniform_over_land"
)

# Access results
flux = ds.flux                # g/m²/s emissions flux
area = ds.grid_cell_area_m2   # cell area in m²
```

### Aggregating Emissions by Country/Region

```python
from mit_inversions.emissions import get_country_emissions, get_region_emissions

by_country = get_country_emissions(ds)  # Returns DataFrame with ADM0_A3 codes
by_region = get_region_emissions(ds)    # Aggregates by continent or custom regions
```

## Project Structure

```
mit_inversions/
├── artemis.py              # ARTEMIS framework implementation
├── run_artemis.py          # Script for running ARTEMIS inversions
├── config.py               # Configuration management
├── simulations.py          # Forward simulation tools
├── sensitivity.py          # Sensitivity analysis for inversion grids
├── basis_functions/        # Basis function algorithms (IESD, RegionalSum)
├── data/                   # Data utilities and constants
├── emissions/              # Emissions distribution and aggregation
├── inversion/              # Bayesian inversion methods (MCMC, analytical)
├── model_error_methods/    # Model error quantification
└── readers/                # Data readers (observations, footprints, boundaries)
```

## Documentation

For more detailed information:
- See individual module READMEs (e.g., [emissions/README.md](mit_inversions/emissions/README.md))
- Explore example notebooks in [inversion/sample/](mit_inversions/inversion/sample/)
  - `test_analytical_inversion.ipynb`
  - `test_hbmcmc_inversion.ipynb`

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

This is an active research project. If you encounter issues or have suggestions, please open an issue on the repository.

## Acknowledgments

Developed at MIT for atmospheric trace gas emissions research. The ARTEMIS framework builds on established inversion methodologies used in the atmospheric science community.
