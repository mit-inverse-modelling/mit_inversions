# artemis.py
# Created: 16 March 2026

import numpy as np
import pandas as pd
import xarray as xr

from simulations import forward_simulation
from model_error_methods.model_error import ModelError


def artemis(data_dict: dict):
    """
    ------------------------ ARTEMIS Framework ------------------------
    Atmospheric Regional Trace gas Emissions Modeling Inverse System (ARTEMIS)
    is a regional inversion framework that uses atmospheric observations to 
    infer surface fluxes of greenhouse gases through Bayesian modeling techniques.
    """

    # Create forward simulation - observation data object
    model_data_dict, flux_grid_prior = forward_simulation(data_dict)

    # Calculate model error
    model_data_dict_out = ModelError(model_data_dict, model_error_method=data_dict["model_error_method"]).calculate()

    


    return model_data_dict, flux_grid_prior