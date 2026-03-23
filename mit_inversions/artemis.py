# artemis.py
# Created: 16 March 2026
# Author: Eric Saboya
# Copyright (c) 2026. All rights reserved.
# License: MIT License
# 
# Description: 
#   This module implements the ARTEMIS framework for regional greenhouse gas flux
#   estimation using atmospheric observations and Bayesian modeling techniques. 
#   It includes functions for forward simulation of atmospheric transport, 
#   calculation of model error, and inference of surface fluxes.


import numpy as np
import pandas as pd
import xarray as xr

from mit_inversions.simulations import forward_simulation
from mit_inversions.model_error_methods.model_error import ModelError
from mit_inversions.sensitivity import inversion_grid_sensitivity

def artemis(data_dict_inputs: dict):
    """
    ------------------------ ARTEMIS ------------------------
    Atmospheric Regional Trace gas Emissions Modeling Inverse System (ARTEMIS)
    is a regional inversion framework that uses atmospheric observations to 
    infer surface fluxes of greenhouse gases through Bayesian modeling techniques.

    This function implements the ARTEMIS framework by performing the following steps:
    1. Forward Simulation: It creates a forward simulation of atmospheric transport
       based on the input data, which includes meteorological fields, emission inventories,
       and observation data. This step generates simulated observations based on the current 
       model parameters.

    2. Model Error Calculation: It calculates the model error by comparing the simulated 
    observations with the actual observations. This step uses a specified model error method 
    to quantify the discrepancies between the model predictions and the observed data, which 
    is crucial for improving the accuracy of the flux estimates. The output of this function 
    includes the model data dictionary, which contains the results of the forward simulation 
    and model error calculations, and the flux grid prior, which provides the prior information 
    on surface fluxes used in the inversion process.

    3. Basis Functions and Regularization: It applies basis functions to represent the spatial 
    and temporal variability of surface fluxes, and incorporates regularization techniques to 
    stabilize the inversion process and prevent overfitting to noisy observations.

    Parameters:
    - data_dict_inputs (dict): A dictionary containing all necessary input data for the ARTEMIS.
    """

    # Create forward simulation - observation data object
    model_obs_dict, flux_grid_prior = forward_simulation(data_dict_inputs)

    # Calculate model error
    model_data_dict = ModelError(model_obs_dict, 
                                 model_error_method=data_dict_inputs['model_error_method'],
                                 ).calculate()
    
    # Basis functions and mapping regions to flux-footprint grid
    fp_sens_dict_out = inversion_grid_sensitivity(data_dict_inputs, model_data_dict)
    
    


    return model_data_dict, flux_grid_prior