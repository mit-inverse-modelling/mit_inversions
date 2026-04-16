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
from mit_inversions.inversion.setup import InversionSetupRun
from mit_inversions.sensitivity import inversion_grid_sensitivity
from mit_inversions.model_error_methods.model_error import ModelError
from mit_inversions.readers.boundary_conditions import BoundaryConditions
from mit_inversions.inversion.post_processing import PostProcessing

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

   # Boundary conditions - calculated from 12-box model
   model_data_dict_bc = BoundaryConditions(species=data_dict_inputs["species"],
                                           fp_obj=fp_sens_dict_out
                                           ).get_sensitivity_matrix()

   # Prepare data for inversion and run inversion
   inversion_data_out  = InversionSetupRun(model_data_dict=fp_sens_dict_out,
                                           bc_dict=model_data_dict_bc,
                                           flux_grid=flux_grid_prior,
                                           inverse_method=data_dict_inputs['inversion']['inverse_method'],
                                           inverse_kwargs=data_dict_inputs.get('inversion', {}),
                                           ).run()
   
   output_dir = "/home/esaboya/cfc11/results"
   post_processing_obj = PostProcessing(species=data_dict_inputs['species'],
                                        start_date=data_dict_inputs['start_date'],
                                        inversion_results=inversion_data_out, 
                                        fp_sens_dict_out=fp_sens_dict_out, 
                                        output_dir=output_dir)
   
   print("Calculating emissions by country...")
   emissions = post_processing_obj.calculate_country_emissions()
   print("Saving emissions to NetCDF...")
   emissions.to_netcdf(f'{output_dir}/country_emissions_{data_dict_inputs["species"]}_{data_dict_inputs["start_date"]}.nc')

   post_processing_obj.plot_gridded_data()
