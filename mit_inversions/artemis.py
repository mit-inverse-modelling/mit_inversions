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

   4. Boundary Conditions: Boundary conditions for the inversion model domain are calculated 
   based the LPDM particle end locations and data from the AGAGE 12-box model. 

   5. Inversion Setup and Run: Data are prepared for the inversion and 
   the inversion using a specified inverse method (e.g., analytical, MCMC) is ran.

   6. Post-Processing: Post-processing of the inversion results into the FLUXIE format, 
   including calculating emissions by country and plotting gridded data.

   Parameters:
   - data_dict_inputs (dict): A dictionary containing all necessary input data for the ARTEMIS.
   """

   # Create forward simulation - observation data object
   model_obs_dict, flux_grid_prior = forward_simulation(data_dict_inputs)

   # Calculate model error
   model_data_dict = ModelError(model_obs_dict, 
                                model_error_method=data_dict_inputs['model_error_method'],
                                ).run()
      
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
                                           inverse_kwargs=data_dict_inputs['inversion']['inverse_kwargs'],
                                           ).run()
   
   post_processing_obj = PostProcessing(species=data_dict_inputs['species'],
                                        start_date=data_dict_inputs['start_date'],
                                        end_date=data_dict_inputs['end_date'],
                                        inversion_results=inversion_data_out,
                                        fp_sens_dict_out=fp_sens_dict_out,
                                        atmospheric_transport_model=data_dict_inputs['footprints']['lpdm'],
                                        inversion_method=data_dict_inputs['inversion']['inverse_method'],
                                        output_dir=data_dict_inputs['output_dir'],
                                        )
   # Process inversion results into FLUXIE format and calculate emissions by country
   (ds_molefraction, ds_flux) = post_processing_obj.fluxie()
   
   print("Saving emissions to NetCDF...")
   # Extract parameters for output file naming
   save_date = data_dict_inputs['start_date'][0:4]
   species = data_dict_inputs['species']
   inverse_method = data_dict_inputs['inversion']['inverse_method']
   basis_method_in = data_dict_inputs['basis_functions']['bf_algorithm']
   if basis_method_in == "regional_sum":
      basis_method = "regionalsum"
   else:
      basis_method = basis_method_in

   if data_dict_inputs["flux"]["mode"] == "customized":
      flux_method = "customized"
   elif data_dict_inputs["flux"]["mode"] == "auto_generated":
      flux_method = data_dict_inputs["flux"]["method"]

   flux_fname_out = f"ARTEMIS_{species}_fluxes_{inverse_method}_{basis_method}_{flux_method}_{save_date}.nc"
   molefraction_fname_out = f"ARTEMIS_{species}_molefraction_{inverse_method}_{basis_method}_{flux_method}_{save_date}.nc"

   ds_flux.to_netcdf(data_dict_inputs['output_dir'] / flux_fname_out)
   ds_molefraction.to_netcdf(data_dict_inputs['output_dir'] / molefraction_fname_out)

