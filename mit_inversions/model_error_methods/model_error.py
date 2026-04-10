# model_error.py
# Created: 20 March 2026
# Author: Eric Saboya
# Copyright (c) 2026. All rights reserved.
# License: MIT License
# 
# Description: 
#   This module implements methods for calculating model error in the ARTEMIS framework.
#   Model error is a crucial component of the inversion process, as it quantifies the
#   discrepancies between the model predictions and the observed data, which is essential for 
#   improving the accuracy of the flux estimates. The ModelError class provides a flexible
#   interface for calculating model error using different methods, allowing users to select 
#   the most appropriate approach for their specific application


import numpy as np 
import xarray as xr 

class ModelError():
    """
    Class for calculating model errors for simulation data
    """
    def __init__(self, 
                 data: dict, 
                 model_error_method: str="pollution_event_error"
                 ):
        self.model_data_dict = data

        expected_methods = [
            "pollution_event_error",
        ]

        if model_error_method in expected_methods:
            self.model_method  = model_error_method
        else:
            raise KeyError(f"{model_error_method} is not a recognised model erorr method. Select one from {expected_methods}")

    def pollution_event_minmodel_error(self, obs, sim):
        """
        Calculate the model error for pollution events, accounting for background variability.

        Parameters:
        - obs (xarray.DataArray): 
            Observed concentrations.
        - sim (xarray.DataArray): 
            Simulated concentrations from the model.
        """
        # Background variability error
        #   Error term to account for times when there are no pollution events, 
        #   and the error is dominated by background variability.
        #   This is estimated by looking at the variability of the observations 
        #   at low percentiles, where we assume there are no pollution events.
        #   This only works for species that are not removed from the atmosphere by
        #   natural processes. 

        bg_stds = []
        bg_mu = []

        for i in [3,5,7,10,12]:
            obs_bg = obs.where(obs <= np.percentile(obs, i))
            bg_stds.append(np.nanstd(obs_bg.values))
            bg_mu.append(np.nanmean(obs_bg.values))
        N = len(bg_mu)

        # Background variability error
        var_bg = ((1/N) * np.sqrt(np.sum(np.array(bg_stds)**2)))**2 + ((1/N) * np.sum(np.array(bg_mu)-np.mean(bg_mu)))**2
        sigma_bg = np.sqrt(var_bg)


        # Pollution event error
        #   Defined as the mean ratio of observation-to-simulation pollution events
        #   at times of 3-sigma observation pollution events. This ratio is multipled
        #   with the simulations of added concentrations. 
        #   Uncertainties scale with the size of the pollution event and assumed Gaussian.

        c_bg = np.mean(bg_mu)
        delta_obs = obs - c_bg
        delta_sim = sim.mean(dim="flux_sector")
        
        # Mask of where 3-sigma pollution events occur in the array
        sig3mask = np.where(delta_obs>=np.std(delta_obs)*3)
        
        # Ratio of observation-to-simulated 3-sigma pollution events
        r = np.mean(delta_obs.values[sig3mask]/delta_sim.values[sig3mask])

        # Model error at each data point
        model_error = np.sqrt((r * delta_sim)**2 + sigma_bg**2)

        return model_error
    
    def calculate(self):
        """
        Wrapper function to calculate the model error using the selected method.
        """

        for site in self.model_data_dict.keys():
            print(f"Calculating model error for {site} ...")
            obs = self.model_data_dict[site]["mf"]
            sim = self.model_data_dict[site]["mf_sim"]

            # Calculate model error
            if self.model_method == "pollution_event_error":
                model_error = self.pollution_event_minmodel_error(obs, sim)
            

            self.model_data_dict[site]['mf_model_error'] = xr.DataArray(model_error, coords=obs.coords)
        
        return self.model_data_dict