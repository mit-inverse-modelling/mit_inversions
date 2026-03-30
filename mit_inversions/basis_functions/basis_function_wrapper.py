# basis_function_wrapper.py
# Created: 5 March 2026
# Author: Eric Saboya
# Copyright (c) 2026. All rights reserved.
# License: MIT License
# 
# Description: 
#   This module implements a wrapper class for calculating basis functions
#   used in the ARTEMIS inversion framework. The BasisFunctions class controls
#   the logic of how basis function regions are calculated based on the choice of
#   algorithm, temporal masking, and spatial masking. It provides a unified interface
#   for computing basis function grids from parsed footprint-flux grids, allowing for
#   flexibility in the selection of algorithms and masking options. The class supports
#   multiple basis function algorithms, including IWASP and Regional Sum, and incorporates 
#   options for applying land-sea and country masking to ensure that basis functions are 
#   physically meaningful and do not overlap with irrelevant regions. 
#   The calculate() method is the main function for computing the basis function grid based
#   on the specified parameters and input data.

import sys 
import glob
import numpy as np
import xarray as xr

from mit_inversions.basis_functions.algorithms.IESD import IWASP
from mit_inversions.basis_functions.algorithms.RegionalSum import RegionalSum

class BasisFunctions:
    """
    Basis function wrapper that controls the logic of how basis function
    regions are calculated by choice of algorithm, temporal masking,
    and spatial masking.

    Use the .calculate() method for computing a basis function
    grid from parsed footprint-flux grids. 

    NB. Basis functions will always be returned with land-sea 
    masking applied. This may result in the number of target
    regions being significantly higher than initially specified
    """

    def __init__(self, 
                 fp_flux_grid: xr.DataArray,
                 bf_algorithm: str,
                 data_mask: np.ndarray=None,
                 country_masking: bool=True,
                 fp_flux_grid_error: xr.DataArray=None,
                 target_regions: int=50,
                 max_iter: int=1000,
                 var_threshold=None,
                 alpha: float=1.0,
                 smooth_sigma: float=1.2,
                 ):
        """
        Initialize basis function class and parameters

        Parameters:
        - fp_flux_grid (np.ndarray):
            3D array of footprint-flux values indexed 
            [time, latitutde, longitude]
        - bf_algorithm (str):
            Basis function algorithm. 
            Select from 'regional_sum' or 'iwasp'.
        - data_mask (np.ndarray):
            Create basis functions from a data subset
            based on the input mask.
            Mask should be a list of True False elements.
            Defaults to None.
        - country_masking (bool):
            Option to apply a country mask to ensure 
            basis functions do not overlap country borders.
        - fp_flux_grid_error (np.ndarray):
            Optional flux uncertainty grid. Needed for iwasp 
            algorithm (defaults to 1/sqrt(F) if not specified).
        - target_regions (int):
            Target number of basis function regions algorithms
            should solve for.
            Defaults to 50.
        - max_iter (int):
            Maximum number of iterations to run the algorithm.
            Defaults to 1000.

        #### PARAMETERS SPECIFIC FOR IWASP ALGORITHM ####
        - var_threshold (float):
            Variability threshold for splitting regions in IWASP algorithm
            (if None, it will be set adaptively based on the distribution 
            of variability).
        - alpha (float):
            Weighting factor for the error in the composite field 
            calculation in the IWASP algorithm.
            Defaults tp 1.0
        - smooth_sigma (float):
            Sigma for Gaussian smoothing of the composite field 
            before seed detection in the IWASP algorithm.
        """

        # Footprint-flux grid
        self.fp_flux_grid = fp_flux_grid

        # Target number of basis functions
        self.target_regions = target_regions

        # Basis function algorithm
        expected_alg = ["iwasp", "regional_sum"]
        if bf_algorithm.lower() not in expected_alg:
            raise ValueError(f"{bf_algorithm} is not a recognised basis function algorithm. Use one of {expected_alg}!")
        else:
            self.algorithm = bf_algorithm
        
        # Mask to apply to time-axis of fp_flux_grid
        self.mask = data_mask
        
        # Country masking    
        self.country_masking = country_masking

        # Footprint-flux uncertainty grid
        self.fp_flux_grid_error = fp_flux_grid_error
        
        # IESD algorithm variables 
        self.max_iter = max_iter
        self.var_threshold = var_threshold
        self.alpha = alpha
        self.smooth_sigma = smooth_sigma


    def get_landsea_mask(self):
        """
        Read relevant model domain land-sea mask
        """
        return 1

    def get_country_mask(self):
        """
        Read relevant country masks for domain
        """
        return 1


    def calculate(self):
        """
        Calculate basis function grid based on specified parameters.

        Function flow
        1. Apply time axis masking if supplied 
        2. Calculate mean footprint-flux field for inversion period
        3. Calculate basis function grid from specified algorithm
        4. Apply land-sea masking (non-optional)
        5. Apply country masking if specified
        """

        # Mask data along the time axis if calculating
        # basis functions at specific times
        # if self.mask is None:
        #     grid = self.fp_flux_grid.mean(dim="time").values
        
        # else:
        #     grid = np.nanmean(self.fp_flux_grid.values[self.mask,:,:], axis=0)
        
        grid = self.fp_flux_grid

        # Calculate basis functions from specified algorithm
        if self.algorithm.lower() == "iwasp":
            bf_setup = IWASP(fp_flux_grid=grid,
                            fp_flux_grid_error=self.fp_flux_grid_error,
                            target_regions=self.target_regions,
                            max_iter=self.max_iter,
                            var_threshold=self.var_threshold,
                            alpha=self.alpha,
                            smooth_sigma=self.smooth_sigma,
                            )
            out = bf_setup.run()

            unique_arr = list(set(out[0].ravel()))
            new_grid = np.zeros_like(out[0])

            for i, arr in enumerate(unique_arr):
                ind_x, ind_y = np.where(out[0] == arr)
                new_grid[ind_x, ind_y] = i
            bf_grid = new_grid
            
        elif self.algorithm.lower() == "regional_sum":            
            bf_setup = RegionalSum(fp_flux_grid=grid,
                                   target_regions=self.target_regions,
                                   max_iter=self.max_iter,
                                   )
            bf_bucket, bf_grid = bf_setup.run()


        return bf_grid                  