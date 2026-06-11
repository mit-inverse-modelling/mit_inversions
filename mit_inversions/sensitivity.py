# sensitivity.py
# Created: 23 March 2026
# Author: Eric Saboya
# Copyright (c) 2026. All rights reserved.
# License: MIT License
# 
# Description:
#   This module implements the sensitivity calculation for ARTEMIS.
#   It calculates the H values on the inversion grid for each measurement site based on the choice of
#   basis function algorithm and parameters. The sensitivity calculation is a crucial step in the
#   inversion process, as it quantifies how changes in surface fluxes affect the observed concentrations
#   at the measurement sites. The function takes in the input data dictionary and the model data dictionary
#   containing the results of the forward simulation and model error calculations, and returns an updated
#   model data dictionary with H matrix information included and the basis function grid. The sensitivity
#   calculation involves preparing the mean footprint-flux grid, checking basis function input arguments,
#   calculating the basis function grid, and computing the H matrix for each site and flux sector based
#   on the selected basis function algorithm. 
#   The output includes the number of basis functions used in the inversion and the sensitivity information 
#   for each site.

import numpy as np
import pandas as pd
import xarray as xr

from mit_inversions.basis_functions.basis_function_wrapper import BasisFunctions
from mit_inversions.readers.masks import get_countries_for_grid

def inversion_grid_sensitivity(data_dict_inputs: dict, 
                               model_data_dict: dict,
                               basis_function_ds: xr.Dataset = None
                               )->dict:
    """
    Calculate the H values on the inversion grid for each measurement site based on 
    the choice of basis function algorithm and parameters.

    Parameters:
    - data_dict_inputs (dict): 
        A dictionary containing all input data for ARTEMIS, including basis function parameters.
    - model_data_dict (dict): 
        A dictionary containing the results of the forward simulation and model error calculations.

    Returns:
    - model_data_dict (dict): 
        The updated model data dictionary with H matrix information included and the basis function grid.
    """
    # Check basis function input arguments and set defaults if not specified
    basis_function_args = data_dict_inputs['basis_functions']
    
    if "country_masking" in basis_function_args.keys():
        country_masking = basis_function_args['country_masking']
    else:        
        country_masking = True

    # Calculate mean footprint-flux grid for basis function calculation
    for i, site in enumerate(model_data_dict.keys()):
        if i == 0:
            # fp_flux_grid_mean = model_data_dict[site]['fp_flux_grid'].mean(dim=('flux_sector', 'time'))
            fp_flux_grid_mean = model_data_dict[site]['fp_flux_grid'].sum(dim=('flux_sector')).mean(dim='time')
        else:
            # fp_flux_grid_mean += model_data_dict[site]['fp_flux_grid'].mean(dim=('flux_sector', 'time'))
            fp_flux_grid_mean += model_data_dict[site]['fp_flux_grid'].sum(dim=('flux_sector')).mean(dim='time')

    # Mean footprint-flux grid across all sites for basis function calculation
    fp_flux_grid_mean /= len(model_data_dict.keys())


    if basis_function_ds is None:

        if "bf_algorithm" in basis_function_args.keys():
            bf_algorithm = basis_function_args['bf_algorithm']

            if bf_algorithm == "iwasp":
                if "fp_flux_grid_error" not in basis_function_args.keys() or basis_function_args['fp_flux_grid_error'] is None:
                    fp_flux_grid_error = np.nan_to_num(1/np.sqrt(fp_flux_grid_mean), nan=0.0, posinf=0.0, neginf=0.0)
                    print("No footprint-flux grid error provided for IWASP algorithm. Defaulting to 1/sqrt(fp_flux_grid_mean).")
                else:
                    fp_flux_grid_error = basis_function_args['fp_flux_grid_error']
                print("Using IWASP basis function algorithm.")

            elif bf_algorithm == "regional_sum":
                print("Using Regional Sum basis function algorithm.")
                fp_flux_grid_error = None

        elif "bf_algorithm" is None or "bf_algorithm" not in basis_function_args.keys():
            print("No basis function algorithm specified. Defaulting to 'regional_sum'.")
            bf_algorithm = "regional_sum"
            fp_flux_grid_error = None

        if "target_regions" in basis_function_args.keys():
            target_regions = basis_function_args['target_regions']
        else:
            target_regions = 50

        # Calculate basis function grid for inversion period 
        basis_function_grid = BasisFunctions(fp_flux_grid=fp_flux_grid_mean,
                                                bf_algorithm=bf_algorithm,
                                                fp_flux_grid_error=fp_flux_grid_error,
                                                target_regions=target_regions,
                                            ).run()

        # Create xarray dataset for basis function grid
        ds_basis_function = xr.Dataset(
            {"basis_function_grid": (["latitude", "longitude"], basis_function_grid)},
            coords={"latitude": model_data_dict[site]['fp_flux_grid'].latitude,
                    "longitude": model_data_dict[site]['fp_flux_grid'].longitude,
                    },
        )

    else:
        ds_basis_function = basis_function_ds

    # Apply country masking if specified
    if country_masking:
        print("Applying country masking to basis function grid ...")
        # Retrieve country mask for model domain 
        country_grid = get_countries_for_grid(
            ds_basis_function.longitude.values,
            ds_basis_function.latitude.values,
            base_data_dir=data_dict_inputs.get("base_data_dir"),
        )
        
        # Stack basis function and country mask grids 
        country_stack = country_grid.stack(space=('latitude', 'longitude')).values
        bf_grid_stack = ds_basis_function['basis_function_grid'].stack(space=('latitude', 'longitude')).values

        df_bf_cmask = pd.DataFrame({"country": country_stack, "region": bf_grid_stack})
        df_bf_cmask['new_region'] = df_bf_cmask['region'].astype(str) + "_" + df_bf_cmask['country'].astype(str)

        # Encode new sub-regions as integers starting from 0
        df_bf_cmask['new_region_id'] = pd.factorize(df_bf_cmask['new_region'])[0]

        print(f"Original number of basis functions: {len(np.unique(bf_grid_stack))}")
        print(f"Updated number of regions after ensuring country uniqueness: {df_bf_cmask['new_region_id'].nunique()}")

        # Reshape back to original 2D grid shape
        original_shape = ds_basis_function['basis_function_grid'].shape
        new_region_grid = df_bf_cmask['new_region_id'].values.reshape(original_shape)

        ds_basis_function['basis_function_grid'].data = new_region_grid




    # Stack basis function grid
    bf_grid_stack = ds_basis_function['basis_function_grid'].stack(space=('latitude', 'longitude'))

    nbasis_functions = int(np.nanmax(bf_grid_stack.values) +1)

    basis_function_matrix = np.zeros((len(bf_grid_stack.values), nbasis_functions))
    for i in range(nbasis_functions):
        basis_function_matrix[:, i] = (bf_grid_stack.values == i).astype(int) * 1

    for site in model_data_dict.keys():
        for si, flux_sector in enumerate(model_data_dict[site]['flux_sector'].data):
            H_all_si = model_data_dict[site]["fp_flux_grid"].sel(flux_sector=flux_sector).stack(space=('latitude', 'longitude')).data

            H_grid_si = H_all_si @ basis_function_matrix

            region_name = [flux_sector + "-" + str(reg) for reg in range(nbasis_functions)]

            coords = {"region": (["region"], region_name), "time": (["time"], model_data_dict[site].coords["time"].data)}
            dimensions = ["time", "region"]
            sensitivity = xr.DataArray(H_grid_si, coords=coords, dims=dimensions)

            if si == 0:
                concat_sensitivity = sensitivity
            else:
                concat_sensitivity = xr.concat([concat_sensitivity, sensitivity], dim="region")
            
        model_data_dict[site]['H'] = concat_sensitivity
    
    model_data_dict[".basis_function_grid"] = ds_basis_function['basis_function_grid']
    
    return model_data_dict
