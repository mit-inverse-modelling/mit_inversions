    # multitracer.py
# Created: 10 June 2026
# Author: Eric Saboya
# Copyright (c) 2026. All rights reserved.
# License: MIT License
# 
# Description: 
#   Module for performing multitracer inversion using atmospheric observations 
#   and Bayesian modeling techniques. This method works for two different tracer 
#   gases co-emitted from the same source (sector 1) which are linked by a ratio
#   (alpha) that is known with some uncertainty (Sa). The function takes in the
#   input data dictionary, the sensitivity outputs for both gases, the boundary
#   condition data for both gases, and the flux grids for both gases. It prepares
#   the data for the multitracer inversion, including reindexing and aligning the
#   time series, and then runs the multitracer inversion using a block matrix approach
#   to calculate the posterior estimates and uncertainties for the fluxes and boundary
#   conditions for both gases. The output is a dictionary containing the results.

import sys
import numpy as np
import pandas as pd
import xarray as xr
from mit_inversions.inversion.setup import InversionSetupRun

def multitracer_inversion(data_dict_inputs: dict,
                          fp_sens_out_gas1: dict,
                          fp_sens_out_gas2: dict,
                          gas1_data_dict_bc: dict,
                          gas2_data_dict_bc: dict,
                          flux_grid_1: xr.Dataset,
                          flux_grid_2: xr.Dataset
                          ):
    """
    Function to prepare data for multitracer inversion and to run the inversion. 
    """
    # Prepare data for multitracer inversion
    #   Gas 1
    (H_fp_concat, 
     H_bc_concat, 
     Y_concat, 
     YError_concat, 
     t_concat, 
     flux_sector_bf, 
     site_indicator,
     obs_site_names, 
     (H_fp_concat, 
      H_bc_concat, 
      Y_concat, 
      YError_concat, 
      t_concat, 
      flux_sector_bf, 
      site_indicator,
      obs_site_names, 
      bc_data_indicator) = InversionSetupRun(model_data_dict=fp_sens_out_gas1,
                                             bc_dict=gas1_data_dict_bc,
                                             flux_grid=flux_grid_1,
                                             inverse_method=data_dict_inputs['inversion']['inverse_method'],
                                             ).run_multitracer()
     #   Gas 2
     (G_fp_concat, 
      G_bc_concat, 
      Y_concat2, 
      YError_concat2, 
      t_concat2, 
      flux_sector_bf2, 
      site_indicator2, 
      obs_site_names2, 
      bc_data_indicator2) = InversionSetupRun(model_data_dict=fp_sens_out_gas2,
                                              bc_dict=gas2_data_dict_bc,
                                              flux_grid=flux_grid_2,
                                              inverse_method=data_dict_inputs['inversion']['inverse_method'],
                                              basis_function_grid=fp_sens_out_gas1['.basis_function_grid']
                                              ).run_multitracer()

    # Create standard time array for both gases (assuming same time range and frequency)
    t_standard = pd.date_range(start=data_dict_inputs['start_date'], end=data_dict_inputs['end_date'], freq="1H")
    t_standard_xr = xr.DataArray({"t_standard": (["t_standard"], t_standard)})

    # Create xarray datasets for each gas
    # Gas 1
    g1_vars = {
        "Y": (["time"], Y_concat),
        "R": (["time"], YError_concat),
        "H1": (["time", "region"], H_fp_concat[0]),
        "H2": (["time", "region"], H_fp_concat[1]),
        "Hbc": (["time", "regionBC"], H_bc_concat),
        "xa1": (["region"], flux_sector_bf.values[0]),
        "xa2": (["region"], flux_sector_bf.values[1]),
        "xbc": (["regionBC"], np.ones(H_bc_concat.shape[1], dtype=np.float64)),
    }
    g1_coords = {
        "time": (["time"], t_concat),
        "region": (["region"], flux_sector_bf['region'].values),
        "regionBC": (["regionBC"], np.array(["0", "1", "2", "3"] * int(H_bc_concat.shape[1]/4))),
    }
    g1_ds = xr.Dataset(data_vars=g1_vars, coords=g1_coords)

    # Gas 2
    g2_vars = {
        "Y": (["time"], Y_concat2),
        "R": (["time"], YError_concat2),
        "G1": (["time", "region"], G_fp_concat[0]),
        "Gbc": (["time", "regionBC"], G_bc_concat),
        "xbc2": (["regionBC"], np.ones(G_bc_concat.shape[1], dtype=np.float64)),
        }
    g2_coords = {
        "time": (["time"], t_concat2),
        "region": (["region"], flux_sector_bf2['region'].values),
        "regionBC": (["regionBC"], np.array(["0", "1", "2", "3"] * int(G_bc_concat.shape[1]/4))),
    }
    g2_ds = xr.Dataset(data_vars=g2_vars, coords=g2_coords)

    t1 = g1_ds["time"].values
    t2 = g2_ds["time"].values
    idx = np.searchsorted(t2, t1)
    idx = np.clip(idx, 0, len(t2) - 1)
    idx_prev = np.clip(idx-1, 0, len(t2) - 1)
    diff_next = np.abs(t2[idx] - t1)
    diff_prev = np.abs(t2[idx_prev] - t1)
    best_idx = np.where(diff_prev < diff_next, idx_prev, idx)

    # Keep only matches within tolerance
    tol = np.timedelta64(1, "h")
    times_to_keep = np.abs(t2[best_idx] - t1) <= tol

    i1 = np.nonzero(times_to_keep)[0]
    i2 = best_idx[times_to_keep]

    g1_ds_clean = g1_ds.isel(time=i1)
    g2_ds_clean = g2_ds.isel(time=i2)

    # Remove instances of NaNs from individual arrays
    # g1_ds = xr.Dataset(data_vars=g1_vars, coords=g1_coords)
    # mask = g1_ds["Y"].notnull()
    # g1_clean = g1_ds.sel(time=mask) 

    # g2_ds = xr.Dataset(data_vars=g2_vars, coords=g2_coords)
    # mask = g2_ds["Y"].notnull()
    # g2_clean = g2_ds.sel(time=mask)

    # # Reindex the datasets to the standard time array
    # g1_ri = g1_clean.reindex_like(t_standard_xr, method="nearest", tolerance=np.timedelta64(1, "h"))
    # g2_ri = g2_clean.reindex_like(t_standard_xr, method="nearest", tolerance=np.timedelta64(1, "h"))
    # g2_ri_g1 = g2_ri.reindex(time=g1_ri.time, method="nearest", tolerance=np.timedelta64(1, "h"))


    # Extract the relevant data for the multitracer inversion
    # Gas 1
    Y1 = np.reshape(g1_ds_clean["Y"].values, (1, -1)).T
    H1 = g1_ds_clean['H1'].values
    H2 = g1_ds_clean['H2'].values
    Hbc = g1_ds_clean['Hbc'].values
    Xa1 = np.reshape(g1_ds_clean["xa1"].values, (1, -1)).T
    Xa2 = np.reshape(g1_ds_clean["xa2"].values, (1, -1)).T
    XaBC1 = g1_ds_clean["xbc"].values.reshape(-1, 1)
    delta_mf_1 = Y1 - (H1 @ Xa1) - (H2 @ Xa2) - (Hbc @ XaBC1)

    # Gas 2
    Y2 = np.reshape(g2_ds_clean["Y"].values, (1, -1)).T
    G = g2_ds_clean['G1'].values
    Gbc = g2_ds_clean['Gbc'].values
    XaBC2 = g2_ds_clean["xbc2"].values.reshape(-1, 1)
    A_alpha = data_dict_inputs['alpha']
    Sa = data_dict_inputs['Sa']
    delta_mf_2 = Y2 - (G @ Xa1) * A_alpha - (Gbc @ XaBC2)

    # Prior uncertainty block matrix B terms (assumed to be diagonal):
    #   B11: Uncertainty on gas 1, sector 1 emissions
    #   B22: Uncertainty on gas 1, sector 2 emissions
    #   Bbc1: Uncertainty on gas 1, boundary conditions
    #   Bbc2: Uncertainty on gas 2, boundary conditions
    B11 = np.diag((data_dict_inputs['xa1_sigma'] * Xa1.flatten()) ** 2)
    B22 = np.diag((data_dict_inputs['xa2_sigma'] * Xa2.flatten()) ** 2)
    Bbc1 = np.diag((data_dict_inputs['xbc1_sigma'] * XaBC1.flatten()) ** 2)
    Bbc2 = np.diag((data_dict_inputs['xbc2_sigma'] * XaBC2.flatten()) ** 2)

    # Model-data uncertainty block matrix R terms (assumed to be diagonal):
    #   R1: Uncertainty on gas 1 observations
    #   R2: Uncertainty on gas 2 observations
    R1 = np.diag(g1_ds_clean["R"].values.flatten() ** 2)
    R2 = np.diag(g2_ds_clean["R"].values.flatten() ** 2) + H1 @ np.diag((Sa * np.ones_like(Xa1.flatten()))) @ H1.T

    print("t1", len(g1_ds_clean['time']))
    print("t2", len(g2_ds_clean['time']))
    print("Y1", np.max(Y1))
    print("Y2", np.max(Y2))
    print("H1", np.max(H1))
    print("H2", np.max(H2))
    print("Hbc", np.max(Hbc))
    print("G", np.max(G))
    print("Gbc", np.max(Gbc))
    print("R1", np.max(R1))
    print("R2", np.max(R2))
    print("B11", np.max(B11))
    print("B22", np.max(B22))
    print("Bbc1", np.max(Bbc1))
    print("Bbc2", np.max(Bbc2))
    print("A_alpha", np.max(A_alpha))
    print("Sa", np.max(Sa))

    # Run the multitracer inversion
    print("Running multitracer inversion ...")
    # Matrix S is a block matrix that has elements S11, S12, S21, S22 where:
    S11 = (H1 @ B11 @ H1.T) + (H2 @ B22 @ H2.T) + (Hbc @ Bbc1 @ Hbc.T) + R1
    S12 = (H1 @ B11 @ G.T) * A_alpha
    S21 = (G @ B11 @ H1.T) * A_alpha
    S22 = (G @ B11 @ G.T) * (A_alpha ** 2) + (Gbc @ Bbc2 @ Gbc.T) + R2

    # Construct the inverse block matrix of S
    #   Define the Schur complement, M, of S
    M = S11 -  S12 @ np.linalg.inv(S22) @ S21
    Sinv11 = np.linalg.inv(M)
    Sinv12 = - np.linalg.inv(M) @ S12 @ np.linalg.inv(S22)
    Sinv21 = - np.linalg.inv(S22) @ S21 @ np.linalg.inv(M)
    Sinv22 = np.linalg.inv(S22) + np.linalg.inv(S22) @ S21 @ np.linalg.inv(M) @ S12 @ np.linalg.inv(S22)

    # Calculate the posterior estimates for the fluxes and boundary conditions for both gases
    x1_post = Xa1 + (B11 @ H1.T @ Sinv11 @ delta_mf_1) + (B11 @ G.T @ Sinv12 @ delta_mf_2)* A_alpha
    x2_post = Xa2 + (B22 @ H2.T @ (Sinv11 @ delta_mf_1 + Sinv12 @ delta_mf_2))
    x1_bc_post = XaBC1 + (Bbc1 @ Hbc.T @ (Sinv11 @ delta_mf_1 + Sinv12 @ delta_mf_2))
    x2_bc_post = XaBC2 + (Bbc2 @ Gbc.T @ (Sinv21 @ delta_mf_1 + Sinv22 @ delta_mf_2))



    # Calculate the posterior uncertainty for the fluxes and boundary conditions for both gases
    lambda11 = (H1.T @ np.linalg.inv(R1) @ H1) + G.T @ np.linalg.inv(R2) @ G * (A_alpha ** 2) + np.linalg.inv(B11)
    lambda12 = (H1.T @ np.linalg.inv(R1) @ H2)
    lambda13 = (H1.T @ np.linalg.inv(R1) @ Hbc)
    lambda14 = (G.T @ np.linalg.inv(R2) @ Gbc) * A_alpha

    lambda21 = (H2.T @ np.linalg.inv(R1) @ H1)
    lambda22 = (H2.T @ np.linalg.inv(R1) @ H2) + np.linalg.inv(B22)
    lambda23 = (H2.T @ np.linalg.inv(R1) @ Hbc)
    lambda24 = 0.0

    lambda31 = (Hbc.T @ np.linalg.inv(R1) @ H1)
    lambda32 = (Hbc.T @ np.linalg.inv(R1) @ H2)
    lambda33 = (Hbc.T @ np.linalg.inv(R1) @ Hbc) + np.linalg.inv(Bbc1)
    lambda34 = 0.0

    lambda41 = (Gbc.T @ np.linalg.inv(R2) @ G) * A_alpha
    lambda42 = 0.0
    lambda43 = 0.0
    lambda44 = (Gbc.T @ np.linalg.inv(R2) @ Gbc) + np.linalg.inv(Bbc2)

    inversion_results = {
        "time": g1_ds_clean.time,
        "Y1": Y1,
        "Y2": Y2,
        "x1_prior": Xa1,
        "x2_prior": Xa2,
        "x1_bc_prior": XaBC1,
        "x2_bc_prior": XaBC2,
        "H1": H1,
        "H2": H2,
        "Hbc": Hbc,
        "G": G,
        "Gbc": Gbc,
        "R1": R1,
        "R2": R2,
        "A_alpha": A_alpha,
        "Sa": Sa,
        "x1_post": x1_post,
        "x2_post": x2_post,
        "x1_bc_post": x1_bc_post,
        "x2_bc_post": x2_bc_post,
        "lambda11": lambda11,
        "lambda12": lambda12,
        "lambda13": lambda13,
        "lambda14": lambda14,
        "lambda21": lambda21,
        "lambda22": lambda22,
        "lambda23": lambda23,
        "lambda24": lambda24,
        "lambda31": lambda31,
        "lambda32": lambda32,
        "lambda33": lambda33,
        "lambda34": lambda34,
        "lambda41": lambda41,
        "lambda42": lambda42,
        "lambda43": lambda43,
        "lambda44": lambda44
    }

    return inversion_results