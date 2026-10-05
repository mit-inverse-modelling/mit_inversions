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
                          flux_grid_2: xr.Dataset,
                          alpha: np.ndarray,
                          Sa: np.ndarray,
                          ):
    """
    Function to prepare data for multitracer inversion and to run the inversion. 
    """
    # Prepare Gas 1 data for multitracer inversion
    (
    H_fp_concat,
    H_bc_concat,
    Y_concat,
    YError_concat,
    t_concat,
    flux_sector_bf,
    site_indicator,
    obs_site_names,
    bc_data_indicator,
    ) = InversionSetupRun(
        model_data_dict=fp_sens_out_gas1,
        bc_dict=gas1_data_dict_bc,
        flux_grid=flux_grid_1,
        inverse_method=data_dict_inputs["inversion"]["inverse_method"],
    ).run_multitracer()
     
    # Prepare Gas 2 data for multitracer inversion
    (
    G_fp_concat, 
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
    
    common_time = pd.date_range(start=data_dict_inputs['start_date'], end=data_dict_inputs['end_date'], freq="h")[0:-1] 
    
    # Get unique site names
    sitenames = []
    for site in obs_site_names:
        if site not in sitenames:
            sitenames.append(site)

    for i, site in enumerate(sitenames):
        # Gas 1
        site_inds_g1 = np.where(np.array(site_indicator) == i)[0]
        g1_vars = {
            "Y": (["time"], Y_concat[site_inds_g1]),
            "R": (["time"], YError_concat[site_inds_g1]),
            "H1": (["time", "region"], H_fp_concat[0][site_inds_g1, :]),
            "H2": (["time", "region"], H_fp_concat[1][site_inds_g1, :]),
            "Hbc": (["time", "regionBC"], H_bc_concat[site_inds_g1, :]),
            "xa1": (["region"], flux_sector_bf.values[0]),
            "xa2": (["region"], flux_sector_bf.values[1]),
            "xbc": (["regionBC"], np.ones(H_bc_concat.shape[1], dtype=np.float64)),
        }
        g1_coords = {
            "time": (["time"], t_concat[site_inds_g1]),
            "region": (["region"], flux_sector_bf['region'].values),
            "regionBC": (["regionBC"], np.array(["0", "1", "2", "3"] * int(H_bc_concat.shape[1]/4))),
        }
        g1_ds = xr.Dataset(data_vars=g1_vars, coords=g1_coords)
        g1_mask = g1_ds["Y"].notnull() & g1_ds["R"].notnull()
        g1 = g1_ds.sel(time=g1_mask)

        # Gas 2
        site_inds_g2 = np.where(np.array(site_indicator2) == i)[0]
        g2_vars = {
            "Y": (["time"], Y_concat2[site_inds_g2]),
            "R": (["time"], YError_concat2[site_inds_g2]),
            "G1": (["time", "region"], G_fp_concat[0][site_inds_g2, :]),
            "Gbc": (["time", "regionBC"], G_bc_concat[site_inds_g2, :]),
            "xbc2": (["regionBC"], np.ones(G_bc_concat.shape[1], dtype=np.float64)),
            }
        g2_coords = {
            "time": (["time"], t_concat2[site_inds_g2]),
            "region": (["region"], flux_sector_bf2['region'].values),
            "regionBC": (["regionBC"], np.array(["0", "1", "2", "3"] * int(G_bc_concat.shape[1]/4))),
        }
        g2_ds = xr.Dataset(data_vars=g2_vars, coords=g2_coords)
        g2_mask = g2_ds["Y"].notnull() & g2_ds["R"].notnull()
        g2 = g2_ds.sel(time=g2_mask)

        # Reindex both datasets to a common hourly time index
        g1_hourly = g1.reindex(time=common_time, method='nearest', tolerance=pd.Timedelta('0.4h'))
        g2_hourly = g2.reindex(time=common_time, method='nearest', tolerance=pd.Timedelta('0.4h'))
        tmask = g1_hourly["Y"].notnull() & g1_hourly["R"].notnull() & g2_hourly["Y"].notnull() & g2_hourly["R"].notnull()

        if i==0:
            t_all = g1_hourly["time"].values[tmask]
            Y1 = g1_hourly["Y"].values[tmask]
            R1 = g1_hourly["R"].values[tmask]
            H1 = g1_hourly["H1"].values[tmask, :]
            H2 = g1_hourly["H2"].values[tmask, :]
            Hbc = g1_hourly["Hbc"].values[tmask, :]
            Xa1 = g1_hourly["xa1"].values
            Xa2 = g1_hourly["xa2"].values
            XaBC1 = g1_hourly["xbc"].values

            Y2 = g2_hourly["Y"].values[tmask]
            R2 = g2_hourly["R"].values[tmask]
            G = g2_hourly["G1"].values[tmask, :]
            Gbc = g2_hourly["Gbc"].values[tmask, :]
            XaBC2 = g2_hourly["xbc2"].values
        else:
            t_all = np.concatenate((t_all, g1_hourly["time"].values[tmask]), axis=0)
            Y1 = np.concatenate((Y1, g1_hourly["Y"].values[tmask]), axis=0)
            R1 = np.concatenate((R1, g1_hourly["R"].values[tmask]), axis=0)
            H1 = np.concatenate((H1, g1_hourly["H1"].values[tmask, :]), axis=0)
            H2 = np.concatenate((H2, g1_hourly["H2"].values[tmask, :]), axis=0)
            Hbc = np.concatenate((Hbc, g1_hourly["Hbc"].values[tmask, :]), axis=0)

            Y2 = np.concatenate((Y2, g2_hourly["Y"].values[tmask]), axis=0)
            R2 = np.concatenate((R2, g2_hourly["R"].values[tmask]), axis=0)
            G = np.concatenate((G, g2_hourly["G1"].values[tmask, :]), axis=0)
            Gbc = np.concatenate((Gbc, g2_hourly["Gbc"].values[tmask, :]), axis=0)

    Y1 = Y1.reshape(-1, 1)
    Y2 = Y2.reshape(-1, 1)
    Xa1 = Xa1.reshape(-1, 1)
    Xa2 = Xa2.reshape(-1, 1)
    XaBC1 = XaBC1.reshape(-1, 1)
    XaBC2 = XaBC2.reshape(-1, 1)

    delta_mf_1 = Y1 - (H1 @ Xa1) - (H2 @ Xa2) - (Hbc @ XaBC1)

    m = H1.shape[0] # No. time points
    n = H1.shape[1] # No. flux basis functions

    # Emissions ratio (alpha) and its uncertainty (Sa) for the multitracer inversion
    if np.ndim(alpha) == 1:
        A_alpha = np.diag(np.nan_to_num(alpha/Xa1.flatten()))
    elif np.ndim(alpha) == 2 and alpha.shape == (n, n):
        A_alpha = alpha
    else:
        raise ValueError("alpha must be a 1D array of length n or a 2D array of shape (n, n).")
    
    # Emission ratio alpha uncertainty (Sa) and its covariance matrix for the multitracer inversion
    if np.ndim(Sa) == 1:
        Sa = np.diag(np.nan_to_num(Sa/Xa1.flatten()))
    elif np.ndim(Sa) == 2 and Sa.shape == (n, n):
        Sa = Sa
    else:
        raise ValueError("Sa must be a 1D array of length n or a 2D array of shape (n, n).")
    
    Sa_cov = Sa ** 2

    delta_mf_2 = Y2 - (G @ A_alpha @ Xa1) - (Gbc @ XaBC2)

    # Model-data uncertainty block matrix R terms (assumed to be diagonal):
    #   R1: Uncertainty on gas 1 observations
    #   R2: Uncertainty on gas 2 observations
    #   R2_tilde: Uncertainty on gas 2 observations, plus the uncertainty from the emissions ratio (alpha) and its uncertainty (Sa)
    R1 = np.diag(R1.flatten() ** 2)
    R2 = np.diag(R2.flatten() ** 2)
    R2_tilde = R2 + (H1 @ Sa_cov @ Xa1 @ Xa1.T @ Sa_cov.T @ H1.T)

    # Prior uncertainty block matrix B terms (assumed to be diagonal):
    #     [B11  0   0    0]
    # B = [ 0   B22 0    0]
    #     [ 0   0   Bbc1 0]
    #     [ 0   0   0    Bbc2]
    #  B11: A priori uncertainty on gas 1, sector 1 emissions
    #  B22: A priori uncertainty on gas 1, sector 2 emissions
    #  Bbc1: A priori uncertainty on gas 1, boundary conditions
    #  Bbc2: A priori uncertainty on gas 2, boundary conditions
    B11 = np.diag((data_dict_inputs['xa1_sigma'] * Xa1.flatten()) ** 2) 
    B22 = np.diag((data_dict_inputs['xa2_sigma'] * Xa2.flatten()) ** 2)
    Bbc1 = np.diag((data_dict_inputs['xbc1_sigma'] * XaBC1.flatten()) ** 2)
    Bbc2 = np.diag((data_dict_inputs['xbc2_sigma'] * XaBC2.flatten()) ** 2)
 
 
    print("Running multitracer inversion ...")
    # We define the 2x2 block matrix S as S = KBK.T + R
    # S = [S11  S12]
    #     [S21  S22]
    S11 = (H1 @ B11 @ H1.T) + (H2 @ B22 @ H2.T) + (Hbc @ Bbc1 @ Hbc.T) + R1
    S12 = H1 @ B11 @ A_alpha.T @ G.T
    S21 = G @ A_alpha @ B11 @ H1.T
    S22 = (G @ A_alpha @ B11 @ A_alpha.T @ G.T) + (Gbc @ Bbc2 @ Gbc.T) + R2_tilde

    # Construct the inverse block matrix of S (Sinv)
    #   Define the Schur complement, M, of S (block S22)
    M = S11 -  S12 @ np.linalg.inv(S22) @ S21
    Sinv11 = np.linalg.inv(M)
    Sinv12 = - np.linalg.inv(M) @ S12 @ np.linalg.inv(S22)
    Sinv21 = - np.linalg.inv(S22) @ S21 @ np.linalg.inv(M)
    Sinv22 = np.linalg.inv(S22) + np.linalg.inv(S22) @ S21 @ np.linalg.inv(M) @ S12 @ np.linalg.inv(S22)

    # Calculate the posterior estimates for the fluxes and boundary conditions for both gases
    x1_post = Xa1 + (B11 @ H1.T @ (Sinv11 @ delta_mf_1 + Sinv12 @ delta_mf_2)) + (B11 @ A_alpha.T @ G.T @ (Sinv21 @ delta_mf_1 + Sinv22 @ delta_mf_2))
    x2_post = Xa2 + (B22 @ H2.T @ (Sinv11 @ delta_mf_1 + Sinv12 @ delta_mf_2))
    x1_bc_post = XaBC1 + (Bbc1 @ Hbc.T @ (Sinv11 @ delta_mf_1 + Sinv12 @ delta_mf_2))
    x2_bc_post = XaBC2 + (Bbc2 @ Gbc.T @ (Sinv21 @ delta_mf_1 + Sinv22 @ delta_mf_2))

    # Calculate the posterior uncertainty for the fluxes and boundary conditions for both gases
    # Block matrix lambda is a 4x4 block matrix 
    # NB. Not including the covariances between fluxes and BCs
    lambda11 = B11 - B11 @ H1.T @ (Sinv11 @ H1 @ B11 + Sinv12 @ G @ A_alpha @ B11)
    lambda12 = - B11 @ H1.T @ (Sinv11 @ H2 @ B22) - B11 @ A_alpha.T @ G.T @ (Sinv21 @ H2 @ B22)
    lambda21 = - B22 @ H2.T @ (Sinv11 @ H1 @ B11 + Sinv12 @ G @ A_alpha @ B11)
    lambda22 = B22 - B22 @ H2.T @ (Sinv11 @ H2 @ B22)

    lambda33 = Bbc1 - Bbc1 @ Hbc.T @ (Sinv11 @ Hbc @ Bbc1)
    lambda34 = - Bbc1 @ Hbc.T @ (Sinv12 @ Gbc @ Bbc2)
    lambda43 = - Bbc2 @ Gbc.T @ (Sinv21 @ Hbc @ Bbc1)
    lambda44 = Bbc2 - Bbc2 @ Gbc.T @ (Sinv22 @ Gbc @ Bbc2)

    # Calculate posterior precision block matrix 
    emi_post_cov = np.block([[lambda11, lambda12], [lambda21, lambda22]])
    bc_post_cov = np.block([[lambda33, lambda34], [lambda43, lambda44]])

    inversion_results = {
        "time": t_all,
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
        "R2_tilde": R2_tilde,
        "A_alpha": A_alpha,
        "Sa": Sa,
        "x1_post": x1_post,
        "x2_post": x2_post,
        "x1_bc_post": x1_bc_post,
        "x2_bc_post": x2_bc_post,
        "emi_post_cov": emi_post_cov,
        "bc_post_cov": bc_post_cov,
    }

    return inversion_results