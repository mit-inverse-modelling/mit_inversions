# setup.py
# Created: 10 April 2026
# Author: Eric Saboya
# Copyright (c) 2026. All rights reserved.
# License: MIT License
# 
# Description:
#  This module sets up the inversion process for ARTEMIS by preparing the necessary data structures
#  and performing the required calculations to run the inversion.

import numpy as np
import xarray as xr
from mit_inversions.inversion.inversion import analytical_inversion, ETKF_inversion

class InversionSetupRun:
    """
    Class to set up and run inversion for a given set of model data, 
    boundary conditions, flux grid, and inversion method.
    """
    def __init__(self, 
                 model_data_dict, 
                 bc_dict, 
                 flux_grid, 
                 inverse_method,
                 inverse_kwargs=None):
        """
        Initialize inversion setup with model data, boundary conditions, 
        flux grid, and inversion method.
        """
        self.fp_sens_dict_out = model_data_dict
        self.model_data_dict_bc = bc_dict
        self.flux_grid_prior = flux_grid

        if inverse_method.lower() not in ["analytical", "mcmc", "etkf"]:
            raise ValueError("inverse_method must be one of 'analytical', 'mcmc', or 'etkf'")

        self.inverse_method = inverse_method.lower()
        self.inverse_kwargs = inverse_kwargs or {}

    def _build_prior_state_and_covariance(self, n_bc):
        """
        Build the prior mean state vector and prior covariance matrix.

        For analytical/ETKF inversions we support emission and boundary-condition
        scaling priors of the form:
            x_emis = x_emis_base * s_emis,   s_emis ~ N(mu_emis, sigma_emis)
            x_bc   = x_bc_base   * s_bc,     s_bc   ~ N(mu_bc, sigma_bc)

        This returns the prior mean of x and a diagonal covariance matrix derived
        from the scaling standard deviations.
        """
        xa_emis_base = np.asarray(
            self.flux_prior_sector["flux_bf"].sum(dim="flux_sector").values,
            dtype=np.float64,
        )
        xa_bc_base = np.ones(int(n_bc), dtype=np.float64)

        emis_scaling_mean = float(self.inverse_kwargs.get("emis_scaling_mean", 1.0))
        emis_scaling_sigma = float(self.inverse_kwargs.get("emis_scaling_sigma", 1.0))
        bc_scaling_mean = float(self.inverse_kwargs.get("bc_scaling_mean", 1.0))
        bc_scaling_sigma = float(self.inverse_kwargs.get("bc_scaling_sigma", 1.0))

        if emis_scaling_sigma < 0 or bc_scaling_sigma < 0:
            raise ValueError("emis_scaling_sigma and bc_scaling_sigma must be non-negative.")

        variance_floor = float(self.inverse_kwargs.get("prior_variance_floor", 1e-12))
        if variance_floor <= 0:
            raise ValueError("prior_variance_floor must be positive.")

        xa_emis = xa_emis_base * emis_scaling_mean
        xa_bc = xa_bc_base * bc_scaling_mean

        emis_var = np.maximum((xa_emis_base * emis_scaling_sigma) ** 2, variance_floor)
        bc_var = np.maximum((xa_bc_base * bc_scaling_sigma) ** 2, variance_floor)

        xa = np.concatenate([xa_emis, xa_bc])
        xa_error = np.diag(np.concatenate([emis_var, bc_var]))

        return xa, xa_error
    
    def _update_sitenames(self):
        """
        Get list of sites and an example site for indexing
        """
        sites_list = [key for key in self.fp_sens_dict_out.keys() if "." not in key]
        site_eg = sites_list[0]

        self.sites = sites_list
        self.site_eg = site_eg
    
    def _map_flux_to_basis_function_grid(self):
        """
        Maps a priori fluxes to basis function regions 
        """
        # Update site names for indexing
        self._update_sitenames()

        # Create basis function grid for mapping flux grid to
        bf_grid_stack = self.fp_sens_dict_out['.basis_function_grid'].stack(space=('latitude', 'longitude')).data
        basis_function_matrix = np.zeros((len(bf_grid_stack), np.nanmax(bf_grid_stack)+1))
        for i in range(np.nanmax(bf_grid_stack)+1):
            basis_function_matrix[:, i] = (bf_grid_stack == i).astype(int)

        # Map a priori flux grid to basis function regions
        flux_grid_prior_sectors_bf = []
        for si, flux_sector in enumerate(self.fp_sens_dict_out[self.site_eg]['flux_sector'].values):
            flux_grid_prior_s = self.flux_grid_prior['flux'].sel(flux_sector=flux_sector).stack(space=('latitude', 'longitude')).data

            flux_grid_prior_region = np.reshape(flux_grid_prior_s, (1, len(flux_grid_prior_s))) @ basis_function_matrix
            flux_grid_prior_sectors_bf.append(xr.Dataset({"flux_bf": (["region"], flux_grid_prior_region[0,:])},
                                                   coords={"region": (["region"], self.fp_sens_dict_out[self.site_eg]['region'].values)}))
    
        self.flux_prior_sector = xr.concat(flux_grid_prior_sectors_bf, dim="flux_sector").assign_coords(flux_sector=self.fp_sens_dict_out[self.site_eg]['flux_sector'].values)

    def _add_boundary_conditions_H(self):
        """
        Add boundary condition H matrix for each site to the sensitivity dictionary.
        """
        # Update site names for indexing
        self._update_sitenames()

        # Calculate simulated boundary conditions at site and concat BC H matrix with sensitivity H matrix for each site
        for site in self.sites:    
            # Get boundary condition H matrix for site 
            H_bc = self.model_data_dict_bc[site]

            # Add Hbc and simulated BC mfs to model data dictionary for site
            self.fp_sens_dict_out[site]['Hbc'] = xr.DataArray(H_bc, 
                                                              dims={
                                                                  'time': (['time'], self.fp_sens_dict_out[site]['time']),
                                                                  'period_edge': (['period_edge'], H_bc['period_edge'].values)
                                                                  })

            self.fp_sens_dict_out[site]['mf_sim_bc'] = xr.DataArray(H_bc.sum(dim='period_edge'), 
                                                                    dims={
                                                                        'time': (['time'], self.fp_sens_dict_out[site]['time'])
                                                                        })

    def run(self):
        """
        Wrapper function to run all setup steps for the inversion.
        """

        # Map fluxes to basis function grid and add boundary condition sensitivities 
        self._map_flux_to_basis_function_grid()
        self._add_boundary_conditions_H()

        if self.inverse_method in ["analytical", "etkf"]:
            site_indicator = []
            bc_data_indicator = []

            for i, site in enumerate(self.sites):
                # H at this point if the flux X footprint data for each basis function
                fpXflux_bf = self.fp_sens_dict_out[site]['H']

                for j in range(len(self.fp_sens_dict_out[site]['time'])):
                    site_indicator.append(i)

                # Sensitivity matrix for site footprints and boundary conditions
                H_fp = (fpXflux_bf / self.flux_prior_sector['flux_bf'].sum(dim="flux_sector")).fillna(0)
                H_bc = self.fp_sens_dict_out[site]['Hbc']
                H_all = xr.concat([H_fp, H_bc.rename({'period_edge': 'region'})], dim='region').rename({'region': 'region_all'})
                
                # BC data indicator: 0 for flux footprint data, 1 for BC data
                nH = H_fp.shape[1]
                nHB = H_bc.shape[1]
                for j in range(nH):
                    bc_data_indicator.append(0)
                for j in range(nHB):
                    bc_data_indicator.append(1)
                
                # Atmospheric mole fraction observations 
                y = self.fp_sens_dict_out[site]['mf'].values
                t = self.fp_sens_dict_out[site]['time'].values
                y_err = np.sqrt(
                    self.fp_sens_dict_out[site]['mf_variability'].values ** 2 
                    + self.fp_sens_dict_out[site]['mf_repeatability'].values ** 2  
                    + self.fp_sens_dict_out[site]['mf_model_error'].values ** 2
                    ) 
                
                if i == 0:
                    H_concat = H_all
                    Y_concat = y
                    YError_concat = y_err
                    t_concat = t
                else:
                    H_concat = xr.concat([H_concat, H_all], dim="time")
                    Y_concat = np.concatenate([Y_concat, y])
                    YError_concat = np.concatenate([YError_concat, y_err])
                    t_concat = np.concatenate([t_concat, t])

            # xa is the prior mean state and xa_error is the prior covariance (P).
            xa, xa_error = self._build_prior_state_and_covariance(nHB)

            if self.inverse_method == "analytical":
                # Perform analytical inversion to get posterior flux estimates and uncertainties
                xhat, ak, shat = analytical_inversion(H_concat.data, Y_concat, YError_concat / 20, xa, xa_error)
            else:
                # ETKF expects a full observation covariance matrix.
                etkf_n = int(self.inverse_kwargs.get("N", 1000))
                default_dist_type = "gaussian" if np.allclose(xa_error, np.diag(np.diag(xa_error))) else "multivariate_normal"
                etkf_dist_type = self.inverse_kwargs.get("dist_type", default_dist_type)
                etkf_dist_params = self.inverse_kwargs.get("dist_params", None)
                etkf_random_seed = self.inverse_kwargs.get("random_seed", 42)
                # Keep the same R convention used by analytical_inversion in this workflow:
                # YError_concat / 20 is interpreted as diagonal variances.
                R_etkf = np.diag(YError_concat / 20)

                xhat, shat = ETKF_inversion(
                    H_concat.data,
                    Y_concat,
                    R_etkf,
                    xa,
                    xa_error,
                    N=etkf_n,
                    dist_type=etkf_dist_type,
                    dist_params=etkf_dist_params,
                    random_seed=etkf_random_seed,
                )
                ak = None

            # Keep state vectors as column vectors for downstream post-processing.
            xhat = np.asarray(xhat, dtype=np.float64).reshape(-1, 1)
            xa = np.asarray(xa, dtype=np.float64).reshape(-1, 1)

            # Dictionary of outputs
            data_dict_out = {
                "time": t_concat,
                "mf_obs": Y_concat,
                "mf_obs_err": YError_concat,
                "H": H_concat.data,
                "xa": xa,
                "xa_error": xa_error,
                "xhat": xhat,
                "ak": ak,
                "shat": shat,
                "site_indicator": np.array(site_indicator),
                "sites": self.sites,
                "bc_data_indicator": np.array(bc_data_indicator),
                "inverse_method": self.inverse_method,
                }

            return data_dict_out
        
        else:
            return 1
