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
import pandas as pd
import xarray as xr
from mit_inversions.inversion.inversion import analytical_inversion, ETKF_inversion, hbmcmc_inversion
from mit_inversions.inversion.mcmc_builder import make_R_prior_sigma_additive, make_x_prior_scaling

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
                 inverse_kwargs=None,
                 basis_function_grid=None):
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

        if basis_function_grid is None:
            self.basis_function_grid = self.fp_sens_dict_out[".basis_function_grid"]
        else:
            self.basis_function_grid = basis_function_grid

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
            np.ones_like(self.flux_prior_sector["flux_bf"].sum(dim="flux_sector").values),
            dtype=np.float64,
        )
        xa_bc_base = np.ones(int(n_bc), dtype=np.float64)

        # Get a priori scaling parameters from inverse_kwargs, defaults to 1 in all cases
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

        # # Create basis function grid for mapping flux grid to
        # bf_grid_stack = self.basis_function_grid.stack(space=('latitude', 'longitude')).data
        # basis_function_matrix = np.zeros((len(bf_grid_stack), np.nanmax(bf_grid_stack)+1))
        # for i in range(np.nanmax(bf_grid_stack)+1):
        #     basis_function_matrix[:, i] = (bf_grid_stack == i).astype(int)

        # # Map a priori flux grid to basis function regions
        # flux_grid_prior_sectors_bf = []
        # for si, flux_sector in enumerate(self.fp_sens_dict_out[self.site_eg]['flux_sector'].values):
        #     flux_grid_prior_s = self.flux_grid_prior['flux'].sel(flux_sector=flux_sector).stack(space=('latitude', 'longitude')).data

        #     flux_grid_prior_region = np.reshape(flux_grid_prior_s, (1, len(flux_grid_prior_s))) @ basis_function_matrix
        #     flux_grid_prior_sectors_bf.append(xr.Dataset({"flux_bf": (["region"], flux_grid_prior_region[0,:])},
        #                                            coords={"region": (["region"], self.fp_sens_dict_out[self.site_eg]['region'].values)}))
    
        # self.flux_prior_sector = xr.concat(flux_grid_prior_sectors_bf, dim="flux_sector").assign_coords(flux_sector=self.fp_sens_dict_out[self.site_eg]['flux_sector'].values)

        # Create basis function grid for mapping flux grid to
        bf_grid_stack = self.basis_function_grid.stack(space=('latitude', 'longitude')).data
        basis_function_matrix = np.zeros((len(bf_grid_stack), np.nanmax(bf_grid_stack)+1))
        for i in range(np.nanmax(bf_grid_stack)+1):
            basis_function_matrix[:, i] = (bf_grid_stack == i).astype(int)

        # Map a priori flux grid to basis function regions
        flux_grid_prior_sectors_bf = []
        for si, flux_sector in enumerate(self.fp_sens_dict_out[self.site_eg]['flux_sector'].values):
            rmask = []
            region_s = []
            for reg in self.fp_sens_dict_out[self.site_eg]['region'].values:
                if flux_sector in reg:
                    region_s.append(reg.split("-")[1])
                    rmask.append(True)
                else:
                    rmask.append(False)

            flux_grid_prior_s = self.flux_grid_prior['flux'].sel(flux_sector=flux_sector).stack(space=('latitude', 'longitude')).data
            flux_grid_prior_region = np.reshape(flux_grid_prior_s, (1, len(flux_grid_prior_s))) @ basis_function_matrix

            flux_grid_prior_sectors_bf.append(flux_grid_prior_region[0,:])
        
        flux_prior_sector = xr.Dataset({"flux_bf": (["flux_sector", "region"], np.array(flux_grid_prior_sectors_bf))},
                                       coords = {"flux_sector": (["flux_sector"], self.fp_sens_dict_out[self.site_eg]['flux_sector'].values),
                                                 "region": (["region"], np.array(region_s))}
                                       )
    
        self.flux_prior_sector = flux_prior_sector
        return self.flux_prior_sector

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

    def _build_group_index(self, site_names, times, group_mode):
        if group_mode == "none":
            return np.zeros(len(site_names), dtype=int)

        group_index = []
        for site, time_value in zip(site_names, times):
            timestamp = pd.Timestamp(time_value)
            if group_mode == "site_year":
                group_index.append(f"{site}_{timestamp.year}")
            elif group_mode == "site_month":
                group_index.append(f"{site}_{timestamp.year}_{timestamp.month:02d}")
            else:
                raise ValueError("R_group must be one of 'site_year', 'site_month', or 'none'")

        return np.asarray(group_index, dtype=object)

    def run(self):
        """
        Wrapper function to run all setup steps for the inversion.
        """

        # Map fluxes to basis function grid and add boundary condition sensitivities 
        self._map_flux_to_basis_function_grid()
        self._add_boundary_conditions_H()

        site_indicator = []
        obs_site_names = []
        bc_data_indicator = []

        for i, site in enumerate(self.sites):
            # fpXflux_bf = self.fp_sens_dict_out[site]['H']
            # H_fp = (fpXflux_bf / self.flux_prior_sector['flux_bf'].sum(dim="flux_sector")).fillna(0)
            H_fp = self.fp_sens_dict_out[site]['H']
            H_bc = self.fp_sens_dict_out[site]['Hbc']
            H_all = xr.concat([H_fp, H_bc.rename({'period_edge': 'region'})], dim='region').rename({'region': 'region_all'})

            t = self.fp_sens_dict_out[site]['time'].values
            y = self.fp_sens_dict_out[site]['mf'].values
            y_err = np.sqrt(self.fp_sens_dict_out[site]['mf_variability'].values ** 2 + self.fp_sens_dict_out[site]['mf_repeatability'].values ** 2 + self.fp_sens_dict_out[site]['mf_model_error'].values ** 2)

            site_indicator.extend([i] * len(t))
            obs_site_names.extend([site] * len(t))
            if i==0:
                bc_data_indicator.extend([0] * H_fp.shape[1])
                bc_data_indicator.extend([1] * H_bc.shape[1])

            if i == 0:
                H_fp_concat = H_fp.data
                H_bc_concat = H_bc.data
                H_concat = H_all
                Y_concat = y
                YError_concat = y_err
                t_concat = t
            else:
                H_fp_concat = np.concatenate([H_fp_concat, H_fp.data], axis=0)
                H_bc_concat = np.concatenate([H_bc_concat, H_bc.data], axis=0)
                H_concat = xr.concat([H_concat, H_all], dim="time")
                Y_concat = np.concatenate([Y_concat, y])
                YError_concat = np.concatenate([YError_concat, y_err])
                t_concat = np.concatenate([t_concat, t])

        nH = H_fp_concat.shape[1]
        nHB = H_bc_concat.shape[1]
        bc_data_indicator = np.array(bc_data_indicator, dtype=int)

        if self.inverse_method in ["analytical", "etkf"]:

            # xa is the prior mean state and xa_error is the prior covariance (P).
            xa, xa_error = self._build_prior_state_and_covariance(nHB)  

            if self.inverse_method == "analytical":
                # Perform analytical inversion to get posterior flux estimates and uncertainties
                xhat, ak, shat = analytical_inversion(H_concat.data, Y_concat, YError_concat**2, xa, xa_error)

            else:
                # ETKF expects a full observation covariance matrix.
                etkf_n = int(self.inverse_kwargs.get("N", 1000))
                default_dist_type = "gaussian" if np.allclose(xa_error, np.diag(np.diag(xa_error))) else "multivariate_normal"
                etkf_dist_type = self.inverse_kwargs.get("dist_type", default_dist_type)
                etkf_dist_params = self.inverse_kwargs.get("dist_params", None)
                etkf_random_seed = self.inverse_kwargs.get("random_seed", 42)
                # Keep the same R convention used by analytical_inversion in this workflow:
                # YError_concat / 20 is interpreted as diagonal variances.
                R_etkf = np.diag(YError_concat)

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
                "bc_data_indicator": bc_data_indicator,
                "inverse_method": self.inverse_method,
                }

            return data_dict_out

        emis_prior = self.flux_prior_sector["flux_bf"].sum(dim="flux_sector").values.astype(np.float64)
        bc_prior = np.ones(nHB, dtype=np.float64)

        emis_scaling = self.inverse_kwargs.get(
            "emis_scaling",
            {"pdf": "lognormal", "mu": 0.2, "sigma": 0.5},
        )
        bc_scaling = self.inverse_kwargs.get(
            "bc_scaling",
            {"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.5, "lower": 0.0},
        )
        R_additive = self.inverse_kwargs.get(
            "R_additive",
            {"pdf": "halfnormal", "sigma": 5.0},
        )
        R_group = self.inverse_kwargs.get("R_group", "site_year")

        x_prior = make_x_prior_scaling(nH, emis_prior, name="x_prior", scaling_prior=emis_scaling)
        bc_prior_builder = make_x_prior_scaling(nHB, bc_prior, name="bc_prior", scaling_prior=bc_scaling)
        R_var = YError_concat / 20
        R_group_index = self._build_group_index(obs_site_names, t_concat, R_group)

        R_prior = make_R_prior_sigma_additive(
            len(Y_concat),
            R=R_var,
            extra_prior=R_additive,
            group_index=R_group_index,
            name="R_prior",
        )

        idata = hbmcmc_inversion(
            H_fp_concat,
            Y_concat,
            R_prior,
            x_prior,
            H_bc=H_bc_concat,
            bc_prior_builder=bc_prior_builder,
            n_samples=self.inverse_kwargs.get("n_samples", 1e4),
            n_tune=self.inverse_kwargs.get("n_tune", 1e4),
            n_chains=self.inverse_kwargs.get("n_chains", 4),
            target_accept=self.inverse_kwargs.get("target_accept", 0.9),
            nuts_sampler=self.inverse_kwargs.get("nuts_sampler", "pymc"),
            nuts_sampler_kwargs=self.inverse_kwargs.get("nuts_sampler_kwargs"),
            progressbar=self.inverse_kwargs.get("progressbar", True),
            cores=self.inverse_kwargs.get("cores"),
            return_trace=False,
            random_seed=self.inverse_kwargs.get("random_seed", None),
            use_mvnormal_if_matrix=self.inverse_kwargs.get("use_mvnormal_if_matrix", True),
        )

        xa = np.concatenate([emis_prior, bc_prior]).reshape(-1, 1)

        data_dict_out = {
            "time": t_concat,
            "mf_obs": Y_concat,
            "mf_obs_err": YError_concat,
            "H": H_concat.data,
            "xa": xa,
            "site_indicator": np.array(site_indicator),
            "sites": self.sites,
            "bc_data_indicator": bc_data_indicator,
            "inverse_method": self.inverse_method,
            "idata": idata,
            "R_var_base": R_var,
            "R_group": R_group,
            "R_group_index": R_group_index,
            "x_prior_spec": x_prior._prior_spec,
            "bc_prior_spec": bc_prior_builder._prior_spec,
            "R_prior_spec": R_prior._prior_spec,
        }

        return data_dict_out

    def run_multitracer(self):
        """
        Wrapper function to run all setup steps for the multitracer inversion.
        """

        # Map fluxes to basis function grid and add boundary condition sensitivities 
        self._map_flux_to_basis_function_grid()
        self._add_boundary_conditions_H()

        site_indicator = []
        obs_site_names = []
        bc_data_indicator = []

        for i, site in enumerate(self.sites):
            Hfp = []
            for flux_sector in self.fp_sens_dict_out[site]['flux_sector'].values:
                region_s = []
                for reg in self.fp_sens_dict_out[site]['region'].values:
                    if flux_sector in reg:
                        region_s.append(reg)
                fpXflux_bf = self.fp_sens_dict_out[site]['H'].sel(region=region_s)
                region_s = np.array(region_s).astype(str)

                region_new = []
                for reg in region_s:
                    reg_split = reg.split("-")
                    region_new.append(reg_split[1])

                fpXflux_bf = fpXflux_bf.assign_coords(region=np.array(region_new))
                H_fp_sector = (fpXflux_bf / self.flux_prior_sector['flux_bf'].sel(flux_sector=flux_sector)).fillna(0)
                Hfp.append(H_fp_sector)

            H_fp = xr.concat(Hfp, dim="flux_sector")
            H_bc = self.fp_sens_dict_out[site]['Hbc']

            t = self.fp_sens_dict_out[site]['time'].values
            y = self.fp_sens_dict_out[site]['mf'].values

            sig1 = np.nan_to_num(self.fp_sens_dict_out[site]['mf_variability'].values) ** 2
            sig2 = np.nan_to_num(self.fp_sens_dict_out[site]['mf_repeatability'].values) ** 2
            sig3 = np.nan_to_num(self.fp_sens_dict_out[site]['mf_model_error'].values) ** 2
            y_err = np.sqrt(sig1 + sig2 + sig3)

            site_indicator.extend([i] * len(t))
            obs_site_names.extend([site] * len(t))
            if i==0:
                bc_data_indicator.extend([0] * H_fp.shape[2])
                bc_data_indicator.extend([1] * H_bc.shape[1])

            if i == 0:
                H_fp_concat = H_fp.data
                H_bc_concat = H_bc.data
                Y_concat = y
                YError_concat = y_err
                t_concat = t
            else:
                H_fp_concat = np.concatenate([H_fp_concat, H_fp.data], axis=1)
                H_bc_concat = np.concatenate([H_bc_concat, H_bc.data], axis=0)
                Y_concat = np.concatenate([Y_concat, y])
                YError_concat = np.concatenate([YError_concat, y_err])
                t_concat = np.concatenate([t_concat, t])

        nH = H_fp_concat.shape[1]
        nHB = H_bc_concat.shape[1]
        bc_data_indicator = np.array(bc_data_indicator, dtype=int)

        return H_fp_concat, H_bc_concat, Y_concat, YError_concat, t_concat, self.flux_prior_sector['flux_bf'], site_indicator, obs_site_names, bc_data_indicator