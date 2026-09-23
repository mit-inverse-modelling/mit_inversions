
import re
import numpy as np
import pandas as pd
import xarray as xr 

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from mit_inversions.config import get_repo_parent_path
from mit_inversions.data.utils import grid_cell_area_m2, seconds_per_year
from mit_inversions.readers.masks import get_countries_for_grid, get_country_info, get_regions_for_grid


class PostProcessing:
    def __init__(self, 
                 species: str,
                 start_date: str,
                 inversion_results: dict,
                 fp_sens_dict_out: dict,
                 output_dir: str,                 
                 ):
        """
        Initialize the PostProcessing class.

        Parameters:
        -----------
        species : str
            The species for which to process inversion results.
        start_date : str
            The start date for the inversion period.
        inversion_results : dict
            A dictionary containing the inversion results.
        fp_sens_dict_out : dict
            A dictionary containing the forward model sensitivity information.
        output_dir : str
            The directory where output files will be saved.
        """
        self.species = species
        self.start_date = start_date
        self.inversion_results = inversion_results
        self.output_dir = output_dir
        self.fp_sens_dict_out = fp_sens_dict_out

        self.species_formatted = self._species_string_format()

    def _species_string_format(self) -> str:
        """
        Format the species string for use in file paths.
        """
        if "br" in self.species:
            return self.species.upper().replace("BR", "Br")
        elif "cl" in self.species:
            return self.species.upper().replace("CL", "Cl")
        else:
            return re.sub(r'^[a-zA-Z]+', lambda m: m.group().upper(), self.species)

    def process_data(self):
        """
        Process the inversion results to extract relevant data for plotting and analysis.
        """
        self.time = self.inversion_results['time']
        self.mf_obs = self.inversion_results['mf_obs']
        self.mf_obs_err = self.inversion_results['mf_obs_err']
        self.H = self.inversion_results['H']
        self.xa = np.reshape(self.inversion_results['xa'], (-1, 1))
        self.xa_error = self.inversion_results['xa_error']
        self.xhat = np.reshape(self.inversion_results['xhat'], (-1, 1))
        self.shat = self.inversion_results['shat']

        # Calculate mole fractions
        self.mf_sim = self.H.data @ self.xa
        self.mf_sim_opt = self.H.data @ self.xhat
        self.mf_sim_opt_err = self.H.data @ np.reshape(np.sqrt(np.diagonal(self.shat)), (-1, 1)) - self.mf_sim_opt

        self.sites = self.inversion_results['sites']
        self.site_indicator = self.inversion_results['site_indicator'][0]

    def calculate_gridded_fluxes(self):
        """
        Calculate gridded fluxes from the inversion results.
        """
        self.process_data()
        # Compute the a priori and a posteriori fluxes on a grid 
        bf_grid = self.fp_sens_dict_out['.basis_function_grid'].values

        flux_post = np.zeros(bf_grid.shape)
        flux_prior = np.zeros(bf_grid.shape) 

        for i in range(len(self.xa[:,0])-4):
            indy, indx = np.where(bf_grid == i)
            for j in range(len(indy)):
                flux_post[indy[j], indx[j]] += float(self.xhat[i,0]) / len(indy)
                flux_prior[indy[j], indx[j]] += float(self.xa[i,0]) / len(indy)

        self.flux_post = flux_post
        self.flux_prior = flux_prior

    def plot_mole_fractions(self):
        """
        Plot the observed and simulated mole fractions at each site, along with error bars.
        """
        self.process_data()
        sf = 1e12 
        # Plot mole fractions at each site
        for i, site in enumerate(self.sites):
            mask = np.where(self.site_indicator==i)[0]

            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(self.time[mask], self.mf_obs[mask] * sf, 'k-', label='Observations')
            ax.fill_between(self.time[mask], (self.mf_obs[mask] - self.mf_obs_err[mask]) * sf, (self.mf_obs[mask] + self.mf_obs_err[mask]) * sf, color='k', alpha=0.2)

            ax.plot(self.time[mask], self.mf_sim[mask] * sf, 'b--', label='Simulated')
            ax.plot(self.time[mask], self.mf_sim_opt[mask] * sf, 'r-', label='Optimized')
            
            ax.set_xlabel('time')
            ax.set_ylabel('Atmospheric mole fraction (ppt)')
            ax.set_title(f'{site}: ({self.species})')
            ax.legend()
            ax.grid()
            fig.tight_layout()
            plt.savefig(f'{self.output_dir}/mf_{site}_{self.species}_{self.start_date}.png', dpi=300)
            plt.close()


    def plot_gridded_data(self):
        """
        Plot the gridded fluxes (prior and posterior) and their difference on a map.
        """
        self.calculate_gridded_fluxes()
        flux_post = self.flux_post
        flux_prior = self.flux_prior

        xx, yy = np.meshgrid(self.fp_sens_dict_out['.basis_function_grid']['longitude'], self.fp_sens_dict_out['.basis_function_grid']['latitude'])

        flux_post_min = np.percentile(flux_post, 5)
        flux_post_max = np.percentile(flux_post, 95)
        flux_prior_min = np.percentile(flux_prior, 5)
        flux_prior_max = np.percentile(flux_prior, 95)


        # Plor prior fluxes 
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, color='k')
        ax.add_feature(cfeature.BORDERS, color='k')
        plot_data = ax.pcolormesh(xx, yy, flux_prior, transform=ccrs.PlateCarree(), cmap='hot_r', alpha=0.7, vmin=flux_prior_min, vmax=flux_prior_max)
        ax.set_xlim((90, 150))
        ax.set_ylim((20, 70))
        ax.set_title(self.species.upper() + ": Prior Flux")
        fig.colorbar(plot_data, ax=ax, orientation='horizontal', label='Flux (mol/m2/s)', pad=0.01)
        fig.tight_layout()
        plt.savefig(f'{self.output_dir}/flux_prior_{self.species}_{self.start_date}.png', dpi=300)
        plt.close()


        # Plot posterior fluxes
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, color='k')
        ax.add_feature(cfeature.BORDERS, color='k')
        plot_data = ax.pcolormesh(xx, yy, flux_post, transform=ccrs.PlateCarree(), cmap='bwr', alpha=0.7, vmin=flux_post_min, vmax=flux_post_max)
        ax.set_xlim((90, 150))
        ax.set_ylim((20, 70))
        ax.set_title(self.species.upper() + ": Posterior Flux")
        fig.colorbar(plot_data, ax=ax, orientation='horizontal', label='Flux (mol/m2/s)', pad=0.01)
        fig.tight_layout()
        plt.savefig(f'{self.output_dir}/flux_post_{self.species}_{self.start_date}.png', dpi=300)
        plt.close()


        # Plot difference between posterior and prior fluxes
        delta_limit = np.percentile(np.abs(flux_post-flux_prior), 95)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, color='k')
        ax.add_feature(cfeature.BORDERS, color='k')
        plot_data = ax.pcolormesh(xx, yy, flux_post-flux_prior, transform=ccrs.PlateCarree(), cmap='bwr', vmin=-delta_limit, vmax=delta_limit)
        ax.set_xlim((90, 150))
        ax.set_ylim((20, 70))
        ax.set_title(f"Posterior fluxes - Prior fluxes {self.species}")
        fig.colorbar(plot_data, ax=ax, orientation='horizontal', label='Flux (mol/m2/s)', pad=0.01)
        fig.tight_layout()
        plt.savefig(f'{self.output_dir}/flux_post_minus_prior_{self.species}_{self.start_date}.png', dpi=300)
        plt.close()


        # Plot basis function 
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, color='w')
        ax.plot(126.1616, 33.2924, marker='o', color='r', ms=9)
        ax.pcolormesh(xx, yy, self.fp_sens_dict_out['.basis_function_grid'], transform=ccrs.PlateCarree(), cmap='flag')
        ax.set_xlim((90, 150))
        ax.set_ylim((20, 70))
        ax.set_title("Basis function grid")
        fig.tight_layout()
        plt.savefig(f'{self.output_dir}/basis_function_grid_{self.species}_{self.start_date}.png', dpi=300)
        plt.close()


    def calculate_country_emissions(self):
        """
        """
        from mit_inversions.data.species_molar_masses import molarmasses
        self.calculate_gridded_fluxes()

        grid_lat = self.fp_sens_dict_out['.basis_function_grid']['latitude'].values
        grid_lon = self.fp_sens_dict_out['.basis_function_grid']['longitude'].values

        # Get country codes for each grid cell
        country_codes = get_countries_for_grid(grid_lon, grid_lat)

        # Get country info table
        country_info = list(set(country_codes.copy().data.ravel()))

        grid_area = grid_cell_area_m2(grid_lat, grid_lon)

        species_mm = 1/molarmasses[self.species_formatted]

        posterior_emissions = self.flux_post * grid_area * (3600*24*365.25) * species_mm
        prior_emissions = self.flux_prior * grid_area * (3600*24*365.25) * species_mm
        
        # Calculate total emissions for each country by summing fluxes in grid cells that belong to that country
        country_prior_emissions = []
        country_posterior_emissions = []
        country_alpha3 = []

        for code in country_info:
            mask_y, mask_x = np.where(country_codes.values == code)
            country_prior_emission = np.sum(prior_emissions[mask_y, mask_x])
            country_posterior_emission = np.sum(posterior_emissions[mask_y, mask_x])

            country_prior_emissions.append(country_prior_emission)
            country_posterior_emissions.append(country_posterior_emission)
            country_alpha3.append(code)

        ds_country_emissions = xr.Dataset({
            "prior_emissions": (["country"], country_prior_emissions),
            "posterior_emissions": (["country"], country_posterior_emissions),
        }, coords={"country": (["country"], country_alpha3)})

        ds_country_emissions['prior_emissions'].attrs['units'] = 'g/year'
        ds_country_emissions['posterior_emissions'].attrs['units'] = 'g/year'    

        return ds_country_emissions




class PostProcessingDataOutputs:
    """
    Class for post-processing inversion results to extract relevant data for plotting and analysis,
    and to calculate gridded fluxes and country-level emissions.
    """

    def __init__(self, 
                 species: str,
                 start_date: str,
                 end_date: str,
                 inversion_results: dict,
                 fp_sens_dict_out: dict,
                 flux_grid_prior: xr.Dataset,
                 atmospheric_transport_model: str, 
                 inversion_method: str,
                 output_dir: str,                 
                 ):
        """
        Initialize the PostProcessingDataOutputs class.
        Parameters:
        -----------
        species (str):
            The species for which to process inversion results.
        start_date (str):
            The start date for the inversion period.
        end_date (str):
            The end date for the inversion period.
        inversion_results (dict):
            A dictionary containing the inversion results.
        fp_sens_dict_out (dict):
            A dictionary containing the forward model sensitivity information.
        flux_grid_prior (xr.Dataset):
            A priori flux grid dataset.
        atmospheric_transport_model (str):
            The atmospheric transport model used in the inversion.
        inversion_method (str):
            The inversion method used (e.g., "analytical", "mcmc").
        output_dir (str):
            The directory where output files will be saved.
        """
        self.species = species
        self.start_date = start_date
        self.end_date = end_date
        self.inversion_results = inversion_results
        self.output_dir = output_dir
        self.fp_sens_dict_out = fp_sens_dict_out
        self.flux_grid_prior = flux_grid_prior
        self.atmospheric_transport_model = atmospheric_transport_model
        self.inversion_method = inversion_method

    def process_data(self):
        """
        Process the inversion results to extract relevant data 
        for plotting and analysis.
        """
        # time - stacked by site 
        self.time = self.inversion_results['time']

        # observed mole fractions - stacked by site 
        self.mf_obs = self.inversion_results['mf_obs']

        # total observational error - stacked by site
        self.mf_obs_err = self.inversion_results['mf_obs_err']

        # Stack the observational error components and 
        # simulated site baseline mole fractions by site
        mf_obs_variability = []
        mf_obs_repeatability = []
        mf_model_error = []
        mf_sim_bc = []

        for site in self.inversion_results['sites']:
            temp_mf_variability = self.fp_sens_dict_out[site]['mf_variability'].values
            temp_mf_repeatability = self.fp_sens_dict_out[site]['mf_repeatability'].values
            temp_mf_model_error = self.fp_sens_dict_out[site]['mf_model_error'].values
            temp_mf_sim_bc = self.fp_sens_dict_out[site]['mf_sim_bc'].values 
        
            for i in range(len(temp_mf_variability)):
                mf_obs_variability.append(temp_mf_variability[i])
                mf_obs_repeatability.append(temp_mf_repeatability[i])
                mf_model_error.append(temp_mf_model_error[i])
                mf_sim_bc.append(temp_mf_sim_bc[i])
        
        self.mf_obs_variability = np.array(mf_obs_variability)
        self.mf_obs_repeatability = np.array(mf_obs_repeatability)
        self.mf_model_error = np.array(mf_model_error)
        self.mf_sim_bc = np.array(mf_sim_bc)

        self.H = self.inversion_results['H']
        self.xa  = self.inversion_results['xa']
        self.xa_error = self.inversion_results['xa_error']
        self.xhat = self.inversion_results['xhat']
        self.shat = self.inversion_results['shat']

        # simulated mole fractions - stacked by site 
        self.mf_sim = self.H.data @ self.xa
        self.mf_sim_opt = self.H.data @ self.xhat.flatten()
        
        self.mf_sim_opt_err = (np.diag(self.H.data @ self.shat @ np.transpose(self.H.data)) + self.mf_model_error**2)**0.5
        self.mf_opt_percentile = np.array([self.mf_sim_opt-self.mf_sim_opt_err, self.mf_sim_opt+self.mf_sim_opt_err]).T

        self.sites = self.inversion_results['sites']
        self.site_indicator = self.inversion_results['site_indicator']
        
        bc_indicator = np.diag(self.inversion_results['bc_data_indicator'])
        self.bc_indicator = self.inversion_results['bc_data_indicator']
        self.mf_sim_opt_bc = self.H @ bc_indicator @ self.xhat.flatten()

    def _get_site_geo_info(self):
        """
        Method to retrieve the longitude, latitude, altitude 
        for each site
        """
        # Read site information json file 
        import json 
        # import mit_inversions.mit_inversions.data.site_data.json as site_data
        repo_path = get_repo_parent_path()
        with open(repo_path / "mit_inversions" / "data" / "site_data.json") as f:
            site_info_f = json.load(f)
        
        # Get site names and siteindicator values
        sites = self.inversion_results['sites']
        siteindicator = self.inversion_results['site_indicator']

        longitudes, latitudes, altitudes = [], [], []
        for i, site in enumerate(sites):
            site_info_dict = site_info_f[site]
            _network = list(site_info_dict.keys())[0]
            N_site_index = len(np.where(siteindicator == i)[0])

            for j in range(N_site_index):
                longitudes.append(site_info_dict[_network]['longitude'])
                latitudes.append(site_info_dict[_network]['latitude'])
                altitudes.append(site_info_dict[_network]['height_station_masl'])

        self.longitude = np.array(longitudes)
        self.latitude = np.array(latitudes)
        self.altitude = np.array(altitudes)

    def _get_time_bounds(self):
        """
        Method to calculate time bounds for each observation based on the time array
        """
        time_bnds = []
        for i in range(len(self.time)-1):
            time_bnds.append([self.time[i], self.time[i+1]])
        time_bnds.append([self.time[-1], self.time[-1] + (self.time[-1] - self.time[-2])])  # Extrapolate last time bound
        self.time_bnds = np.array(time_bnds)

    def calculate_gridded_fluxes(self):
        """
        Calculate gridded fluxes from the inversion results.
        """
        self.process_data()

        # Compute the a priori and a posteriori fluxes on a grid 
        bf_grid = self.fp_sens_dict_out[".basis_function_grid"].values

        flux_post = np.zeros(bf_grid.shape)
        flux_prior = self.flux_grid_prior.flux.sum("flux_sector").values
        flux_post_lower_percentile = np.zeros(bf_grid.shape)
        flux_post_upper_percentile = np.zeros(bf_grid.shape)

        xa_count = 0 # Counter to track the index of non-BC grid points in xa 
        for i in range(len(self.xa[:,0])):
            if self.bc_indicator[i] == 0:  # Only consider non-bc grid points
                indy, indx = np.where(bf_grid == xa_count)
                for j in range(len(indy)):
                    flux_post[indy[j], indx[j]] = flux_prior[indy[j], indx[j]] * float(self.xhat[i,0]) 
                    flux_post_lower_percentile[indy[j], indx[j]] = flux_prior[indy[j], indx[j]] *  float(self.shat[i, i]) 
                    flux_post_upper_percentile[indy[j], indx[j]] = flux_prior[indy[j], indx[j]] * float(self.shat[i, i]) 
                xa_count += 1

        mycoords = {'latitude': self.fp_sens_dict_out['.basis_function_grid']['latitude'].values,
                    'longitude': self.fp_sens_dict_out['.basis_function_grid']['longitude'].values}

        self.flux_post = xr.Dataset({'flux': (['latitude', 'longitude'], flux_post)}, coords=mycoords)
        self.flux_prior = xr.Dataset({'flux': (['latitude', 'longitude'], flux_prior)}, coords=mycoords)
        
        self.flux_post_error = np.array([flux_post-flux_post_lower_percentile, flux_post+flux_post_upper_percentile])
        # self.longitude_grid = self.fp_sens_dict_out['.basis_function_grid']['longitude'].values
        # self.latitude_grid = self.fp_sens_dict_out['.basis_function_grid']['latitude'].values

    def calculate_country_emissions(self, subregion=None)->xr.Dataset:
        """
        Method to calculate emission totals for countries in model domain
        """
        from mit_inversions.data.species_molar_masses import molarmasses
        self.process_data()
        self.calculate_gridded_fluxes()

        grid_lat = self.fp_sens_dict_out[".basis_function_grid"]["latitude"].values
        grid_lon = self.fp_sens_dict_out[".basis_function_grid"]["longitude"].values

        # Get country codes for each grid cell
        if subregion == None:
            country_codes = get_countries_for_grid(grid_lon, grid_lat)
        else:
            country_codes = get_regions_for_grid(subregion, grid_lon, grid_lat)

        # Get country info table
        country_info = np.sort(list(set(country_codes.copy().data.ravel())))

        # Calculate the geographical area of each grid cell in m2
        print("Calculating grid cell areas ...")
        grid_area = grid_cell_area_m2(grid_lat, grid_lon)
        self.grid_area = grid_area
        species_mm = molarmasses[self.species]

        year = pd.to_datetime(self.start_date).year
        seconds_year = seconds_per_year(year)
        print("Calculating country emissions ...")

        # Convert fluxes to emissions in g/year for each grid cell
        prior_emissions = self.flux_prior['flux'] * grid_area['area'] * seconds_year * species_mm
        
        # Build linear mapping from state vector to country emissions so full
        # covariance (including off-diagonal terms) can be propagated.
        bf_grid = self.fp_sens_dict_out[".basis_function_grid"].values
        xhat_flat = np.asarray(self.xhat).reshape(-1)
        shat = np.asarray(self.shat, dtype=float)
        n_state = xhat_flat.size

        # Calculate total emissions for each country by summing fluxes in grid cells that belong to that country
        country_prior_emissions = []
        country_posterior_emissions = []
        country_posterior_emissions_lower = []
        country_posterior_emissions_upper = []
        country_alpha3 = []
        A = np.zeros((len(country_info), n_state), dtype=float)

        prior_cell_emissions = prior_emissions.values
        bc_indicator = np.asarray(self.bc_indicator).astype(int)

        for country_idx, code in enumerate(country_info):
            mask = (country_codes == code) * 1.0
            country_prior_emi = (prior_emissions.values * mask).sum()

            # Build weights for non-BC state elements (BC elements remain zero).
            xa_count = 0
            for state_idx in range(n_state):
                if bc_indicator[state_idx] == 0:
                    region_mask = (bf_grid == xa_count)
                    A[country_idx, state_idx] = (prior_cell_emissions * mask * region_mask).sum()
                    xa_count += 1

            country_posterior_emi = float(A[country_idx, :] @ xhat_flat)

            country_prior_emissions.append(country_prior_emi)
            country_posterior_emissions.append(country_posterior_emi)
            country_alpha3.append(code)

        # Propagate full posterior covariance: Cov_country = A * Shat * A^T
        country_cov = A @ shat @ A.T
        country_sigma = np.sqrt(np.clip(np.diag(country_cov), a_min=0.0, a_max=None))

        for i in range(len(country_posterior_emissions)):
            country_posterior_emissions_lower.append(country_posterior_emissions[i] - country_sigma[i])
            country_posterior_emissions_upper.append(country_posterior_emissions[i] + country_sigma[i])

        ds_country_emissions = xr.Dataset({
            "prior_emissions": (["country"], country_prior_emissions),
            "posterior_emissions": (["country"], country_posterior_emissions),
            "posterior_emissions_lower": (["country"], country_posterior_emissions_lower),
            "posterior_emissions_upper": (["country"], country_posterior_emissions_upper),
        }, coords={"country": (["country"], country_alpha3)})

        ds_country_emissions['prior_emissions'].attrs['units'] = 'g/year'
        ds_country_emissions['posterior_emissions'].attrs['units'] = 'g/year'
        ds_country_emissions['posterior_emissions_lower'].attrs['units'] = 'g/year'
        ds_country_emissions['posterior_emissions_upper'].attrs['units'] = 'g/year'

        return ds_country_emissions



    def fluxie(self):
        """
        Process inversion results in concentration and flux formats that align 
        with FLUXIE formatting
        """
        def global_attrs(ds):
            ds.attrs["title"] = f"ARTEMIS inversion results for {self.species}"
            ds.attrs["institution"] = "Massachusetts Institute of Technology (MIT)"
            ds.attrs["creator"] = "MIT Atmospheric Inverse Modeling Group"
            ds.attrs["creation_date"] = pd.Timestamp.now().isoformat()
            ds.attrs["contact"] = "lwestern@mit.edu, esaboya@mit.edu, mindean@mit.edu"
            ds.attrs["transport_model"] = self.atmospheric_transport_model

            ds.attrs["inversion_system"] = "ARTEMIS"
            ds.attrs["inversion_method"] = self.inversion_method
            ds.attrs["species"] = self.species
            ds.attrs['conventions'] = "CF-1.8"
            ds.attrs["license"] = "CC-BY-4.0"
            return ds

        def concentration_attrs(ds):
            """
            Add attributes to the concentration dataset variables according to FLUXIE formatting guidelines
            """
            ds["longitude"].attrs["long_name"] = "sample_longitude_in_decimal_degrees"
            ds["longitude"].attrs["units"] = "degrees_east"
            ds["longitude"].attrs["comment"] = "Longitude at which air sample was collected."
            ds["longitude"].attrs["standard_name"] = "longitude"

            ds["latitude"].attrs["long_name"] = "sample_latitude_in_decimal_degrees"
            ds["latitude"].attrs["units"] = "degrees_north"
            ds["latitude"].attrs["comment"] = "Latitude at which air sample was collected."
            ds["latitude"].attrs["standard_name"] = "latitude"

            ds["altitude"].attrs["long_name"] = "sample_altitude_in_decimal_degrees"
            ds["altitude"].attrs["units"] = "m"
            ds["altitude"].attrs["comment"] = "Altitude (surface elevation plus sample intake height) at which air sample was collected."
            ds["altitude"].attrs["standard_name"] = "altitude"

            ds["number_of_identifier"].attrs["long_name"] = "Index of identifier of observing platform"
            ds["number_of_identifier"].attrs["units"] = "1"

            # Observations
            ds["mf_observed"].attrs["units"] = "mol mol-1"
            ds["mf_observed"].attrs["long_name"] = "observed mole fraction of co2 in dry air"

            ds["stdev_mf_observed_repeatability"].attrs["units"] = "mol mol-1"
            ds["stdev_mf_observed_repeatability"].attrs["long_name"] = "repeatability uncertainty of observed mole fraction"
            ds["stdev_mf_observed_repeatability"].attrs["comment"] = "understood as combined analytical uncertainty"

            ds["stdev_mf_observed_variability"].attrs["units"] = "mol mol-1"
            ds["stdev_mf_observed_variability"].attrs["long_name"] = "variability uncertainty of observed mole fraction"

            ds["stdev_mf_model"].attrs["units"] = "mol mol-1"
            ds["stdev_mf_model"].attrs["long_name"] = "model uncertainty of simulated mole fraction"

            ds["stdev_mf_total"].attrs["units"] = "mol mol-1"
            ds["stdev_mf_total"].attrs["long_name"] = "total model-data-mismatch uncertainty applied in inversion"
            
            # Simulations
            ds["mf_prior"].attrs["units"] = "mol mol-1"
            ds["mf_prior"].attrs["long_name"] = "prior simulated mole fraction of co2 in dry air"
            ds["mf_posterior"].attrs["units"] = "mol mol-1"
            ds["mf_posterior"].attrs["long_name"] = "posterior simulated mole fraction of co2 in dry air"
            ds["mf_bc_prior"].attrs["units"] = "mol mol-1"
            ds["mf_bc_prior"].attrs["long_name"] = "prior simulated boundary condition mole fraction including site bias"
            ds["mf_bc_posterior"].attrs["units"] = "mol mol-1"
            ds["mf_bc_posterior"].attrs["long_name"] = "posterior simulated boundary condition mole fraction including site bias"

            # ds["percentile_mf_posterior"].attrs["units"] = "mol mol-1"
            # ds["percentile_mf_posterior"].attrs["long_names"] = "percentile of posterior simulated mole fraction due to state vector uncertainty"
            
            # Auxiliary
            ds["platform"].attrs["long_name"] = "identifier of observing platform; e.g., 3 letter ID for surface in-situ sites plus inlet height above ground: MHD-10"
            # ds["sector"].attrs["long_name"] = "short name of flux sector"
            # ds["percentile"].attrs["units"] = "1"
            # ds["percentile"].attrs["long_name"] = "reported percentiles for non-Gaussian probability distribution functions"

            return ds

        def concentrations():
            """
            Process mandatory variables only
            """
            self.process_data()
            self._get_site_geo_info()
            self._get_time_bounds()

            my_data_vars = {
                "time": (["index"], self.time),
                "time_bnds": (["index", "nbnds"], self.time_bnds),
                "longitude": (["index"], self.longitude),
                "latitude": (["index"], self.latitude),
                "altitude": (["index"], self.altitude),
                "mf_observed": (["index"], self.mf_obs),
                "stdev_mf_observed_repeatability": (["index"], self.mf_obs_repeatability),
                "stdev_mf_observed_variability": (["index"], self.mf_obs_variability),
                "stdev_mf_model": (["index"], self.mf_model_error),
                "stdev_mf_total": (["index"], self.mf_obs_err),
                "mf_prior": (["index"], self.mf_sim.flatten()),
                "mf_posterior": (["index"], self.mf_sim_opt.flatten()),
                "mf_bc_prior": (["index"], self.mf_sim_bc.flatten()),
                "mf_bc_posterior": (["index"], self.mf_sim_opt_bc),
                "percentile_mf_posterior": (["index", "percentile"], self.mf_opt_percentile),
                "number_of_identifier": (["index"], self.site_indicator),
                }
            
            mydims = {"platform": (["platform"], self.sites), 
                      "index": (["index"], np.arange(len(self.time))),
                      "percentile": (["percentile"], np.array([0.16, 0.84])),
                      }

            ds_concentrations = xr.Dataset(data_vars=my_data_vars, coords=mydims)
            ds_conc = concentration_attrs(ds_concentrations)
            ds_out = global_attrs(ds_conc)

            return ds_out
        
        def flux_attrs(ds_fluxes):
            """
            Add attributes to the flux dataset variables according to FLUXIE formatting guidelines
            """
            ds_fluxes["longitude"].attrs["units"] = "degrees_east"
            ds_fluxes["longitude"].attrs["long_name"] = "longitude of grid cell centre"
            ds_fluxes["longitude"].attrs["standard_name"] = "longitude"
            ds_fluxes['longitude'].attrs["axis"] = "X"

            ds_fluxes["latitude"].attrs["units"] = "degrees_north"
            ds_fluxes["latitude"].attrs["long_name"] = "latitude of grid cell centre"
            ds_fluxes["latitude"].attrs["standard_name"] = "latitude"
            ds_fluxes['latitude'].attrs["axis"] = "Y"

            # ds_fluxes["time"].attrs["units"] = "days since 1970-01-01T00:00:00"
            ds_fluxes["time"].attrs["long_name"] = "mid of flux interval in UTC"
            ds_fluxes["time"].attrs["standard_name"] = "time"
            ds_fluxes["time"].attrs["axis"] = "T"

            ds_fluxes["percentile"].attrs["units"] = "1"
            ds_fluxes["percentile"].attrs["long_name"] = "reported percentiles for non-Gaussian probability distribution functions"

            # ds_fluxes["time_bnds"].attrs["units"] = "days since 1970-01-01T00:00:00"
            ds_fluxes["time_bnds"].attrs["long_name"] = "start and end points of each time step"

            # Fluxes
            ds_fluxes["flux_total_prior"].attrs["units"] = "mol m-2 s-1"
            ds_fluxes["flux_total_prior"].attrs["_FillValue"] = float('nan')
            ds_fluxes["flux_total_prior"].attrs["long_name"] = f"prior total surface flux of {self.species}"

            ds_fluxes["flux_total_posterior"].attrs["units"] = "mol m-2 s-1"
            ds_fluxes["flux_total_posterior"].attrs["_FillValue"] = float('nan')
            ds_fluxes["flux_total_posterior"].attrs["long_name"] = f"posterior total surface flux of {self.species}"

            # ds_fluxes["percentile_flux_total_prior"].attrs["units"] = "mol m-2 s-1"
            # ds_fluxes["percentile_flux_total_prior"].attrs["long_name"] = f"percentiles of prior total surface flux of {self.species}."
            # ds_fluxes["percentile_flux_total_prior"].attrs["_FillValue"] = "NaNf"

            ds_fluxes["percentile_flux_total_posterior"].attrs["units"] = "mol m-2 s-1"
            ds_fluxes["percentile_flux_total_posterior"].attrs["long_name"] = f"percentiles of posterior total surface flux of {self.species}."
            ds_fluxes["percentile_flux_total_posterior"].attrs["_FillValue"] = float('nan')

            # Country emisisons 
            ds_fluxes["flux_total_prior_country"].attrs["units"] = "g/year"
            ds_fluxes["flux_total_prior_country"].attrs["_FillValue"] = float('nan')
            ds_fluxes["flux_total_prior_country"].attrs["long_name"] = f"prior total surface flux of {self.species} by country"

            ds_fluxes["flux_total_posterior_country"].attrs["units"] = "g/year"
            ds_fluxes["flux_total_posterior_country"].attrs["_FillValue"] = float('nan')
            ds_fluxes["flux_total_posterior_country"].attrs["long_name"] = f"posterior total surface flux of {self.species} by country"

            # ds_fluxes["percentile_flux_total_prior_country"].attrs["units"] = "g/year"
            # ds_fluxes["percentile_flux_total_prior_country"].attrs["_FillValue"] = "NaNf"
            # ds_fluxes["percentile_flux_total_prior_country"].attrs["long_name"] = f"percentiles of prior total surface flux of {self.species} by country"

            ds_fluxes["percentile_flux_total_posterior_country"].attrs["units"] = "g/year"
            ds_fluxes["percentile_flux_total_posterior_country"].attrs["_FillValue"] = float('nan')
            ds_fluxes["percentile_flux_total_posterior_country"].attrs["long_name"] = f"percentiles of posterior total surface flux of {self.species} by country"

            # Aux variables 
            ds_fluxes["country"].attrs["long_name"] = "country ISO alpha-3 code"

            ds_fluxes["cell_area"].attrs["units"] = "m2"
            ds_fluxes["cell_area"].attrs["long_name"] = "area of grid cell"
            return ds_fluxes


        def fluxes():
            """
            Process fluxes for FLUXIE output
            """
            self.process_data()
            self._get_site_geo_info()

            time = np.array([str(pd.to_datetime(self.start_date) + (pd.to_datetime(self.end_date) - pd.to_datetime(self.start_date)) / 2)])
            time_bnds = np.array([str(pd.to_datetime(self.start_date)), str(pd.to_datetime(self.end_date))]).reshape(1, 2)
            
            ds_country_emissions = self.calculate_country_emissions()

            flux_posterior_percentiles = np.expand_dims(self.flux_post_error, axis=0)
            country_posterior_percentiles = np.expand_dims(np.array([ds_country_emissions['posterior_emissions_lower'].values, ds_country_emissions['posterior_emissions_upper'].values]), axis=0)


            my_data_vars = {
                "longitude": (["longitude"], self.flux_post['longitude'].values),
                "latitude": (["latitude"], self.flux_post['latitude'].values),
                "time": (["time"], time),
                "time_bnds": (["time", "nbnds"], time_bnds),
                "flux_total_prior": (["time", "latitude", "longitude"], self.flux_prior['flux'].values.reshape(1, self.flux_prior['flux'].shape[0], self.flux_prior['flux'].shape[1])),
                "flux_total_posterior": (["time", "latitude", "longitude"], self.flux_post['flux'].values.reshape(1, self.flux_post['flux'].shape[0], self.flux_post['flux'].shape[1])),

                    # "percentile_flux_total_prior": ([("time", "percentile", "latitude", "longitude")], self.xa_percentiles),
                "percentile_flux_total_posterior": (["time", "percentile", "latitude", "longitude"], flux_posterior_percentiles),

                "flux_total_prior_country": (["time", "country"], ds_country_emissions['prior_emissions'].values.reshape(1, ds_country_emissions['country'].shape[0])),
                "flux_total_posterior_country": (["time", "country"], ds_country_emissions['posterior_emissions'].values.reshape(1, ds_country_emissions['country'].shape[0])),

                    # "percentile_flux_total_prior_country": (["time", "percentile", "country"], self.country_prior_emissions_stdev),
                "percentile_flux_total_posterior_country": (["time", "percentile", "country"], country_posterior_percentiles),

                }

            mydims = {"nbnds": (["nbnds"], np.array([0,1])),
                      "country": (["country"], ds_country_emissions['country'].values),
                      "percentile": (["percentile"], np.array([0.16, 0.84])),
                      "cell_area": (["latitude", "longitude"], self.grid_area['area'].values),
                      }

            ds_fluxes = xr.Dataset(data_vars=my_data_vars, coords=mydims)
            ds_flux = flux_attrs(ds_fluxes)
            ds_out = global_attrs(ds_flux)
            return ds_out


        return concentrations(), fluxes()



class PostProcessingMultiTracer:
    """
    A class for post-processing multi-tracer inversion results.
    """
    def __init__(self, 
                 start_date: str,
                 end_date: str,
                 species: str,
                 species2: str,
                 inversion_results: dict,
                 fp_sens_dict_out: dict,
                 fp_sens_dict_out2: dict,
                 flux_grid_prior: xr.Dataset,
                 flux_grid_prior2: xr.Dataset,
                 atmospheric_transport_model: str,
                 inverse_method: str,
                 output_dir: str | None = None,
                 outputname_id: str | None = None,
                 ):
        """
        Initialize the PostProcessingMultiTracer class with the necessary parameters.
        Parameters:
        -----------
        start_date : str
            The start date for the analysis period.
        end_date : str
            The end date for the analysis period.
        species : str
            The first species to be analyzed.
        species2 : str
            The second (tracer) species to be analyzed.
        inversion_results : dict
            A dictionary containing the inversion results.
        fp_sens_dict_out : dict
            A dictionary containing the forward model sensitivities for the first species.
        fp_sens_dict_out2 : dict
            A dictionary containing the forward model sensitivities for the second species.
        flux_grid_prior : xr.Dataset
            The prior flux grid for the first species.
        flux_grid_prior2 : xr.Dataset
            The prior flux grid for the second species.
        atmospheric_transport_model : str
            The atmospheric transport model used.
        inverse_method : str
            The inversion method used.
        output_dir : str
            The directory where output files will be saved.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.species = species
        self.species2 = species2
        self.inversion_results = inversion_results
        self.fp_sens_dict_out = fp_sens_dict_out
        self.fp_sens_dict_out2 = fp_sens_dict_out2
        self.flux_grid_prior = flux_grid_prior
        self.flux_grid_prior2 = flux_grid_prior2
        self.atmospheric_transport_model = atmospheric_transport_model
        self.inverse_method = inverse_method
        self.output_dir = output_dir
        self.outputname_id = outputname_id

    def process_data(self):
        """
        Process the data for the multi-tracer analysis.
        """
        self.time = self.inversion_results['time']['time'].values

        # Observations
        self.mf_g1 = self.inversion_results['Y1'].flatten()
        self.mf_err_g1 = np.diagonal(self.inversion_results['R1'])
        self.mf_g2 = self.inversion_results['Y2'].flatten()
        self.mf_err_g2 = np.diagonal(self.inversion_results['R2'])

        # Prior sectoral fluxes (gas 1) for each basis function
        self.xprior_s1 = self.inversion_results['x1_prior']
        self.xprior_s2 = self.inversion_results['x2_prior']
        self.xpriorBC_g1 = self.inversion_results['x1_bc_prior']
        self.xpriorBC_g2 = self.inversion_results['x2_bc_prior']

        # Posterior mean sectoral fluxes (gas 1) for each basis function
        self.xpost_s1 = self.inversion_results['x1_post']
        self.xpost_s2 = self.inversion_results['x2_post']
        self.xpostBC_g1 = self.inversion_results['x1_bc_post']
        self.xpostBC_g2 = self.inversion_results['x2_bc_post']

        # Sensitivity matrices for each sector
        self.H_s1_g1 = self.inversion_results['H1']
        self.H_s2_g1 = self.inversion_results['H2']
        self.H_s1_g2 = self.inversion_results['G']

        self.nbasis = self.H_s1_g1.shape[1]
        self.ntime = self.H_s1_g1.shape[0]
        
        self.HBC_g1 = self.inversion_results['Hbc']
        self.HBC_g2 = self.inversion_results['Gbc']

        self.nbasisBC = self.HBC_g1.shape[1]

        # Uncertainties in the posterior fluxes
        self.emi_post_cov = self.inversion_results['emi_post_cov']
        self.bc_post_cov = self.inversion_results['bc_post_cov']

        # Emissions ratios
        self.A_alpha = self.inversion_results['A_alpha']
        self.Sa_alpha = self.inversion_results['Sa']

    def compute_molefractions(self):
        """
        Compute mole fractions for both species using the sensitivity matrices and fluxes.
        """
        self.process_data()
        nbasis = self.nbasis
        nbasisBC = self.nbasisBC

        # Gas 1
        mf_g1_obs = self.mf_g1
        mf_g1_obs_err = self.mf_err_g1

        mf_g1_priorBC = self.HBC_g1 @ self.xpriorBC_g1
        mf_g1_prior = (self.H_s1_g1 @ self.xprior_s1) + (self.H_s2_g1 @ self.xprior_s2) + mf_g1_priorBC

        mf_g1_postBC = self.HBC_g1 @ self.xpostBC_g1
        mf_g1_post = (self.H_s1_g1 @ self.xpost_s1) + (self.H_s2_g1 @ self.xpost_s2) + mf_g1_postBC

        emi_s1_std = np.sqrt(np.abs(np.diagonal(self.emi_post_cov[0:nbasis, 0:nbasis]).reshape(-1, 1)))
        emi_s2_std = np.sqrt(np.abs(np.diagonal(self.emi_post_cov[nbasis:, nbasis:]).reshape(-1, 1)))
        g1BC_std = np.sqrt(np.abs(np.diagonal(self.bc_post_cov[0:nbasisBC, 0:nbasisBC]).reshape(-1, 1)))

        mf_g1_postBC_err = self.HBC_g1 @ g1BC_std
        mf_g1_post_err = (self.H_s1_g1 @ emi_s1_std)**2 + (self.H_s2_g1 @ emi_s2_std)**2 + mf_g1_postBC_err **2


        # Gas 2
        mf_g2_obs = self.mf_g2
        mf_g2_obs_err = self.mf_err_g2

        mf_g2_priorBC = self.HBC_g2 @ self.xpriorBC_g2
        mf_g2_prior = (self.H_s1_g2 @ self.A_alpha @ self.xprior_s1) + mf_g2_priorBC

        mf_g2_postBC = self.HBC_g2 @ self.xpostBC_g2
        mf_g2_post = (self.H_s1_g2 @ self.A_alpha @ self.xpost_s1) + mf_g2_postBC

        g2BC_std = np.sqrt(np.diagonal(self.bc_post_cov[nbasisBC:, nbasisBC:]).reshape(-1, 1))
        mf_g2_postBC_err = self.HBC_g2 @ g2BC_std
        mf_g2_post_err = (self.H_s1_g2 @ self.A_alpha @ emi_s1_std)**2 + mf_g2_postBC_err **2

        # GAS 1 XARRAY DATASET
        ds_species1_out = xr.Dataset({
            "time": (["index"], self.time),
            "mf": (["index"], mf_g1_obs),
            "mf_err": (["index"], mf_g1_obs_err),
            "mf_prior": (["index"], mf_g1_prior.flatten()),
            "mf_priorBC": (["index"], mf_g1_priorBC.flatten()),
            "mf_post": (["index"], mf_g1_post.flatten()),
            "mf_post_68": (["index"], np.sqrt(mf_g1_post_err.flatten())),
            "mf_postBC": (["index"], mf_g1_postBC.flatten()),
            "mf_postBC_68": (["index"], np.sqrt(mf_g1_postBC_err.flatten())),
        }, coords={"index": (["index"], np.arange(len(self.time)))})

        ds_species1_out.attrs["title"] = f"ARTEMIS inversion results for {self.species}"
        ds_species1_out.attrs["institution"] = "Massachusetts Institute of Technology (MIT)"
        ds_species1_out.attrs["creator"] = "MIT Atmospheric Inverse Modeling Group"
        ds_species1_out.attrs["creation_date"] = pd.Timestamp.now().isoformat()
        ds_species1_out.attrs["contact"] = "esaboya@mit.edu"
        ds_species1_out.attrs["transport_model"] = self.atmospheric_transport_model
        ds_species1_out.attrs["inversion_system"] = "ARTEMIS"
        ds_species1_out.attrs["inversion_method"] = self.inverse_method
        ds_species1_out.attrs["species"] = self.species

        ds_species1_out["mf"].attrs['units'] = 'mol mol-1'
        ds_species1_out["mf_err"].attrs['units'] = 'mol mol-1'
        ds_species1_out['mf_prior'].attrs['units'] = 'mol mol-1'
        ds_species1_out['mf_priorBC'].attrs['units'] = 'mol mol-1'
        ds_species1_out['mf_post'].attrs['units'] = 'mol mol-1'
        ds_species1_out['mf_post_68'].attrs['units'] = 'mol mol-1'
        ds_species1_out['mf_postBC'].attrs['units'] = 'mol mol-1'
        ds_species1_out['mf_postBC_68'].attrs['units'] = 'mol mol-1'

        ds_species1_out["mf"].attrs['long_name'] = f'Atmospheric measurements of {self.species} mole fraction in dry air'
        ds_species1_out["mf_err"].attrs['long_name'] = f'Atmospheric measurements uncertainty of {self.species} mole fraction in dry air'
        ds_species1_out['mf_prior'].attrs['long_name'] = f'Forward simulated atmospheric mole fractions of {self.species}'
        ds_species1_out['mf_priorBC'].attrs['long_name'] = f'Background simulated atmospheric mole fractions of {self.species}'
        ds_species1_out['mf_post'].attrs['long_name'] = f'Posterior atmospheric mole fractions of {self.species}'
        ds_species1_out['mf_post_68'].attrs['long_name'] = f'Posterior 68% confidence interval of {self.species} mole fraction in dry air'
        ds_species1_out['mf_postBC'].attrs['long_name'] = f'Posterior background atmospheric mole fractions of {self.species}'
        ds_species1_out['mf_postBC_68'].attrs['long_name'] = f'Posterior background 68% confidence interval of {self.species} mole fraction in dry air'

        # GAS 2 XARRAY DATASET
        ds_species2_out = xr.Dataset({
            "time": (["index"], self.time),
            "mf": (["index"], mf_g2_obs.flatten()),
            "mf_err": (["index"], mf_g2_obs_err),
            "mf_priorBC": (["index"], mf_g2_priorBC.flatten()),
            "mf_post": (["index"], mf_g2_post.flatten()),
            "mf_post_68": (["index"], np.sqrt(mf_g2_post_err.flatten())),
            "mf_postBC": (["index"], mf_g2_postBC.flatten()),
            "mf_postBC_68": (["index"], np.sqrt(mf_g2_postBC_err.flatten())),
        }, coords={"index": (["index"], np.arange(len(self.time)))})

        ds_species2_out.attrs["title"] = f"ARTEMIS inversion results for {self.species2}"
        ds_species2_out.attrs["institution"] = "Massachusetts Institute of Technology (MIT)"
        ds_species2_out.attrs["creator"] = "MIT Atmospheric Inverse Modeling Group"
        ds_species2_out.attrs["creation_date"] = pd.Timestamp.now().isoformat()
        ds_species2_out.attrs["contact"] = "esaboya@mit.edu"
        ds_species2_out.attrs["transport_model"] = self.atmospheric_transport_model
        ds_species2_out.attrs["inversion_system"] = "ARTEMIS"
        ds_species2_out.attrs["inversion_method"] = self.inverse_method
        ds_species2_out.attrs["species"] = self.species2

        ds_species2_out["mf"].attrs['units'] = 'mol mol-1'
        ds_species2_out["mf_err"].attrs['units'] = 'mol mol-1'
        ds_species2_out['mf_prior'].attrs['units'] = 'mol mol-1'
        ds_species2_out['mf_priorBC'].attrs['units'] = 'mol mol-1'
        ds_species2_out['mf_post'].attrs['units'] = 'mol mol-1'
        ds_species2_out['mf_post_68'].attrs['units'] = 'mol mol-1'
        ds_species2_out['mf_postBC'].attrs['units'] = 'mol mol-1'
        ds_species2_out['mf_postBC_68'].attrs['units'] = 'mol mol-1'

        ds_species2_out["mf"].attrs['long_name'] = f'Atmospheric measurements of {self.species2} mole fraction in dry air'
        ds_species2_out["mf_err"].attrs['long_name'] = f'Atmospheric measurements uncertainty of {self.species2} mole fraction in dry air'
        ds_species2_out['mf_prior'].attrs['long_name'] = f'Forward simulated atmospheric mole fractions of {self.species2}'
        ds_species2_out['mf_priorBC'].attrs['long_name'] = f'Background simulated atmospheric mole fractions of {self.species2}'
        ds_species2_out['mf_post'].attrs['long_name'] = f'Posterior atmospheric mole fractions of {self.species2}'
        ds_species2_out['mf_post_68'].attrs['long_name'] = f'Posterior 68% confidence interval of {self.species2} mole fraction in dry air'
        ds_species2_out['mf_postBC'].attrs['long_name'] = f'Posterior background atmospheric mole fractions of {self.species2}'
        ds_species2_out['mf_postBC_68'].attrs['long_name'] = f'Posterior background 68% confidence interval of {self.species2} mole fraction in dry air'

        if self.output_dir is not None:
            ds_species1_out.to_netcdf(f"{self.output_dir}/ARTEMIS_molefraction_{self.inverse_method}_{self.species}_{self.start_date}_{self.outputname_id}.nc")
            ds_species2_out.to_netcdf(f"{self.output_dir}/ARTEMIS_molefraction_{self.inverse_method}_{self.species2}_{self.start_date}_{self.outputname_id}.nc")
        return ds_species1_out, ds_species2_out

    def compute_fluxes(self):
        """
        Calculate gridded fluxes from the inversion results.
        """
        from mit_inversions.data.species_molar_masses import molarmasses
        self.process_data()

        # Get molar masses for the species
        species1_mm = molarmasses[self.species]
        species2_mm = molarmasses[self.species2]

        # Get basis function and model domain grid information
        bf_grid = self.fp_sens_dict_out[".basis_function_grid"].values
        nbasis = self.nbasis
        latitude = self.flux_grid_prior['latitude'].values
        longitude = self.flux_grid_prior['longitude'].values
        grid_cell_area = grid_cell_area_m2(latitude, longitude)

        year = pd.to_datetime(self.start_date).year
        seconds_year = seconds_per_year(year)

        # Fluxes in model domain
        # --- Gas 1 ---
        g1_flux_prior = self.flux_grid_prior
        g1_flux_post_sf_bf = np.array([self.xpost_s1/self.xprior_s1, self.xpost_s2/self.xprior_s2])

        g1_flux_post_sf_grid = []
        for i in range(g1_flux_post_sf_bf.shape[0]):
            _flux_post_sf = np.zeros(bf_grid.shape)
            for j in range(nbasis):
                indy, indx = np.where(bf_grid == j)
                for k in range(len(indy)):
                    _flux_post_sf[indy[k], indx[k]] = g1_flux_post_sf_bf[i, j]
            g1_flux_post_sf_grid.append(_flux_post_sf)
        g1_flux_post_sf_grid = np.array(g1_flux_post_sf_grid)
        g1_flux_post_grid = g1_flux_prior['flux'].values * g1_flux_post_sf_grid


        # --- Gas 2 ---
        g2_flux_prior = self.flux_grid_prior2
        g2_flux_post_sf_bf = np.array([self.xpost_s1/self.xprior_s1])

        g2_flux_post_sf_grid = np.zeros(bf_grid.shape)
        for j in range(nbasis):
            indy, indx = np.where(bf_grid == j)
            for k in range(len(indy)):
                g2_flux_post_sf_grid[indy[k], indx[k]] = g2_flux_post_sf_bf[0,j,0]
        g2_flux_post_grid = g2_flux_prior['flux'].values[0] * g2_flux_post_sf_grid


        # Posterior covariance matrices for each sector
        emi_post_g1s1_cov = self.emi_post_cov[0:nbasis,0:nbasis]    # Gas 1, sector 1 covariance basis functions
        emi_post_g1s2_cov = self.emi_post_cov[nbasis:,nbasis:]      # Gas 1, sector 2 covariance basis functions
        emi_post_g1s1s2_cov = self.emi_post_cov[0:nbasis,nbasis:]   # Gas 1, sector 1 and sector 2 covariance basis functions

        var_g1s1 = np.diagonal(emi_post_g1s1_cov)                   # Gas 1, sector 1 posterior variance on each basis function
        var_g1s2 = np.diagonal(emi_post_g1s2_cov)                   # Gas 1, sector 1 posterior variance on each basis function
        cov_g1s1s2 = np.diagonal(emi_post_g1s1s2_cov)               # Gas 1, sector 1 and sector 2 posterior covariance on each basis function
        var_g1_total_bf = var_g1s1 + var_g1s2 + 2*cov_g1s1s2        # Total Gas 1 posterior variance on each basis function
        corr_g1s1s2 = cov_g1s1s2 / np.sqrt(var_g1s1 * var_g1s2)     # Correlation between Gas 1, sector 1 and sector 2 posterior emissions on each basis function

        var_g2s1 = np.diagonal(emi_post_g1s1_cov * np.diagonal(self.A_alpha))    # Gas 2, sector 1 posterior variance on each basis function
        var_g2_total_bf = var_g2s1                                  # Total Gas 2 posterior variance on each basis function


        #   flux uncertainties
        _flux_post_68_s1 = np.zeros(bf_grid.shape)
        _flux_post_68_s2 = np.zeros(bf_grid.shape)
        _flux_post_68_total = np.zeros(bf_grid.shape)
        _flux_post_corr = np.zeros(bf_grid.shape)

        for i in range(nbasis):
            indy, indx = np.where(bf_grid == i)
            for j in range(len(indy)):
                _flux_post_68_s1[indy[j], indx[j]] = np.sqrt(var_g1s1[i]) /len(indy)
                _flux_post_68_s2[indy[j], indx[j]] = np.sqrt(var_g1s2[i]) /len(indy)
                _flux_post_68_total[indy[j], indx[j]] = np.sqrt(var_g1_total_bf[i]) /len(indy)
                _flux_post_corr[indy[j], indx[j]] = corr_g1s1s2[i]

        g1_flux_post_grid_68 = np.array([_flux_post_68_s1, _flux_post_68_s2])
        g1_flux_post_grid_68_total = np.array(_flux_post_68_total)
        g1_flux_post_grid_corr = np.array(_flux_post_corr)

        mycoords = {
            'flux_sector': self.flux_grid_prior['flux_sector'].values,
            'latitude': self.flux_grid_prior['latitude'].values,
            'longitude': self.flux_grid_prior['longitude'].values,
                    }

        flux_out = xr.Dataset({
            'flux_prior': (['flux_sector', 'latitude', 'longitude'], g1_flux_prior['flux'].values),
            'flux_posterior': (['flux_sector', 'latitude', 'longitude'], g1_flux_post_grid),
            'flux_posterior_68': (['flux_sector', 'latitude', 'longitude'], g1_flux_post_grid_68),
            'flux_posterior_68_total': (['latitude', 'longitude'], g1_flux_post_grid_68_total),
            'flux_posterior_corr': (['latitude', 'longitude'], g1_flux_post_grid_corr),
            }, 
            coords=mycoords)
        
        emi_prior_grid = flux_out['flux_prior'] * grid_cell_area['area'] * seconds_year * species1_mm
        emi_post_grid = flux_out['flux_posterior'] * grid_cell_area['area'] * seconds_year * species1_mm
        emi_post_grid_68 = flux_out['flux_posterior_68'] * grid_cell_area['area'] * seconds_year * species1_mm
        emi_post_grid_68_total = flux_out['flux_posterior_68_total'] * grid_cell_area['area'] * seconds_year * species1_mm

        flux_out['emissions_prior'] = emi_prior_grid
        flux_out['emissions_posterior'] = emi_post_grid
        flux_out['emissions_posterior_68'] = emi_post_grid_68
        flux_out['emissions_posterior_68_total'] = emi_post_grid_68_total
        flux_out['cell_area'] = grid_cell_area['area']

        flux_out.attrs['species'] = self.species
        flux_out.attrs['title'] = f"ARTEMIS inversion results for {self.species}"
        flux_out.attrs['institution'] = "Massachusetts Institute of Technology (MIT)"
        flux_out.attrs['creator'] = "MIT Atmospheric Inverse Modeling Group"
        flux_out.attrs['creation_date'] = pd.Timestamp.now().isoformat()
        flux_out.attrs['contact'] = "esaboya@mit.edu"
        flux_out.attrs['transport_model'] = self.atmospheric_transport_model
        flux_out.attrs['inversion_system'] = "ARTEMIS"
        flux_out.attrs['inverse_method'] = self.inverse_method
        flux_out.attrs['Alpha'] = np.diagonal(self.A_alpha)[0]
        flux_out.attrs['Sa'] = np.diagonal(self.Sa_alpha)[0]

        # Unit attributes
        flux_out['flux_prior'].attrs['units'] = 'mol/m2/s'
        flux_out['flux_posterior'].attrs['units'] = 'mol/m2/s'
        flux_out['flux_posterior_68'].attrs['units'] = 'mol/m2/s'
        flux_out['flux_posterior_68_total'].attrs['units'] = 'mol/m2/s'
        flux_out['flux_posterior_corr'].attrs['units'] = 'None'
        flux_out['emissions_prior'].attrs['units'] = 'g/yr'
        flux_out['emissions_posterior'].attrs['units'] = 'g/yr'
        flux_out['emissions_posterior_68'].attrs['units'] = 'g/yr'
        flux_out['emissions_posterior_68_total'].attrs['units'] = 'g/yr'
        flux_out['cell_area'].attrs['units'] = 'm2'

        # Long name
        flux_out['flux_prior'].attrs['long_name'] = 'Prior fluxes for Sector 1 and Sector 2'
        flux_out['flux_posterior'].attrs['long_name'] = 'Posterior fluxes for Sector 1 and Sector 2'
        flux_out['flux_posterior_68'].attrs['long_name'] = 'Posterior fluxes 68% confidence interval for Sector 1 and Sector 2'
        flux_out['flux_posterior_68_total'].attrs['long_name'] = 'Posterior fluxes 68% confidence interval for total fluxes (Sector 1 + Sector 2 + covariance)'
        flux_out['flux_posterior_corr'].attrs['long_name'] = 'Correlation between Sector 1 and Sector 2'
        flux_out['emissions_prior'].attrs['long_name'] = 'Prior emissions for Sector 1 and Sector 2'
        flux_out['emissions_posterior'].attrs['long_name'] = 'Posterior emissions for Sector 1 and Sector 2'
        flux_out['emissions_posterior_68'].attrs['long_name'] = 'Posterior emissions 68% confidence interval for Sector 1 and Sector 2'
        flux_out['emissions_posterior_68_total'].attrs['long_name'] = 'Posterior emissions 68% confidence interval for total emissions (Sector 1 + Sector 2 + covariance)'
        flux_out['cell_area'].attrs['long_name'] = 'Grid cell area'


        # --- Gas 2 ---
        #   flux uncertainties
        _flux_post_68_s1 = np.zeros(bf_grid.shape)
        _flux_post_68_total = np.zeros(bf_grid.shape)

        for i in range(nbasis):
            indy, indx = np.where(bf_grid == i)
            for j in range(len(indy)):
                _flux_post_68_s1[indy[j], indx[j]] = np.sqrt(var_g2s1[i]) / len(indy)
                _flux_post_68_total[indy[j], indx[j]] = np.sqrt(var_g2_total_bf[i]) / len(indy)

        g2_flux_post_grid_68 = np.array([_flux_post_68_s1])
        g2_flux_post_grid_68_total = np.array(_flux_post_68_total) 

        mycoords2 = {
            'flux_sector': np.array([self.flux_grid_prior['flux_sector'].values[0]]),
            'latitude': self.flux_grid_prior['latitude'].values,
            'longitude': self.flux_grid_prior['longitude'].values,
                    }
        
        flux_out_2 = xr.Dataset({
            'flux_prior': (['flux_sector', 'latitude', 'longitude'], g2_flux_prior['flux'].values),
            'flux_posterior': (['flux_sector', 'latitude', 'longitude'], np.reshape(g2_flux_post_grid, (1, g2_flux_post_grid.shape[0], g2_flux_post_grid.shape[1]))),
            'flux_posterior_68': (['flux_sector', 'latitude', 'longitude'], g2_flux_post_grid_68),
            'flux_posterior_68_total': (['latitude', 'longitude'], g2_flux_post_grid_68_total),
            }, 
            coords=mycoords2)
        
        emi2_prior_grid = flux_out_2['flux_prior'] * grid_cell_area['area'] * seconds_year * species2_mm
        emi2_post_grid = flux_out_2['flux_posterior'] * grid_cell_area['area'] * seconds_year * species2_mm
        emi2_post_grid_68 = flux_out_2['flux_posterior_68'] * grid_cell_area['area'] * seconds_year * species2_mm
        emi2_post_grid_68_total = flux_out_2['flux_posterior_68_total'] * grid_cell_area['area'] * seconds_year * species2_mm

        flux_out_2['emissions_prior'] = emi2_prior_grid
        flux_out_2['emissions_posterior'] = emi2_post_grid
        flux_out_2['emissions_posterior_68'] = emi2_post_grid_68
        flux_out_2['emissions_posterior_68_total'] = emi2_post_grid_68_total
        flux_out_2['cell_area'] = grid_cell_area['area']

        flux_out_2.attrs['species'] = self.species2
        flux_out_2.attrs['Alpha'] = np.diagonal(self.A_alpha)[0]
        flux_out_2.attrs['Sa'] = np.diagonal(self.Sa_alpha)[0]

        # Unit attributes
        flux_out_2['flux_prior'].attrs['units'] = 'mol/m2/s'
        flux_out_2['flux_posterior'].attrs['units'] = 'mol/m2/s'
        flux_out_2['flux_posterior_68'].attrs['units'] = 'mol/m2/s'
        flux_out_2['flux_posterior_68_total'].attrs['units'] = 'mol/m2/s'
        flux_out_2['emissions_prior'].attrs['units'] = 'g/yr'
        flux_out_2['emissions_posterior'].attrs['units'] = 'g/yr'
        flux_out_2['emissions_posterior_68'].attrs['units'] = 'g/yr'
        flux_out_2['emissions_posterior_68_total'].attrs['units'] = 'g/yr'
        flux_out_2['cell_area'].attrs['units'] = 'm2'

        # Long name
        flux_out_2['flux_prior'].attrs['long_name'] = 'Prior fluxes for Sector 1'
        flux_out_2['flux_posterior'].attrs['long_name'] = 'Posterior fluxes for Sector 1'
        flux_out_2['flux_posterior_68'].attrs['long_name'] = 'Posterior fluxes 68% confidence interval for Sector 1'
        flux_out_2['flux_posterior_68_total'].attrs['long_name'] = 'Posterior fluxes 68% confidence interval for total fluxes (Sector 1)'

        flux_out_2['emissions_prior'].attrs['long_name'] = 'Prior emissions for Sector 1'
        flux_out_2['emissions_posterior'].attrs['long_name'] = 'Posterior emissions for Sector 1'
        flux_out_2['emissions_posterior_68'].attrs['long_name'] = 'Posterior emissions 68% confidence interval for Sector 1'
        flux_out_2['emissions_posterior_68_total'].attrs['long_name'] = 'Posterior emissions 68% confidence interval for total emissions (Sector 1)'
        flux_out_2['cell_area'].attrs['long_name'] = 'Grid cell area'

        return flux_out, flux_out_2

    def compute_country_emissions(self):
        """
        Calculate country totals for prior and posterior emissions, including uncertainties, for both species.
        """
        g1_flux_out, g2_flux_out = self.compute_fluxes()

        domain_lon = g1_flux_out['longitude'].values
        domain_lat = g1_flux_out['latitude'].values
        cmask = get_countries_for_grid(domain_lon, domain_lat)
        countries_in_domain = list(np.sort(list(set(cmask.data.flatten()))))
        ncountries = len(countries_in_domain)

        # Get the number of basis functions
        bf_grid = self.fp_sens_dict_out['.basis_function_grid'].values
        nbasis = len(np.unique(bf_grid))

        country_names = []

        # GAS 1
        country_prior_emissions_g1s1 = []
        country_prior_emissions_g1s2 = []
        country_post_emissions_g1s1 = []
        country_post_emissions_g1s2 = []

        # Emissions uncertainties w/o covariance
        country_post_emissions_g1s1_68 = []
        country_post_emissions_g1s2_68 = []

        # Emissions uncertainties w/ covariance
        country_post_emissions_g1_68_total = []

        # GAS 2
        country_prior_emissions_g2s1 = []
        country_post_emissions_g2s1 = []
        country_post_emissions_g2s1_68 = []

        name_mapping = {}
        name_mapping['g1_sector1'] = g1_flux_out['flux_sector'].values[0]
        name_mapping['g1_sector2'] = g1_flux_out['flux_sector'].values[1]
        name_mapping['g2_sector1'] = g2_flux_out['flux_sector'].values[0]

        for i, country in enumerate(countries_in_domain):
            country_names.append(country)
            country_mask = (cmask.data == country)
            
            country_prior_emissions_g1s1.append(np.sum(country_mask * g1_flux_out['emissions_prior'].sel(flux_sector=name_mapping['g1_sector1']).values))
            country_prior_emissions_g1s2.append(np.sum(country_mask * g1_flux_out['emissions_prior'].sel(flux_sector=name_mapping['g1_sector2']).values))
            country_post_emissions_g1s1.append(np.nansum(country_mask * g1_flux_out['emissions_posterior'].sel(flux_sector=name_mapping['g1_sector1']).values))
            country_post_emissions_g1s2.append(np.nansum(country_mask * g1_flux_out['emissions_posterior'].sel(flux_sector=name_mapping['g1_sector2']).values))
            country_post_emissions_g1s1_68.append(np.nansum((country_mask * g1_flux_out['emissions_posterior_68'].sel(flux_sector=name_mapping['g1_sector1']).values)))
            country_post_emissions_g1s2_68.append(np.nansum((country_mask * g1_flux_out['emissions_posterior_68'].sel(flux_sector=name_mapping['g1_sector2']).values)))
            country_post_emissions_g1_68_total.append(np.nansum((country_mask * g1_flux_out['emissions_posterior_68_total'].values)))

            country_prior_emissions_g2s1.append(np.sum(country_mask * g2_flux_out['emissions_prior'].sel(flux_sector=name_mapping['g2_sector1']).values))
            country_post_emissions_g2s1.append(np.nansum(country_mask * g2_flux_out['emissions_posterior'].sel(flux_sector=name_mapping['g2_sector1']).values))
            country_post_emissions_g2s1_68.append(np.nansum((country_mask * g2_flux_out['emissions_posterior_68'].sel(flux_sector=name_mapping['g2_sector1']).values)))


        ds_country_emissions_g1 = xr.Dataset({
            'country_names': (['country'], country_names),
            'country_prior_emissions_g1s1': (['country'], country_prior_emissions_g1s1),
            'country_prior_emissions_g1s2': (['country'], country_prior_emissions_g1s2),
            'country_post_emissions_g1s1': (['country'], country_post_emissions_g1s1),
            'country_post_emissions_g1s2': (['country'], country_post_emissions_g1s2),
            'country_post_emissions_g1s1_68': (['country'], country_post_emissions_g1s1_68),
            'country_post_emissions_g1s2_68': (['country'], country_post_emissions_g1s2_68),
            'country_post_emissions_g1_68_total': (['country'], country_post_emissions_g1_68_total),
        })

        ds_country_emissions_g1.attrs['species'] = self.species
        ds_country_emissions_g1.attrs['g1_sector1'] = name_mapping['g1_sector1']
        ds_country_emissions_g1.attrs['g1_sector2'] = name_mapping['g1_sector2']
        ds_country_emissions_g1.attrs['units'] = 'g/yr'
        ds_country_emissions_g1.attrs['long_name'] = f'Country totals for prior and posterior emissions, including uncertainties, for both sectors of {self.species}'
        ds_country_emissions_g1.attrs['start_date'] = self.start_date
        ds_country_emissions_g1.attrs['end_date'] = self.end_date
        
        ds_country_emissions_g2 = xr.Dataset({
            'country_names': (['country'], country_names),
            'country_prior_emissions_g2s1': (['country'], country_prior_emissions_g2s1),
            'country_post_emissions_g2s1': (['country'], country_post_emissions_g2s1),
            'country_post_emissions_g2s1_68': (['country'], country_post_emissions_g2s1_68),
        })

        ds_country_emissions_g2.attrs['species'] = self.species2
        ds_country_emissions_g2.attrs['g2_sector1'] = name_mapping['g2_sector1']
        ds_country_emissions_g2.attrs['units'] = 'g/yr'
        ds_country_emissions_g2.attrs['long_name'] = f'Country totals for prior and posterior emissions, including uncertainties, for {self.species2}'
        ds_country_emissions_g2.attrs['start_date'] = self.start_date
        ds_country_emissions_g2.attrs['end_date'] = self.end_date

        if self.output_dir is not None:
            ds_country_emissions_g1.to_netcdf(f"{self.output_dir}/country_emissions_{self.species}_{self.start_date}_{self.outputname_id}.nc")
            ds_country_emissions_g2.to_netcdf(f"{self.output_dir}/country_emissions_{self.species2}_{self.start_date}_{self.outputname_id}.nc")
            g1_flux_out.to_netcdf(f"{self.output_dir}/ARTEMIS_fluxes_{self.species}_{self.start_date}_{self.outputname_id}.nc")
            g2_flux_out.to_netcdf(f"{self.output_dir}/ARTEMIS_fluxes_{self.species2}_{self.start_date}_{self.outputname_id}.nc")

        return g1_flux_out, g2_flux_out, ds_country_emissions_g1, ds_country_emissions_g2
