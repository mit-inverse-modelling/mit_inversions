
import xarray as xr 
import numpy as np
import re
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from mit_inversions.data.utils import grid_cell_area_m2
from mit_inversions.readers.masks import get_countries_for_grid, get_country_info


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

    def __init__(self, 
                 species: str,
                 start_date: str,
                 inversion_results: dict,
                 fp_sens_dict_out: dict,
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
        inversion_results (dict):
            A dictionary containing the inversion results.
        fp_sens_dict_out (dict):
            A dictionary containing the forward model sensitivity information.
        output_dir (str):
            The directory where output files will be saved.
        """
        self.species = species
        self.start_date = start_date
        self.inversion_results = inversion_results
        self.output_dir = output_dir
        self.fp_sens_dict_out = fp_sens_dict_out

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
        # self.xa = np.reshape(self.inversion_results['xa'], (-1, 1))
        self.xa  = self.inversion_results['xa']
        self.xa_error = self.inversion_results['xa_error']
        self.xhat = self.inversion_results['xhat'].flatten()
        self.shat = self.inversion_results['shat']

        # simulated mole fractions - stacked by site 
        self.mf_sim = self.H.data @ self.xa

        self.mf_sim_opt = self.H.data @ self.xhat
        self.mf_sim_opt_err = self.H.data @ np.sqrt(np.diagonal(self.shat)) - self.mf_sim_opt

        self.sites = self.inversion_results['sites']
        self.site_indicator = self.inversion_results['site_indicator'][0]
        
        bc_indicator = np.diag(self.inversion_results['bc_data_indicator'])
        self.mf_sim_opt_bc = self.H @ bc_indicator @ self.xhat

    def _get_site_geo_info(self):
        """
        Method to retrieve the longitude, latitude, altitude 
        for each site
        """
        # Read site information json file 
        import json 
        # import mit_inversions.mit_inversions.data.site_data.json as site_data
        with open("/home/esaboya/mit_inversions/mit_inversions/data/site_data.json") as f:
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

    def fluxie(self):
        """
        Process inversion results in concentration and flux formats that align 
        with FLUXIE formatting
        """
        
        def concentration_attrs(ds_concentrations):
            """
            Add attributes to the concentration dataset variables according to FLUXIE formatting guidelines
            """
            ds_concentrations["longitude"].attrs["long_name"] = "sample_longitude_in_decimal_degrees"
            ds_concentrations["longitude"].attrs["units"] = "degrees_east"
            ds_concentrations["longitude"].attrs["comment"] = "Longitude at which air sample was collected."
            ds_concentrations["longitude"].attrs["standard_name"] = "longitude"

            ds_concentrations["latitude"].attrs["long_name"] = "sample_latitude_in_decimal_degrees"
            ds_concentrations["latitude"].attrs["units"] = "degrees_north"
            ds_concentrations["latitude"].attrs["comment"] = "Latitude at which air sample was collected."
            ds_concentrations["latitude"].attrs["standard_name"] = "latitude"

            ds_concentrations["altitude"].attrs["long_name"] = "sample_altitude_in_decimal_degrees"
            ds_concentrations["altitude"].attrs["units"] = "m"
            ds_concentrations["altitude"].attrs["comment"] = "Altitude (surface elevation plus sample intake height) at which air sample was collected."
            ds_concentrations["altitude"].attrs["standard_name"] = "altitude"

            ds_concentrations["number_of_identifier"].attrs["long_name"] = "Index of identifier of observing platform"
            ds_concentrations["number_of_identifier"].attrs["units"] = "1"

            # Observations
            ds_concentrations["mf_observed"].attrs["units"] = "mol mol-1"
            ds_concentrations["mf_observed"].attrs["long_name"] = "observed mole fraction of co2 in dry air"

            ds_concentrations["stdev_mf_observed_repeatability"].attrs["units"] = "mol mol-1"
            ds_concentrations["stdev_mf_observed_repeatability"].attrs["long_name"] = "repeatability uncertainty of observed mole fraction"
            ds_concentrations["stdev_mf_observed_repeatability"].attrs["comment"] = "understood as combined analytical uncertainty"

            ds_concentrations["stdev_mf_observed_variability"].attrs["units"] = "mol mol-1"
            ds_concentrations["stdev_mf_observed_variability"].attrs["long_name"] = "variability uncertainty of observed mole fraction"

            ds_concentrations["stdev_mf_model"].attrs["units"] = "mol mol-1"
            ds_concentrations["stdev_mf_model"].attrs["long_name"] = "model uncertainty of simulated mole fraction"

            ds_concentrations["stdev_mf_total"].attrs["units"] = "mol mol-1"
            ds_concentrations["stdev_mf_total"].attrs["long_name"] = "total model-data-mismatch uncertainty applied in inversion"
            
            # Simulations
            ds_concentrations["mf_prior"].attrs["units"] = "mol mol-1"
            ds_concentrations["mf_prior"].attrs["long_name"] = "prior simulated mole fraction of co2 in dry air"
            ds_concentrations["mf_posterior"].attrs["units"] = "mol mol-1"
            ds_concentrations["mf_posterior"].attrs["long_name"] = "posterior simulated mole fraction of co2 in dry air"
            ds_concentrations["mf_bc_prior"].attrs["units"] = "mol mol-1"
            ds_concentrations["mf_bc_prior"].attrs["long_name"] = "prior simulated boundary condition mole fraction including site bias"
            ds_concentrations["mf_bc_posterior"].attrs["units"] = "mol mol-1"
            ds_concentrations["mf_bc_posterior"].attrs["long_name"] = "posterior simulated boundary condition mole fraction including site bias"

            # ds_concentrations["percentile_mf_posterior"].attrs["units"] = "mol mol-1"
            # ds_concentrations["percentile_mf_posterior"].attrs["long_names"] = "percentile of posterior simulated mole fraction due to state vector uncertainty"
            
            # Auxiliary
            ds_concentrations["platform"].attrs["long_name"] = "identifier of observing platform; e.g., 3 letter ID for surface in-situ sites plus inlet height above ground: MHD-10"
            # ds_concentrations["sector"].attrs["long_name"] = "short name of flux sector"
            # ds_concentrations["percentile"].attrs["units"] = "1"
            # ds_concentrations["percentile"].attrs["long_name"] = "reported percentiles for non-Gaussian probability distribution functions"

            return ds_concentrations

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
                "mf_bc_posterior": (["index"], self.mf_sim_opt_bc.flatten()),
                "stdev_mf_posterior": (["index"], self.mf_sim_opt_err),
                }
            
            mydims = {"platform": (["platform"], self.sites), "index": (["index"], np.arange(len(self.time)))}

            ds_concentrations = xr.Dataset(data_vars=my_data_vars, coords=mydims)
            # ds_conc = concentration_attrs(ds_concentrations)
            return ds_concentrations
        
        return concentrations()
