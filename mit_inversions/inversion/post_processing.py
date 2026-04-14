
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