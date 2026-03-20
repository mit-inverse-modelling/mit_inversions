# observations.py

"""Load AGAGE observation data."""
import pint
import pint_xarray
import xarray as xr
import numpy as np
import pandas as pd
from ..config import data_path, get_data_path

class Observations():
    """
    Class for retrieving and processing atmospheric observations data
    """
    def __init__(self, 
                 species: str, 
                 sites: list | str, 
                 start_date: str,
                 end_date: str,
                 latest_release: bool=False,
                 base_data_path: str=data_path,
                 ):
        """
        Initialize the Observations class with metadata parameters
        and check inputs are as expected.

        Parameters:
        - species (str):
            The chemical species of the desired observations
            (e.g., 'HFC-23').
        - sites: (str | list):
            The name(s) of the site(s) for which footprints are to be 
            retrieved.
        - start_date (str):
            The start date for the footprint data (format: 'YYYY-MM-DD').
        - end_date (str):
            The end date for the footprint data (format: 'YYYY-MM-DD').
        - latest_release (bool):
            Use most recently publicly released data. 
            Defaults to False
        - base_data_oath (str):
            Option to set the AGAGE data path. 
            Defaults to data_path from config.py
        """
        # species 
        if type(species) is str:
            self.species = species
        else:
            raise ValueError(f"{species} should be a str!")

        # sites
        if type(sites) is list:
            self.sites = sites
        elif type(sites) is str:
            self.sites = [sites]
        else:
            raise ValueError(f"{sites} should be either a str or list")

        # start_date and end_date checks
        self.start_date = start_date
        self.end_date = end_date
        try:
            np.datetime64(self.start_date)
            np.datetime64(self.end_date)
   
        except ValueError:
            raise ValueError("start_date and end_date must be in 'YYYY-MM-DD' format.")

        # latest_release 
        self.latest_release = latest_release

        # Set observations data path
        self.data_path = get_data_path(base_data_path)

    def unit_registry(self)-> pint_xarray.UnitRegistry:
        """
        Create and return a pint_xarray UnitRegistry for handling units in the 
        observations data.
        
        This method should:
        - Initialize a UnitRegistry instance.
        - Define any custom units or aliases needed for the specific datasets 
          being used.
        - Return the configured UnitRegistry instance.
        
        Returns:
        - ureg: 
            A pint_xarray.UnitRegistry instance with necessary units defined.
        """
        # Initialize the UnitRegistry
        ureg = pint.UnitRegistry(force_ndarray=True)
        ureg.define('ppm = 1e-6 * mole / mole')
        ureg.define('ppb = 1e-9 * mole / mole')
        ureg.define('ppt = 1e-12 * mole / mole')
        ureg.define('m2 = m * m = meter ** 2')
        return ureg

    def get_netcdf(self)->dict:
        """
        Load AGAGE observations for given species and sites.

        Returns
        -------
        dict
            Dictionary mapping site codes to xarray.Dataset containing observations.
        """
        obs_root_path = self.data_path / "observations" 

        if self.latest_release:
            obs_path = get_data_path(obs_root_path / "agage_released")
        else:
            obs_path = get_data_path(obs_root_path / "agage_unreleased")

        ds_site_dict = {}
        for site in self.sites:
            # Glob for the observation file path
            obs_fn_paths = list((obs_path / self.species.lower()).glob(f"*{site.lower()}*.nc"))

            # If no files found, raise an error
            if not obs_fn_paths:
                raise ValueError(f"No observation files found for site: {site}")
            # If there is more than one, take the one with the latest date
            elif len(obs_fn_paths) > 1:
                obs_fn_path = max(obs_fn_paths, key=lambda p: p.stat().st_mtime)
            else:
                obs_fn_path = obs_fn_paths[0]

            ds_site = xr.open_dataset(obs_fn_path)
            ds_site_dict[site] = ds_site

            self.site_dict = ds_site_dict
        return ds_site_dict

    def slice_obs(self, obs_dict: dict)->dict:
        """
        Slice observation data between start_date and end_date.

        Parameters:
        - obs_dict (dict):
            Dictionary of loaded AGAGE observations keyed by site
        """
        sliced_obs_dict = {}
        
        # we don't want to include the end date, so subtract 1 minute
        end_date = pd.to_datetime(self.end_date) - pd.Timedelta(minutes=1)
        for site, ds in obs_dict.items():
            sliced_obs_dict[site] = ds.sel(time=slice(self.start_date, end_date))
        return sliced_obs_dict
    
    def get_observations(self):
        """
        Get sliced AGAGE observations for given species and sites.

        Returns
        -------
        dict
            Dictionary mapping site codes to xarray.Dataset containing sliced observations.
        """
        # Get data
        obs_dict = self.get_netcdf()

        # Slice to desired time range
        sliced_obs = self.slice_obs(obs_dict)

        sliced_obs_dict = {}

        # Load unit registry
        ureg = self.unit_registry()
        pint_xarray.accessors.default_registry = ureg
 
        # Attach units to the concentration variables in each dataset
        obs_unit = "mol/mol"
        print("Loading observations ...")
        for site, ds in sliced_obs.items():
            sliced_obs_dict[site] = ds.copy()

            for var in ["mf", "mf_repeatability", "mf_repeatability"]:
                try:
                    iunit_sf = float(ds[var].attrs["units"])
                    sliced_obs_dict[site][var] = sliced_obs[site][var].pint.quantify(ureg.parse_units(obs_unit)) * iunit_sf
                except KeyError:
                    print(f"Variable '{var}' not found in dataset for site '{site}'. Skipping unit assignment for this variable.")  

        return sliced_obs_dict