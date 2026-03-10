# observations.py
"""Load AGAGE observation data."""
import xarray as xr
import numpy as np
import pandas as pd
from ..config import data_path, get_data_path

class Observations():
    """

    """
    def __init__(self, 
                 species: str, 
                 sites: list | str, 
                 start_date: str,
                 end_date: str,
                 latest_release: bool=False,
                 ):
        """
        Initialize the Observations class with metadata parameters

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
        Use most recently publicly released data. Defaults to False
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
        try:
            np.datetime64(start_date)
            np.datetime64(end_date)
            self.start_date
            self.end_date 
        except ValueError:
            raise ValueError("start_date and end_date must be in 'YYYY-MM-DD' format.")

        # latest_release 
        self.latest_release = latest_release


    def get_netcdf(self)->dict:
        """
        Load AGAGE observations for given species and sites.

        Returns
        -------
        dict
            Dictionary mapping site codes to xarray.Dataset containing observations.
        """
        obs_root_path = data_path / "observations" 

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

        Parameters
        ----------
        species : str
            Chemical species to load (e.g., 'CFC-11', 'CFC-12').
        sites : list of str
            List of site codes to filter the observations. If None, load all sites.
        start_date : str, optional
            Start date for slicing observations (default is "2000-01-01").
        end_date : str, optional
            End date for slicing observations (default is "2020-12-31").
        latest_release : bool, optional
            Whether to load the latest released data. Default is False.

        Returns
        -------
        dict
            Dictionary mapping site codes to xarray.Dataset containing sliced observations.
        """
        obs_dict = self.get_netcdf(self.species, 
                                   self.sites, 
                                   latest_release=self.latest_release,
                                   )
        
        sliced_obs = self.slice_obs(obs_dict, 
                                    start_date=self.start_date, 
                                    end_date=self.end_date,
                                    )
        return sliced_obs