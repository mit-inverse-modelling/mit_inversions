"""Load AGAGE observation data."""
import xarray as xr
import numpy as np
import pandas as pd
from ..config import data_path, get_data_path

def get_netcdf(species, sites, latest_release=False):
    """Load AGAGE observations for given species and sites.

    Parameters
    ----------
    species : str
        Chemical species to load (e.g., 'CFC-11', 'CFC-12').
    sites : list of str
        List of site codes to filter the observations. If None, load all sites.
    latest_release : bool, optional
        Whether to load the latest released data. Default is False.

    Returns
    -------
    dict
        Dictionary mapping site codes to xarray.Dataset containing observations.
    """
    obs_root_path = data_path / "observations" 

    if type(sites) is not list:
        sites = list(sites)

    if latest_release:
        obs_path = get_data_path(obs_root_path / "agage_released")
    else:
        obs_path = get_data_path(obs_root_path / "agage_unreleased")

    ds_site_list = {}

    for site in sites:
        # Glob for the observation file path
        obs_fn_paths = list((obs_path / species.lower()).glob(f"*{site.lower()}*.nc"))
        
        # If no files found, raise an error
        if not obs_fn_paths:
            raise ValueError(f"No observation files found for site: {site}")
        # If there is more than one, take the one with the latest date
        elif len(obs_fn_paths) > 1:
            obs_fn_path = max(obs_fn_paths, key=lambda p: p.stat().st_mtime)
        else:
            obs_fn_path = obs_fn_paths[0]

        ds_site = xr.open_dataset(obs_fn_path)

        ds_site_list[site] = ds_site 
    
    return ds_site_list


def slice_obs(obs_dict, start_date="1900-01-01", end_date="2100-01-01"):
    """Slice observation data between start_date and end_date."""
    sliced_obs = {}
    # we don't want to include the end date, so subtract 1 minute
    end_date = pd.to_datetime(end_date) - pd.Timedelta(minutes=1)
    for site, ds in obs_dict.items():
        sliced_obs[site] = ds.sel(time=slice(start_date, end_date))
    return sliced_obs


def get_observations(species, sites, start_date="1900-01-01", end_date="2100-01-01", latest_release=False):
    """Get sliced AGAGE observations for given species and sites.

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
    obs_dict = get_netcdf(species, sites, latest_release=latest_release)
    sliced_obs = slice_obs(obs_dict, start_date=start_date, end_date=end_date)
    return sliced_obs