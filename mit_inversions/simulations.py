# simulations.py
# Created: 17 March 2026
# Author: Eric Saboya
# Copyright (c) 2026. All rights reserved.
# License: MIT License
# 
# Description: 
#  This module implements the forward simulation component of the ARTEMIS framework.

import xarray as xr
from .readers.observations import Observations
from .readers.footprint_flux_reader import FootprintFlux
# from .readers.data_filters import DataFiltering

def data_merge(observations, fp_flux_grid, mf_sim)->dict:
    """
    Merge observations and footprint flux data into a single dataset.
    """
    data_aligned_dict = {}
    for site in observations.keys():
        # Dimensions
        lat_dim = fp_flux_grid["latitude"].values
        lon_dim = fp_flux_grid["longitude"].values
        time_dim = observations[site]["time"].values
        flux_dim = mf_sim["flux_sector"].values
        
        # Variables
        mf_sim_aligned = mf_sim.sel({"site": site}).reindex({"time": observations[site]['time']}, tolerance="1H", method="nearest").dropna("time")
        fp_flux_grid_aligned = fp_flux_grid.sel({"site": site}).reindex({"time": observations[site]['time']}, tolerance="0.5H", method="nearest").dropna("time")
        
        if len(mf_sim_aligned["time"]) > len(observations[site]["time"]):
            raise ValueError(f"More aligned time points in mf_sim than observations for site {site}. Check alignment.")

        data_aligned_dict[site] = xr.Dataset(
            {
                "mf": (("time",), observations[site]["mf"].values),
                "mf_variability": (("time",), observations[site]["mf_variability"].values),
                "mf_repeatability": (("time",), observations[site]["mf_repeatability"].values),
                "mf_sim": (("flux_sector", "time",), mf_sim_aligned.values),
                "fp_flux_grid": (("flux_sector", "time", "latitude", "longitude"), fp_flux_grid_aligned.values),
            },
            coords={
                "time": time_dim,
                "latitude": lat_dim,
                "longitude": lon_dim,
                "flux_sector": flux_dim,
            }
        )
    return data_aligned_dict


def forward_simulation(data_dict: dict)->dict:
    """
    Run the forward simulation of the ARTEMIS framework using the aligned 
    observations and footprint fluxes data.
    """
    # Get observations and footprint-flux data
    observations = Observations(species=data_dict['species'],
                                sites=data_dict['sites'],
                                start_date=data_dict['start_date'],
                                end_date=data_dict['end_date'],
                                latest_release=data_dict['observations']['latest_release'],
                                base_data_path=data_dict['base_data_dir'],
                                ).get_observations()
    
    (fp_flux_grid,
     mf_sim,
     flux_grid,
     ) = FootprintFlux(start_date=data_dict['start_date'],
                       end_date=data_dict['end_date'],
                       sites=data_dict['sites'],
                       site_inlets=data_dict['footprints']['site_inlets'],
                       lpdm=data_dict['footprints']['lpdm'],
                       met_model=data_dict['footprints']['met_model'],
                       species=data_dict['species'],
                       flux_model=data_dict['footprints']['flux_model'],
                       flux_model_version=data_dict['footprints']['flux_model_version'],
                       base_data_dir=data_dict['base_data_dir'],
                       ).align_flux_footprint()
    
    # Align forward simulations to observations
    data_aligned_dict = data_merge(observations, fp_flux_grid, mf_sim)

    # DATA FILTERING (optional)

    # Average data into specified bins
    average = data_dict['observations']['average']
    if average is not None:
        average = average.lower()
        data_ave_dict = {}
        for site in data_aligned_dict.keys():
            data_temp = data_aligned_dict[site].resample({"time": average}).pad().dropna("time")

            if len(data_temp["time"]) > len(data_aligned_dict[site]["time"]):
                raise ValueError (f"WARNING! {average} averaging produces more time points that in initial observations. Use a greater averging period!")

            data_ave_dict[site] = data_temp
            
        return data_ave_dict, flux_grid

    else:
        return data_aligned_dict, flux_grid