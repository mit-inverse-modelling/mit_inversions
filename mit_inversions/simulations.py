# simulations.py
# Created: 17 March 2026
# Author: Eric Saboya
# Copyright (c) 2026. All rights reserved.
# License: MIT License
# 
# Description: 
#  This module implements the forward simulation component of the ARTEMIS framework.

import xarray as xr
import numpy as np
from .readers.observations import Observations
from .readers.footprint_flux_reader import FootprintFlux
from .readers.data_filters import DataFiltering

def data_merge(observations, fp_flux_grid, mf_sim, fps, tolerance="1h") -> dict:
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
        height_dim = fps['height'].values
        
        # Use SAME tolerance and align all variables together
        TOLERANCE = tolerance  # Use provided tolerance for all alignments
        
        # Align mf_sim
        mf_sim_aligned = mf_sim.sel({"site": site}).reindex(
            {"time": observations[site]['time']}, 
            tolerance=TOLERANCE, 
            method="nearest"
        ).dropna("time")
        
        # Align fp_flux_grid with SAME tolerance
        fp_flux_grid_aligned = fp_flux_grid.sel({"site": site}).reindex(
            {"time": observations[site]['time']}, 
            tolerance=TOLERANCE, 
            method="nearest"
        ).dropna("time")
        
        # Align footprint data with SAME tolerance
        p_loc_n_aligned = fps['particle_locations_n'].sel({"site": site}).reindex(
            {"time": observations[site]['time']}, 
            tolerance=TOLERANCE, 
            method="nearest"
        ).dropna("time")
        
        p_loc_s_aligned = fps['particle_locations_s'].sel({"site": site}).reindex(
            {"time": observations[site]['time']}, 
            tolerance=TOLERANCE, 
            method="nearest"
        ).dropna("time")
        
        p_loc_e_aligned = fps['particle_locations_e'].sel({"site": site}).reindex(
            {"time": observations[site]['time']}, 
            tolerance=TOLERANCE, 
            method="nearest"
        ).dropna("time")
        
        p_loc_w_aligned = fps['particle_locations_w'].sel({"site": site}).reindex(
            {"time": observations[site]['time']}, 
            tolerance=TOLERANCE, 
            method="nearest"
        ).dropna("time")

        srr_aligned = fps['srr'].sel({"site": site}).reindex(
            {"time": observations[site]['time']}, 
            tolerance=TOLERANCE, 
            method="nearest"
        ).dropna("time")
        
        # After alignment, find common time indices
        # Get time coordinates of all aligned variables
        times = [
            set(mf_sim_aligned.time.values),
            set(fp_flux_grid_aligned.time.values),
            set(p_loc_n_aligned.time.values),
            set(p_loc_s_aligned.time.values),
            set(p_loc_e_aligned.time.values),
            set(p_loc_w_aligned.time.values),
            set(srr_aligned.time.values),
        ]
        
        # Find intersection of all valid times
        common_times = times[0]
        for t in times[1:]:
            common_times = common_times.intersection(t)
        
        if not common_times:
            raise ValueError(f"No common time points found for site {site} after alignment")
        
        # Reindex all variables to common times
        common_times_sorted = np.sort(list(common_times))
        
        mf_sim_aligned = mf_sim_aligned.sel(time=common_times_sorted)
        fp_flux_grid_aligned = fp_flux_grid_aligned.sel(time=common_times_sorted)
        p_loc_n_aligned = p_loc_n_aligned.sel(time=common_times_sorted)
        p_loc_s_aligned = p_loc_s_aligned.sel(time=common_times_sorted)
        p_loc_e_aligned = p_loc_e_aligned.sel(time=common_times_sorted)
        p_loc_w_aligned = p_loc_w_aligned.sel(time=common_times_sorted)
        srr_aligned = srr_aligned.sel(time=common_times_sorted)
        
        # Subset observations to same times
        obs_subset = observations[site].sel(time=common_times_sorted)
        
        # Verify all have same time dimension
        assert len(mf_sim_aligned.time) == len(fp_flux_grid_aligned.time) == len(srr_aligned.time), \
            f"Time dimension mismatch after alignment for {site}"
        
        data_aligned_dict[site] = xr.Dataset(
            {
                "mf": (("time",), obs_subset["mf"].values),
                "mf_variability": (("time",), obs_subset["mf_variability"].values),
                "mf_repeatability": (("time",), obs_subset["mf_repeatability"].values),
                "mf_sim": (("flux_sector", "time",), mf_sim_aligned.values),
                "fp_flux_grid": (("flux_sector", "time", "latitude", "longitude"), fp_flux_grid_aligned.values),
                "srr": (("time", "latitude", "longitude"), srr_aligned.values),
                "particle_locations_n": (("time", "height", "longitude"), p_loc_n_aligned.values),
                "particle_locations_s": (("time", "height", "longitude"), p_loc_s_aligned.values),
                "particle_locations_e": (("time", "height", "latitude"), p_loc_e_aligned.values),
                "particle_locations_w": (("time", "height", "latitude"), p_loc_w_aligned.values),
            },
            coords={
                "time": common_times_sorted,
                "latitude": lat_dim,
                "longitude": lon_dim,
                "flux_sector": flux_dim,
                "height": height_dim,
            }
        )
    
    return data_aligned_dict


def forward_simulation(data_dict: dict)->dict:
    """
    Run the forward simulation of the ARTEMIS framework using the aligned 
    observations and footprint fluxes data.

    Parameters:
    - data_dict (dict): 
        A dictionary containing all necessary input data for the forward simulation, including:
    """
    footprints_cfg = data_dict.get('footprints', {})
    met_model = footprints_cfg.get('met_model')
    if isinstance(met_model, str):
        if not met_model.strip():
            raise ValueError("data_dict['footprints']['met_model'] must be a non-empty string when provided as a string.")
    elif isinstance(met_model, list):
        if len(met_model) == 0:
            raise ValueError("data_dict['footprints']['met_model'] list cannot be empty.")
        for mm in met_model:
            if not isinstance(mm, str) or not mm.strip():
                raise ValueError("Each entry in data_dict['footprints']['met_model'] must be a non-empty string.")
    else:
        raise ValueError("data_dict['footprints']['met_model'] is required and must be a non-empty string or list of non-empty strings.")

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
     fps,
     ) = FootprintFlux(start_date=data_dict['start_date'],
                       end_date=data_dict['end_date'],
                       sites=data_dict['sites'],
                       site_inlets=data_dict['footprints'].get('site_inlets'),
                       lpdm=data_dict['footprints']['lpdm'],
                       met_model=met_model,
                       species=data_dict['species'],
                       flux=data_dict['flux'],
                       base_data_dir=data_dict['base_data_dir'],
                       ).align_flux_footprint()
    
    # Align forward simulations to observations
    data_aligned_dict = data_merge(observations, fp_flux_grid, mf_sim, fps)

    # DATA FILTERING
    if data_dict['observations']['data_filters'] is not None:
        filter_methods = data_dict['observations']['data_filters']
        if not isinstance(filter_methods, list):
            filter_methods = [filter_methods]
        data_aligned_dict = DataFiltering(dataset=data_aligned_dict, 
                                          filters=filter_methods).run()

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
