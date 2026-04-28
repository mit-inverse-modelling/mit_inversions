# data_filters.py
# Created: 17 March 2026
# Author: Eric Saboya
# Copyright (c) 2026. All rights reserved.
# License: MIT License

import json
import numpy as np
import pandas as pd

class DataFiltering():
    """
    Class used for filtering observations data
    """

    def __init__(self, 
                 dataset: dict,
                 filters: str | list | None = None,
                 keep_missing: bool = False,
                 start_hour: int=11,
                 end_hour: int=15,
                 ):
        """
        Initiliaze DataFiltering class with metadata parameters.

        use .print_filters() method to print list of accepted filters.

        Parameters
        - dataset (dict):
            Dictionary of observations data with site names as keys.
        - filters (str | list | None):
            Data filters to apply to observations data. 
            Defaults to None. 
            Use .print_filters() option to list all available data filtering
            approaches.
        - keep_missing (bool):
            Option to reindex to retain missing data.
        - start_hour (int):
            Start hour to filter data from.
            Defaults to 11
        - end_hour (int):
            End hour to filter data too (inclusive).
            Defaults to 15
        """
        # Observations data dict
        self.data = dataset

        # Data filtering
        if type(filters) is str:
            filters = list(filters)
        self.filters = filters

        # Keep missing data
        self.keep_missing = keep_missing
        
        # Start and end hour values to filter between
        self.start_hour = start_hour
        self.end_hour = end_hour

    def print_filters(self):
        """
        Prints different data filtering options
        """
        # methods_list = [method for method in dir(DataFiltering) if callable(getattr(DataFiltering, method)) and not method.startswith("__")]

        print("----------------- DATA FILTERING OPTIONS -----------------")
        print("daytime: Selects data between start_hour and end_hour LT")
        print("nighttime: Selects data before start_hour and after end_hour")
        print("specific: Selects data from specific hour (start_hour) only")
        print("daily_median: Calculates daily median value")
        print("local_ratio: Filters data based on local to domain-wide footprint influence")
        print("----------------------------------------------------------")

    def local_solar_time(self, site:str):
        """
        Returns hour of day as a function of local solar time
        relative to the Greenwich Meridian.

        Parameters
        - site (str):
            Atmospheric measurement site ID
        """
        # Load site data json file 
        with open("../data/site_data.json", "r") as f:
            site_data = json.load(f)
        
        site_lon = site_data[site][list(site_data[site].keys())[0]].get("longitude")

        if site_lon>180:
            site_lon -= 360
        
        dataset_copy = self.data[site].copy()
        dataset_copy["time"] += pd.Timedelta(minutes=float(24 * 60 * site_lon / 360.0))
        hours = dataset_copy["time"].to_pandas().index.hour
        
        return hours

    def daily_median(self, site:str):
        """
        Method that calculates the daily median.
        """
        if self.keep_missing:
            return self.data[site].resample(indexer={"time": "1D"}).median()
        else:
            return self.data[site].resample(indexer={"time": "1D"}).median().dropna(dim="time")

    def timerange(self, site:str, filter_method:str):
        """
        Method that filters time-stamped data values 
        based on retaining data between certain hours 
        or at a specified time.

        Parameters:
        - site (str):
            Atmospheric site ID
        - filter_method (str):
            One of daytime, nighttime, specific.
            
            > daytime: selects data between start_hour and end_hour
                       AND range below
            > nighttime: selects data before start_hour and after end_hour
                       OR range below
            > specific: selects data at a specific time determined by start_hour

        ----------------------------------------
        |/ .. OR ..\ | -- AND -- |/ .. OR .. \|
        |  1 | 2 | 3 | ... | ... | 22 | 23 | 0| 
                        ^
                   |-SPECIFIC-|
        ----------------------------------------
        """
        hours = self.local_solar_time(site)
        
        if filter_method == "daytime":
            t_mask = [i for i, h in enumerate(hours) if h>=self.start_hour and h<=self.end_hour]
        
        elif filter_method == "nighttime":
            t_mask = [i for i, h in enumerate(hours) if h>=self.start_hour or h<=self.end_hour]
        
        elif filter_method == "specific":
            t_mask = [i for i, h in enumerate(hours) if h==self.start_hour]

        dataset_temp = self.data[site][dict(time=t_mask)]
        if self.keep_missing:    
            dataset_out = dataset_temp.reindex_like(self.data[site])
        else:
            dataset_out = dataset_temp
        return dataset_out        
    
    def local_ratio(self, site:str):
        """
        Method that filters data based on the ratio of local to domain-wide footprint influence.
        """
        # Load site data json file 
        with open("mit_inversions/data/site_data.json", "r") as f:
            site_data = json.load(f)
        
        site_lon = site_data[site][list(site_data[site].keys())[0]].get("longitude")
        site_lat = site_data[site][list(site_data[site].keys())[0]].get("latitude")

        srr_lat = self.data[site]['latitude'].values
        srr_lon = self.data[site]['longitude'].values

        dx = np.abs(srr_lon[10] - srr_lon[9])
        dy = np.abs(srr_lat[10] - srr_lat[9])

        x_ind = np.where((srr_lon >= site_lon - dx/2) & (srr_lon <= site_lon + dx/2))[0]
        y_ind = np.where((srr_lat >= site_lat - dy/2) & (srr_lat <= site_lat + dy/2))[0]
        
        local_ratio = np.zeros(len(self.data[site].time))

        for t in range(len(local_ratio)):
            local_sum = np.nansum(self.data[site]['srr'][t, y_ind, x_ind])
            domain_sum = np.nansum(self.data[site]['srr'][t, :, :])
            local_ratio[t] = local_sum / domain_sum if domain_sum > 0 else np.nan
        
        mask = local_ratio <= 0.10
        dataset_temp = self.data[site][dict(time=mask)]
        if self.keep_missing:    
            dataset_out = dataset_temp.reindex_like(self.data[site])
        else:            
            dataset_out = dataset_temp
        return dataset_out
    
    def run(self):
        """
        Method to run the specified data filters on the dataset.
        """
        filtered_data = {}

        for site in self.data.keys():
            for filter_method in self.filters:
                if filter_method in ["daytime", "nighttime", "specific"]:
                    site_data = self.timerange(site, filter_method)

                elif filter_method == "daily_median":
                    site_data = self.daily_median(site)

                elif filter_method == "local_ratio":
                    site_data = self.local_ratio(site)

                else:
                    raise ValueError(f"Unknown filter method: {filter_method}")
                
            filtered_data[site] = site_data
        return filtered_data