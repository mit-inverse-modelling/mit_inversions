

class DataFiltering():
    """
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
        """
        print("----------------- DATA FILTERING OPTIONS -----------------")
        print("daytime: Selects data between start_hour and end_hour LT")
        print("nighttime: Selects data between 23.00 and 03.00 LT")
        print("noon: Selects data from 12.00 LT only")
        print("daily_median: Calculates daily median value")
        print("----------------------------------------------------------")


    def local_solar_time(self):
        

        return 


    def daytime(self, site:str):
        """
        Method that retains data between start_hour and end_hour 
        """
        hours = self.local_solar_time(self.data[site])
        t_mask = [i for i, h in enumerate(hours) if h>=self.start_hour and h<=self.end_hour]
        
        dataset_temp = self.data[site][dict(time=t_mask)]
        if self.keep_missing:    
            dataset_out = dataset_temp.reindex_like(self.data[site])
        else:
            dataset_out = dataset_temp
        return dataset_out

