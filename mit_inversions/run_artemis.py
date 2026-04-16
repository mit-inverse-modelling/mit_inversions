# run_orion.py
# Define input data dictionaries and run the ORION framework.

# User-defined Data inputs 

data_dict = {
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "sites": [],
    "species": "cfc-11",
    "base_data_dir": "",
    "model_error_method": "",

    "observations": {
        "latest_release": False,
        "average": "4h",
        },
    
    "footprints": {
        "site_inlets": [],
        "lpdm": "STILT",
        "met_model": "GDAS",
        "flux": {
            "mode": "auto_generation",
            "total_emissions_Gg": 100.0,
            "method": "uniform",
            "region": None,
            "region_portion": 1.0,
            "outside_method": None,
        },
        },

    "basis_functions": {
        "bf_algorithm": "iwasp",
        "model_domain": "EASTASIA",
        "country_masking": True,
        "target_regions": 50,
        "fp_flux_grid_error": None,
        },

    "inversion": {
        "inverse_method": "analytical",
        "emis_scaling_mean": 1,
        "emis_scaling_sigma": 1, 
        "bc_scaling_mean": 1,
        "bc_scaling_sigma": 0.2,
    }
}





# ======== DOCS ========
# start_date: 
#   The start date for the inversion (format: 'YYYY-MM-DD').
# end_date: 
#   The end date for the inversion (format: 'YYYY-MM-DD').
# sites: 
#   A list of site names for which to run the inversion.
# species: 
#   The chemical species for which to run the inversion (e.g., 'HFC-23').
# base_data_dir: 
#   The base directory where input data are stored.
# observations: 
#   A dictionary of parameters for the Observations class.
# footprints:
#   A dictionary of parameters for the FootprintFluxReader class.


##### for DOCS: from Minde #####


# Example of a customized flux input for the footprints section:
# "flux": {
#         "mode": "customized",
#         "path": "/path/to/prior.nc",
#         "variable": "flux",
#     },

# Example for mcmc inversion
