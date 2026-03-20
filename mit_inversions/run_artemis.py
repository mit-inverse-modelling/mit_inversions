# run_orion.py
# Define input data dictionaries and run the ORION framework.

# User-defined Data inputs 

data_dict = {
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "sites": [],
    "species": "C",
    "base_data_dir": "",
    "model_error_method": "",
    "observations": {
        "latest_release": False,
        "average": "4H",
        },
    
    "footprints": {
        "site_inlets": [],
        "lpdm": "STILT",
        "met_model": "GDAS",
        "flux_model": "EDGAR",
        "flux_model_version": "v8",
        },
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
