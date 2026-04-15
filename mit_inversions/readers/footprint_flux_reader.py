# footprint_flux_reader.py
# Created: 16 March 2026
# Author: Eric Saboya
# Copyright (c) 2026. All rights reserved.
# License: MIT License
# 
# Description: 
#   This module retrieve and processes footprint and flux data for specified sites, date ranges, and
#   flux/lpdm models. It includes the FootprintFlux class, which has methods for validating inputs,
#   loading footprint and flux data, regridding flux data to the footprint grid, and aligning the
#   data for use in inverse modeling and forward simulations. 

import glob
import re
from pathlib import Path
import pint
import pint_xarray
import xarray as xr
import numpy as np

from datetime import datetime
from iris.coords import DimCoord
from iris.cube import Cube
from iris.analysis import AreaWeighted

from ..config import get_data_path
from ..emissions.distribution import generate_emissions_distribution

class FootprintFlux():
    """
    Class for retrieving and processing footprints and flux data for specified 
    sites, date ranges, and flux/lpdm models.
    """
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 sites: list | str | None = None,
                 site_inlets: list | str | None = None,
                 lpdm: str | None = None,
                 met_model: list | str | None = None,
                 species: str | None = None,
                 flux: dict | None = None,
                 base_data_dir: str = "/net/fs01/data/AGAGE",
                 ):
        """
        Initialize the FootprintFlux class with metadata parameters.
        Parameters:
        - start_date (str): 
            The start date for the footprint data (format: 'YYYY-MM-DD').
        - end_date (str): 
            The end date for the footprint data (format: 'YYYY-MM-DD').
        - site (list | str): 
            The name(s) of the site(s) for which footprints are to be 
            retrieved.
        - site_inlet (list | str): 
            The inlet(s) associated with the site(s).
        - lpdm (str): 
            The name of the Lagrangian Particle Dispersion Model used for the
            footprints (e.g., 'STILT').
        - met_model (str): 
            The name of the meteorological model used for the footprints 
            (e.g., 'GFAS').
        - species (str):
            The chemical species for which the flux is being calculated 
            (e.g., 'HFC-23').
        - flux (dict):
            Flux configuration dictionary. Supported modes are
            'auto_generation' and 'customized'.
        - base_data_dir (str):
            Directory to base level of where footprints and flux data are stored. 
            Defaults to /net/fs01/data/AGAGE
        """
        self.start_date = start_date
        self.end_date = end_date
        self.site = sites
        self.site_inlet = site_inlets
        self.lpdm = lpdm
        self.met_model = met_model
        self.species = species
        self.flux = flux

        # Set data paths
        data_path_base = get_data_path(base_data_dir)
        self.base_data_dir = str(data_path_base)
        self.fp_dir = data_path_base / "footprints"

    def _check_common_inputs(self):
        """
        Validate common input parameters shared by footprint and flux workflows.
                This method checks:
                - Date values can be parsed by numpy datetime.
        
        Raises:
        - ValueError: If any of the input parameters are invalid.
        """
        # Check date formats
        try:
            np.datetime64(self.start_date)
            np.datetime64(self.end_date)
        except ValueError:
            raise ValueError("start_date and end_date must be in 'YYYY-MM-DD' format.")
        
    def _check_footprint_inputs(self):
        """
        Validate and normalize footprint-specific inputs.
        """
        if self.lpdm is None or not isinstance(self.lpdm, str) or not self.lpdm.strip():
            raise ValueError("lpdm must be provided as a non-empty string for footprint operations.")

        if self.met_model is None:
            raise ValueError("met_model must be provided for footprint operations.")

        self.lpdm = self.lpdm.strip()

        # Normalize site and site_inlet. site_inlet is optional and defaults to wildcard.
        if isinstance(self.site, str):
            self.site = [self.site]
        elif not isinstance(self.site, list):
            raise ValueError("site must be either a string or list of strings.")

        if self.site_inlet is None:
            self.site_inlet = ["*"] * len(self.site)
        elif isinstance(self.site_inlet, str):
            self.site_inlet = [self.site_inlet] * len(self.site)
        elif isinstance(self.site_inlet, list):
            if len(self.site) != len(self.site_inlet):
                raise ValueError("If site and site_inlet are lists, they must be of the same length.")
        else:
            raise ValueError("site_inlet must be a string, list of strings, or None.")

        # Normalize met_model. A single string applies to all sites.
        if isinstance(self.met_model, str):
            if not self.met_model.strip():
                raise ValueError("met_model must be a non-empty string when provided as a string.")
            self.met_model = [self.met_model] * len(self.site)
        elif isinstance(self.met_model, list):
            if len(self.met_model) != len(self.site):
                raise ValueError("If site and met_model are lists, they must be of the same length.")
        else:
            raise ValueError("met_model must be a string, list of strings, or None.")

        site_norm = []
        inlet_norm = []
        met_norm = []
        for site_val, inlet_val, met_val in zip(self.site, self.site_inlet, self.met_model):
            if not isinstance(site_val, str) or not site_val.strip():
                raise ValueError("Each site must be a non-empty string for footprint operations.")

            if inlet_val is None:
                inlet_clean = "*"
            elif isinstance(inlet_val, str):
                inlet_clean = inlet_val.strip() or "*"
            else:
                raise ValueError("Each site_inlet must be a string or None for footprint operations.")

            if not isinstance(met_val, str) or not met_val.strip():
                raise ValueError("Each met_model must be a non-empty string for footprint operations.")

            site_norm.append(site_val.strip())
            inlet_norm.append(inlet_clean)
            met_norm.append(met_val.strip())

        self.site = site_norm
        self.site_inlet = inlet_norm
        self.met_model = met_norm

    def _check_flux_inputs(self):
        """
        Validate and normalize flux-specific inputs.
        """
        if not self.species:
            raise ValueError("species must be provided for flux operations.")

        if not isinstance(self.flux, dict):
            raise ValueError("flux must be provided as a dictionary.")

        mode = str(self.flux.get("mode", "")).strip().lower()
        if mode not in {"auto_generation", "customized"}:
            raise ValueError("flux['mode'] must be either 'auto_generation' or 'customized'.")

        self.flux["mode"] = mode
        if mode == "auto_generation":
            if "total_emissions_Gg" not in self.flux:
                raise ValueError("flux['total_emissions_Gg'] must be provided for auto_generation.")
            if "method" not in self.flux or not str(self.flux["method"]).strip():
                raise ValueError("flux['method'] must be provided for auto_generation.")

            self.flux["method"] = str(self.flux["method"]).strip().lower()
            region = self.flux.get("region")
            if region is not None:
                if not isinstance(region, dict):
                    raise ValueError("flux['region'] must be a dictionary with lat/lon bounds.")
                required_keys = {"lat_min", "lat_max", "lon_min", "lon_max"}
                missing = required_keys.difference(region.keys())
                if missing:
                    missing_str = ", ".join(sorted(missing))
                    raise ValueError(f"flux['region'] is missing required keys: {missing_str}")
                self.flux["region_portion"] = float(self.flux.get("region_portion", 1.0))
                if not 0.0 <= self.flux["region_portion"] <= 1.0:
                    raise ValueError("flux['region_portion'] must be between 0 and 1.")

                outside_method = self.flux.get("outside_method")
                if self.flux["region_portion"] < 1.0 and not outside_method:
                    raise ValueError("flux['outside_method'] must be provided when region_portion is less than 1.")
                if outside_method is not None:
                    self.flux["outside_method"] = str(outside_method).strip().lower()
            else:
                self.flux["region_portion"] = float(self.flux.get("region_portion", 1.0))
        else:
            path = self.flux.get("path")
            variable = self.flux.get("variable")
            if not path:
                raise ValueError("flux['path'] must be provided for customized mode.")
            if not variable:
                raise ValueError("flux['variable'] must be provided for customized mode.")
            self.flux["path"] = str(path)
            self.flux["variable"] = str(variable)
    
    def _generate_month_range(self)->list:
        """
        Generate a list of year-month strings (YYYYMM format) 
        between start_date and end_date.
        
        Returns:
        - months: 
            A list of strings in 'YYYYMM' format
        """
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        months = []
        current = start.replace(day=1)
        
        while current < end:
            months.append(current.strftime('%Y%m'))
            # Move to the first day of next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return months

    def _file_search_pattern(self, site: str, inlet: str, met_model: str, yyyymm: str) -> str:
        """Construct the file search pattern for footprint files based on site, inlet, and year-month.
        Parameters:
        - site (str): The name of the site (e.g., 'Mace Head').
        - inlet (str): The name of the inlet (e.g., 'inlet1').
        - yyyymm (str): The year and month in 'YYYYMM' format (e.g., '202401').
        Returns:    
        - search_pattern (str): The constructed file search pattern for glob.
        """        
        inlet_pattern = "*" if inlet in (None, "", "*") else f"*{inlet}*"
        search_pattern = (
            f"{self.fp_dir}/{self.lpdm.lower()}/{site.lower()}/"
            f"{inlet_pattern}_{self.lpdm.upper()}*{met_model}*_inert_*{yyyymm}*.nc"
        )
        return search_pattern
    
    def _find_files_for_month(self, 
                              site: str, 
                              inlet: str, 
                              met_model: str,
                              yyyymm: str,
                              )->list:
        """
        Find footprint files for a specific site, inlet, and month.
        
        Parameters:
        - site (str):
            Site name
        - inlet (str): 
            Inlet name
        - yyyymm (str): 
            Year-month in 'YYYYMM' format
        
        Returns:
        - matching_files: 
            List of matching file paths
        """
        # First pass: strict glob pattern (kept for backward compatibility and debug output).
        search_pattern = self._file_search_pattern(site, inlet, met_model, yyyymm)
        strict_matches = glob.glob(search_pattern)
        if strict_matches:
            return sorted(strict_matches)

        # Second pass: robust case-insensitive token matching over all monthly files.
        site_dir = f"{self.fp_dir}/{self.lpdm.lower()}/{site.lower()}"
        monthly_candidates = glob.glob(f"{site_dir}/*{yyyymm}*.nc")
        if not monthly_candidates:
            return []

        lpdm_token = str(self.lpdm).lower()
        met_token = str(met_model).lower()
        inlet_token = str(inlet).strip().lower()

        def _has_required_tokens(path: str) -> bool:
            name = path.rsplit("/", 1)[-1].lower()
            return (
                lpdm_token in name
                and met_token in name
                and "_inert_" in name
                and yyyymm in name
            )

        met_matches = [p for p in monthly_candidates if _has_required_tokens(p)]
        if not met_matches:
            return []

        # If inlet was not explicitly requested, accept all met-matching files.
        if inlet_token in ("", "*", "none"):
            return sorted(met_matches)

        # Prefer files that include the requested inlet token, but don't fail hard if
        # naming does not encode inlet exactly as expected.
        inlet_matches = [p for p in met_matches if inlet_token in p.rsplit("/", 1)[-1].lower()]
        if inlet_matches:
            return sorted(inlet_matches)

        return sorted(met_matches)
    
    def _species_string_format(self, format=None) -> str:
        if format is None:
            return self.species
        elif format == "lower":
            return self.species.lower()
        elif format == "upper":
            return self.species.upper()
        else:
            return re.sub(r'^[a-zA-Z]+', lambda m: m.group().upper(), self.species)

    def _standardize_flux_dataset(self, flux_ds: xr.Dataset, flux_var: str) -> xr.Dataset:
        """Standardize a flux dataset to the internal 2D lat/lon + fluxes format."""
        if flux_var not in flux_ds.data_vars:
            raise ValueError(f"Flux variable {flux_var!r} not found in dataset.")

        flux_da = flux_ds[flux_var].squeeze(drop=True)
        rename_map = {}
        if "latitude" in flux_da.dims:
            rename_map["latitude"] = "lat"
        if "longitude" in flux_da.dims:
            rename_map["longitude"] = "lon"
        if rename_map:
            flux_da = flux_da.rename(rename_map)

        if "lat" not in flux_da.dims or "lon" not in flux_da.dims:
            raise ValueError("Flux variable must have latitude/longitude or lat/lon dimensions.")

        extra_dims = [dim for dim in flux_da.dims if dim not in {"lat", "lon"}]
        if extra_dims:
            raise ValueError(
                f"Flux variable must be two-dimensional after squeezing; found extra dimensions: {extra_dims}"
            )

        flux_da = flux_da.transpose("lat", "lon")
        if "units" not in flux_da.attrs:
            raise ValueError("Flux variable must define a 'units' attribute.")

        return xr.Dataset({"fluxes": flux_da})

    def _grids_match(self, flux_data: xr.Dataset, footprint_data: xr.Dataset) -> bool:
        """Return True when flux and footprint grids already match."""
        return (
            np.array_equal(flux_data["lat"].values, footprint_data["latitude"].values)
            and np.array_equal(flux_data["lon"].values, footprint_data["longitude"].values)
        )

    def _load_auto_generated_flux(self, footprint_data: xr.Dataset) -> xr.Dataset:
        """Generate a flux prior directly on the footprint grid."""
        flux_ds = generate_emissions_distribution(
            total_Gg=self.flux["total_emissions_Gg"],
            method=self.flux["method"],
            year=int(self.start_date[0:4]),
            lats=footprint_data["latitude"].values,
            lons=footprint_data["longitude"].values,
            nightlights_path=self.flux.get("nightlights_path"),
            population_path=self.flux.get("population_path"),
            base_data_dir=self.base_data_dir,
            region=self.flux.get("region"),
            region_portion=self.flux.get("region_portion", 1.0),
            outside_method=self.flux.get("outside_method"),
        )
        return self._standardize_flux_dataset(flux_ds, "flux")

    def _load_customized_flux(self) -> xr.Dataset:
        """Load a customized prior from a user-provided NetCDF file."""
        flux_path = Path(self.flux["path"]).expanduser()
        if not flux_path.exists():
            raise FileNotFoundError(f"Customized flux file not found: {flux_path}")

        with xr.open_dataset(flux_path) as flux_ds:
            flux_loaded = flux_ds.load()
        return self._standardize_flux_dataset(flux_loaded, self.flux["variable"])


    def regrid_flux_to_footprint(self, 
                                 flux_data, 
                                 footprint_data, 
                                 global_grid=False):
        """
        Regrid the flux data onto the spatial grid of the footprint data using area-weighted regridding.
        Parameters:
        - flux_data:
            An xarray Dataset containing the flux data with 'lat' and 'lon' coordinates.
        - footprint_data:
            An xarray Dataset containing the footprint data with 'latitude' and 'longitude' coordinates.
        - global_grid (bool):
            If True, handle latitude bounds at the poles for global grids. Default is False.    
        Returns:
        - regridded_flux:
            A 2D numpy array of the flux data regridded onto the footprint grid.
        - regridded_cube:
            An iris Cube object containing the regridded flux data with appropriate coordinates and metadata.
        """
        # Extract lat/lon from both datasets
        lat_in = flux_data['lat'].values.copy()
        lon_in = flux_data['lon'].values.copy()
        lat_fp = footprint_data['latitude'].values.copy()
        lon_fp = footprint_data['longitude'].values.copy()
        
        # Adjust longitude values if necessary (e.g., from 0-360 to -180 to 180)
        mtohe = lon_in > 180
        lon_in[mtohe] = lon_in[mtohe] - 360
        ordinds = np.argsort(lon_in)
        lon_in = lon_in[ordinds]
        flux_data['fluxes'].values = flux_data['fluxes'].values[:, ordinds]

        # Define the flux cube
        cube_lat_in = DimCoord(lat_in, 
                               standard_name='latitude', 
                               units='degrees')
        
        cube_lon_in = DimCoord(lon_in, 
                               standard_name='longitude', 
                               units='degrees')
        
        cube_flux = Cube(flux_data['fluxes'].values, 
                         dim_coords_and_dims=[(cube_lat_in, 0), (cube_lon_in, 1)])

        cube_flux.coord('latitude').guess_bounds()      
        cube_flux.coord('longitude').guess_bounds()     

        # Define the target footprint grid to regrid onto
        cube_lat_fp = DimCoord(lat_fp, standard_name='latitude', units='degrees')
        cube_lon_fp = DimCoord(lon_fp, standard_name='longitude', units='degrees')
        cube_fp_grid = Cube(np.zeros((len(lat_fp), len(lon_fp))), dim_coords_and_dims=[(cube_lat_fp, 0), (cube_lon_fp, 1)])

        # Handle latitude bounds at poles if global grid
        if global_grid:
            lat_bounds = np.zeros((len(lat_fp), 2))
            lat_bounds[1:,0] = (lat_fp[1:] + lat_fp[:-1]) / 2
            lat_bounds[:-1,1] = (lat_fp[1:] + lat_fp[:-1]) / 2
            lat_bounds[0,0] = lat_fp[0]
            lat_bounds[-1,1] = lat_fp[-1]
            cube_fp_grid.coord('latitude').bounds = lat_bounds[:,:]
        else:
            cube_fp_grid.coord('latitude').guess_bounds()
        cube_fp_grid.coord('longitude').guess_bounds()

        print("Regridding flux data to footprint grid ...")
        cube_regridded = cube_flux.regrid(cube_fp_grid, AreaWeighted(mdtol=1.0))

        return cube_regridded.data, cube_regridded

    def species_molar_mass(self) -> float:
        """
        Return the molar mass of the specified species in grams per mole.
        This method should:
        - Define a dictionary mapping common species names to their molar masses.
        - Look up the molar mass for self.species and return it.
        - Raise an error if the species is not found in the dictionary.
        
        Returns:
        - molar_mass (float): 
            The molar mass of the specified species in g/mol.
        """
        from mit_inversions.data.species_molar_masses import molarmasses
        # Make all keys in molarmasses lowercase for case-insensitive lookup
        molarmasses = {k.lower(): v for k, v in molarmasses.items()}
        if self.species.lower() in molarmasses.keys():
            return molarmasses[self.species.lower()]
        else:
            raise ValueError(f"Molar mass for species {self.species} not found. Please update the molarmasses dictionary.")
        
    def get_footprints(self):
        """
        Retrieve all footprint files for the sites and date range specified
        in the initializer.
        
        Stores data in self.footprints as a nested dictionary: 
        {site: {yyyymm: data}}
        """
        # Check the validity of the input parameters
        self._check_common_inputs()
        self._check_footprint_inputs()
        
        # Generate all months in the date range
        months = self._generate_month_range()
        
        # Construct the file path based on the input parameters
        for i, site in enumerate(self.site):
            inlet = self.site_inlet[i]
            met_model = self.met_model[i]
            
            # Initialize nested dict for this site if not present
            if not hasattr(self, 'footprints'):
                self.footprints = {}
            if site not in self.footprints:
                self.footprints[site] = {}
            
            # Loop through each month in the date range
            site_fps = []

            for yyyymm in months:
                matching_files = self._find_files_for_month(site, inlet, met_model, yyyymm)
                
                if matching_files:
                    # If multiple files match, use the first one
                    fp_file = matching_files[0]
                    if len(matching_files) > 1:
                        print(f"Warning: Multiple files found for {site} ({inlet}) in {yyyymm}. Using {fp_file}")
                    
                    try:
                        fp_data = xr.open_dataset(fp_file)
                        site_fps.append(fp_data)
                        print(f"Successfully loaded footprint data for {site} ({yyyymm}) from {fp_file}")
                    except Exception as e:
                        print(f"Error loading {fp_file}: {e}")
                else:
                    print(f"No footprint file found for {site} ({inlet}) in {yyyymm}")

            if len(site_fps) == 0:
                raise ValueError(
                    f"No footprint files found for site={site}, inlet={inlet}, "
                    f"date range {self.start_date} to {self.end_date}, "
                    f"lpdm={self.lpdm}, met_model={met_model}."
                    f"Search string: {self._file_search_pattern(site, inlet, met_model, yyyymm)}"
                )

            self.footprints[site] = xr.concat(site_fps, dim='time')
        return self.footprints
        
    def get_flux(self, footprint_data: xr.Dataset):
        """
        Retrieve or generate the configured flux prior.
        """
        # Check the validity of the input parameters
        self._check_common_inputs()
        self._check_flux_inputs()

        if self.flux["mode"] == "auto_generation":
            flux_ds = self._load_auto_generated_flux(footprint_data)
            print("Successfully generated flux prior on the footprint grid.")
        else:
            flux_ds = self._load_customized_flux()
            print(f"Successfully loaded customized flux prior from {self.flux['path']}")

        self.fluxes = {"prior": flux_ds}
        return self.fluxes

    def unit_registry(self)-> pint.UnitRegistry:
        """
        Create and return a pint_xarray UnitRegistry for handling units in the 
        footprint and flux data.
        
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
    
    def normalize_cf_units(self, unit_str: str) -> str:
        """
        Normalize CF-style unit strings to a format compatible with pint. 
        This method should:
        - Convert CF-style exponents (e.g., m-2) to pint-compatible format 
          (e.g., m**-2).
        - Handle common unit patterns and ensure they are in a format that pint
          can parse.
        - Return the normalized unit string.
        """
        # Convert CF-style exponents: m-2 → m**-2
        unit_str = unit_str.replace("^", "**")
    
        # Handle patterns like m-2, s-1, kg-1
        import re
        unit_str = re.sub(r'([a-zA-Z]+)(-?\d+)', r'\1**\2', unit_str)

        # Add multiplication symbols
        unit_str = unit_str.replace(" ", " * ")

        return unit_str

    def align_flux_footprint(self):
        """
        Align the flux and footprint data in time and space for use in 
        inverse modeling.
        
        This method should:
        - Ensure that the time dimensions of the flux and footprint data match.
        - Regrid or interpolate the flux data to the spatial grid of the 
          footprints if necessary.
        - Store the aligned data in a new attribute (e.g., self.aligned_data).
        """
        self._check_common_inputs()
        self._check_footprint_inputs()
        self._check_flux_inputs()

        # Load unit registry
        ureg = self.unit_registry()
        pint_xarray.accessors.default_registry = ureg
        
        # Load footprints data and quantify units
        print("Loading footprints data ...")
        fps_dict = self.get_footprints()

        for site in fps_dict.keys():
            fps_outs = fps_dict[site]
            iunit = fps_outs["srr"].attrs["units"]
            for _coord in fps_outs["srr"].coords:
                fps_outs["srr"][_coord].attrs.pop("units", None)

            fps_dict[site]["srr"] = fps_outs["srr"].pint.quantify(ureg.parse_units(iunit))
            fps_dict[site]["particle_locations_n"] = fps_outs["particle_locations_n"]
            fps_dict[site]["particle_locations_s"] = fps_outs["particle_locations_s"]
            fps_dict[site]["particle_locations_e"] = fps_outs["particle_locations_e"]
            fps_dict[site]["particle_locations_w"] = fps_outs["particle_locations_w"]

        # Load flux data and quantify units
        print("Loading flux data ...")
        fp_for_regridding = fps_dict[self.site[0]]  # Use the first site for grid reference

        flux_data = self.get_flux(fp_for_regridding)
        for flux_key in flux_data.keys():
            iunit = flux_data[flux_key]["fluxes"].attrs['units']
            for _coord in flux_data[flux_key]["fluxes"].coords:
                flux_data[flux_key]["fluxes"][_coord].attrs.pop("units", None)
            if "-" in iunit:
                iunit = self.normalize_cf_units(iunit)

            flux_data[flux_key]["fluxes"] = flux_data[flux_key]["fluxes"].pint.quantify(ureg.parse_units(iunit)).pint.to("g / m^2 / s")

        # Regrid flux data to footprint grid
        regridded_fluxes = []
        regridded_fluxes_dim = []

        for flux_key in flux_data.keys():
            if self._grids_match(flux_data[flux_key], fp_for_regridding):
                regridded_flux = flux_data[flux_key]["fluxes"].values.copy()
            else:
                regridded_flux, _ = self.regrid_flux_to_footprint(flux_data[flux_key], fp_for_regridding)

            # Convert from g/m2/s to mol/m2/s
            regridded_flux = regridded_flux / self.species_molar_mass()

            ds_flux = xr.Dataset({"flux": (["latitude", "longitude"], regridded_flux)},
                                 coords={"latitude": fp_for_regridding['latitude'].values,
                                         "longitude": fp_for_regridding['longitude'].values})

            regridded_fluxes.append(ds_flux)
            regridded_fluxes_dim.append(flux_key)
        
        # Concatenate regridded fluxes along a new dimension for flux model
        self.fluxes_regridded = xr.concat(regridded_fluxes, dim="flux_sector").assign_coords(flux_sector=regridded_fluxes_dim)
        
        # Combine fluxes and footprints into a single dataset 
        print("Combining flux and footprint data ...")
        fpXflux = []
        sites_dim = []
        fp_out = []

        for site in fps_dict.keys():
            fp_site = fps_dict[site]['srr']
            fp_out.append(fps_dict[site])

            site_fp_x_flux = []

            for flux_key in flux_data.keys():
                flux_i = self.fluxes_regridded.sel(flux_sector=flux_key)
                site_fp_x_flux.append(fp_site * flux_i['flux'])
            
            site_fp_x_flux_concat = xr.concat(site_fp_x_flux, dim="flux_sector").assign_coords(flux_sector=regridded_fluxes_dim)
            fpXflux.append(site_fp_x_flux_concat)
            sites_dim.append(site)
        
        self.fpXflux = xr.concat(fpXflux, dim="site").assign_coords(site=sites_dim)
        self.mf_sim = self.fpXflux.sum(dim=["latitude", "longitude"])
        self.fps = xr.concat(fp_out, dim="site").assign_coords(site=sites_dim)

        return self.fpXflux, self.mf_sim, self.fluxes_regridded, self.fps
