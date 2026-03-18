# -*- coding: utf-8 -*-
"""
Generate emissions distribution from total (Gg) and a spatial proxy.
"""
from pathlib import Path
import numpy as np
import xarray as xr

from ..config import data_path, get_data_path
from ..data.grid_constants import TARGET_LAT, TARGET_LON, TARGET_RES_DEG
from ..data.utils import grid_cell_area_km2, grid_cell_area_m2, seconds_per_year, GG_TO_G


def _load_proxy(path, var_name, lats, lons):
    """
    Load a proxy variable from a global NetCDF and interpolate onto a custom grid.
    """
    with xr.open_dataset(path) as ds:
        lat_dim = "latitude" if "latitude" in ds.dims else "lat"
        lon_dim = "longitude" if "longitude" in ds.dims else "lon"
        if var_name not in ds:
            candidates = list(ds.data_vars)
            if not candidates:
                raise KeyError(f"No data variables in {path}")
            var_name = candidates[0]
        da = ds[var_name].interp(
            {lat_dim: lats, lon_dim: lons},
            method="nearest",
        )
        return np.asarray(da.values, dtype=np.float64)


def generate_emissions_distribution(
    total_Gg,
    method,
    year=None,
    lats=None,
    lons=None,
    res_deg=None,
    nightlights_path=None,
    population_path=None,
    out_path=None,
):
    """
    Generate an emissions field on the given grid that sums to total_Gg.

    Parameters
    ----------
    total_Gg : float
        Total emissions in Gg for the specified region/grid.
    method : str
        One of: "nightlights", "population", "uniform", "uniform_over_land".
    year : int, optional
        Calendar year for seconds-per-year (leap year aware).
        If None, uses mean tropical year (365.2422 days).
    lats : array-like, optional
        1D latitude array (cell centres, degrees_north).
        Defaults to global 0.1° grid (TARGET_LAT).
    lons : array-like, optional
        1D longitude array (cell centres, degrees_east).
        Defaults to global 0.1° grid (TARGET_LON).
    res_deg : float, optional
        Grid resolution in degrees (for area calculation).
        Defaults to TARGET_RES_DEG (0.1°).
    nightlights_path : str or Path, optional
        Path to night lights NetCDF (global 0.1°).
        Default: data_path / "reference/nightlights_0.1deg.nc"
    population_path : str or Path, optional
        Path to population NetCDF (global 0.1°).
        Default: data_path / "reference/population_0.1deg.nc"
    out_path : str or Path, optional
        If set, write the emissions dataset to this NetCDF.

    Returns
    -------
    xarray.Dataset
        With variables:
        - emissions: (latitude, longitude), flux in g/m2/s
        - grid_cell_area_m2: (latitude, longitude), area in m2
    """
    method = method.strip().lower()
    if method not in ("nightlights", "population", "uniform", "uniform_over_land"):
        raise ValueError(
            "method must be one of: nightlights, population, uniform, uniform_over_land"
        )

    if lats is None:
        lats = TARGET_LAT
    if lons is None:
        lons = TARGET_LON
    if res_deg is None:
        res_deg = TARGET_RES_DEG
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)

    lat_da = xr.DataArray(lats, dims=["latitude"], attrs={"units": "degrees_north"})
    lon_da = xr.DataArray(lons, dims=["longitude"], attrs={"units": "degrees_east"})
    area_2d = grid_cell_area_km2(lats, lons, res_deg)
    area_m2 = grid_cell_area_m2(lats, lons, res_deg)

    if method == "uniform":
        weights = area_2d / np.sum(area_2d)
    elif method == "uniform_over_land":
        from ..readers.masks import get_countries_for_grid
        codes = get_countries_for_grid(lons, lats).values
        land = codes != "OCN"
        if not np.any(land):
            raise ValueError("No land cells found; check country mask.")
        land_area = np.where(land, area_2d, 0.0)
        weights = land_area / np.sum(land_area)
    elif method == "nightlights":
        path = Path(nightlights_path or str(get_data_path(data_path / "reference/nightlights_0.1deg.nc")))
        if not path.exists():
            raise FileNotFoundError(
                f"Night lights NetCDF not found: {path}. Run build_nightlights_netcdf first."
            )
        w = _load_proxy(path, "stable_lights", lats, lons)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if float(np.sum(w)) <= 0:
            raise ValueError("Night lights grid has no positive values in region.")
        w_area = w * area_2d
        weights = w_area / np.sum(w_area)
    else:  # population
        path = Path(population_path or str(get_data_path(data_path / "reference/population_0.1deg.nc")))
        if not path.exists():
            raise FileNotFoundError(
                f"Population NetCDF not found: {path}. Run build_population_netcdf first."
            )
        w = _load_proxy(path, "population_density", lats, lons)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if float(np.sum(w)) <= 0:
            raise ValueError("Population grid has no positive values in region.")
        w_area = w * area_2d
        weights = w_area / np.sum(w_area)

    total_g = float(total_Gg) * GG_TO_G
    spy = seconds_per_year(year)
    flux = (total_g * weights / (area_m2 * spy)).astype(np.float32)

    ds = xr.Dataset(
        {
            "flux": (
                ("latitude", "longitude"),
                flux,
                {
                    "long_name": "Emissions flux",
                    "units": "g/m2/s",
                    "total_regional_Gg": float(total_Gg),
                    "distribution_method": method,
                },
            ),
            "grid_cell_area_m2": (
                ("latitude", "longitude"),
                area_m2.astype(np.float32),
                {"long_name": "Grid cell area", "units": "m2"},
            ),
        },
        coords={"latitude": lat_da, "longitude": lon_da},
        attrs={
            "title": "Emissions distribution",
            "total_emissions_Gg": float(total_Gg),
            "distribution_method": method,
            "resolution_deg": res_deg,
        },
    )
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out_path, format="NETCDF4", engine="netcdf4")
    return ds
