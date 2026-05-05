# -*- coding: utf-8 -*-
"""Grid geometry and unit conversion utilities."""

import calendar
import numpy as np
import xarray as xr
# from .grid_constants import TARGET_LAT, TARGET_LON, TARGET_RES_DEG

EARTH_RADIUS_KM = 6371.0
KM2_TO_M2 = 1e6
GG_TO_G = 1e9

def seconds_per_year(year=None)->float:
    """
    Seconds in a given year, accounting for leap years.

    Parameters:
    year (int, optional):
        Calendar year. If None, uses the mean tropical year (365.2422 days).
    
    Returns:
    float:
        Number of seconds in the specified year.
    """
    if year is None:
        return 365.2422 * 24.0 * 3600.0
    days = 366 if calendar.isleap(year) else 365
    return float(days * 24 * 3600)

def grid_cell_area_km2(lat_deg, lon_deg, res_deg=None)->xr.Dataset:
    """
    Geographical area of each grid cell in km2, returned as a 2D xr.Dataset.

    Parameters:
    lat_deg (array-like):
        1D latitudes.
    lon_deg (array-like):
        1D longitudes (only its length matters).
    res_deg (float, optional):
        Grid resolution in degrees. Defaults to TARGET_RES_DEG.

    Returns:
    xr.Dataset: 
        Geographical area in km2.
    """    
    if res_deg is None:
        # res_deg = TARGET_RES_DEG
        dy = np.diff(lat_deg).mean().astype(np.float32)
        dx = np.diff(lon_deg).mean().astype(np.float32)
    else:
        dy = res_deg
        dx = res_deg

    lat_rad = np.deg2rad(np.asarray(lat_deg))
    dlon = np.deg2rad(dx)
    dlat = np.deg2rad(dy)
    area_1d = (EARTH_RADIUS_KM ** 2) * dlon * np.cos(lat_rad) * dlat
    area = np.broadcast_to(area_1d[:, None], (len(lat_deg), len(lon_deg)))

    ds = xr.Dataset(
        {"area": (["latitude", "longitude"], area)},
        
        coords={"latitude": (["latitude"], lat_deg), "longitude": (["longitude"], lon_deg)},
    )
    return ds

def grid_cell_area_m2(lat_deg, lon_deg, res_deg=None)->xr.Dataset:
    """
    Geographical area of each grid cell in m2, returned as a 2D xr.Dataset.

    Parameters
    ----------
    lat_deg : array-like
        1D latitudes.
    lon_deg : array-like
        1D longitudes (only its length matters).
    res_deg : float, optional
        Grid resolution in degrees. Defaults to TARGET_RES_DEG.

    Returns
    -------
    xr.Dataset
        area in m2.
    """
    ds = grid_cell_area_km2(lat_deg, lon_deg, res_deg)
    ds['area'].values = KM2_TO_M2 * ds['area'].values
    return ds