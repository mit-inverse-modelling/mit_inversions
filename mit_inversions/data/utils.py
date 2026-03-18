# -*- coding: utf-8 -*-
"""Grid geometry and unit conversion utilities."""
import calendar
import numpy as np
from .grid_constants import TARGET_LAT, TARGET_LON, TARGET_RES_DEG

EARTH_RADIUS_KM = 6371.0
KM2_TO_M2 = 1e6
GG_TO_G = 1e9


def seconds_per_year(year=None):
    """
    Seconds in a given year, accounting for leap years.

    Parameters
    ----------
    year : int, optional
        Calendar year. If None, uses the mean tropical year (365.2422 days).

    Returns
    -------
    float
    """
    if year is None:
        return 365.2422 * 24.0 * 3600.0
    days = 366 if calendar.isleap(year) else 365
    return float(days * 24 * 3600)


def grid_cell_area_km2(lat_deg=None, lon_deg=None, res_deg=None):
    """
    Area of each grid cell in km², returned as a 2D array (lat, lon).

    Parameters
    ----------
    lat_deg : array-like, optional
        1D latitudes. Defaults to TARGET_LAT.
    lon_deg : array-like, optional
        1D longitudes (only its length matters). Defaults to TARGET_LON.
    res_deg : float, optional
        Grid resolution in degrees. Defaults to TARGET_RES_DEG.

    Returns
    -------
    numpy.ndarray
        Shape (n_lat, n_lon), area in km².
    """
    if lat_deg is None:
        lat_deg = TARGET_LAT
    if lon_deg is None:
        lon_deg = TARGET_LON
    if res_deg is None:
        res_deg = TARGET_RES_DEG
    lat_rad = np.deg2rad(np.asarray(lat_deg))
    dlon = np.deg2rad(res_deg)
    dlat = np.deg2rad(res_deg)
    area_1d = (EARTH_RADIUS_KM ** 2) * dlon * np.cos(lat_rad) * dlat
    return np.broadcast_to(area_1d[:, None], (len(lat_deg), len(lon_deg)))


def grid_cell_area_m2(lat_deg=None, lon_deg=None, res_deg=None):
    """
    Area of each grid cell in m², returned as a 2D array (lat, lon).

    Parameters
    ----------
    lat_deg : array-like, optional
        1D latitudes. Defaults to TARGET_LAT.
    lon_deg : array-like, optional
        1D longitudes (only its length matters). Defaults to TARGET_LON.
    res_deg : float, optional
        Grid resolution in degrees. Defaults to TARGET_RES_DEG.

    Returns
    -------
    numpy.ndarray
        Shape (n_lat, n_lon), area in m².
    """
    return grid_cell_area_km2(lat_deg, lon_deg, res_deg) * KM2_TO_M2
