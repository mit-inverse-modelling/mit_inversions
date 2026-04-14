# -*- coding: utf-8 -*-
"""
Generate emissions distribution from total (Gg) and a spatial proxy.
"""
from pathlib import Path
import numpy as np
import xarray as xr

from ..config import data_path, get_data_path
from ..data.grid_constants import TARGET_LAT, TARGET_LON
from ..data.utils import seconds_per_year, GG_TO_G
from ..readers.masks import get_countries_for_grid


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
        target_lons = np.asarray(lons, dtype=np.float64).copy()
        source_lons = np.asarray(ds[lon_dim].values, dtype=np.float64)
        if np.nanmax(source_lons) <= 180.0 and np.any(target_lons > 180.0):
            target_lons[target_lons > 180.0] -= 360.0
        elif np.nanmin(source_lons) >= 0.0 and np.any(target_lons < 0.0):
            target_lons[target_lons < 0.0] += 360.0
        da = ds[var_name].interp(
            {lat_dim: lats, lon_dim: target_lons},
            method="nearest",
        )
        return np.asarray(da.values, dtype=np.float64)


def _infer_cell_bounds(coords, clip=None):
    """Infer cell bounds from 1D coordinate centres."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 1 or coords.size < 2:
        raise ValueError("Coordinate centres must be a 1D array with at least 2 points.")

    mids = 0.5 * (coords[1:] + coords[:-1])
    lower = np.empty_like(coords)
    upper = np.empty_like(coords)
    lower[1:] = mids
    upper[:-1] = mids
    lower[0] = coords[0] - 0.5 * (coords[1] - coords[0])
    upper[-1] = coords[-1] + 0.5 * (coords[-1] - coords[-2])

    bounds = np.column_stack([lower, upper])
    if clip is not None:
        bounds[:, 0] = np.maximum(bounds[:, 0], clip[0])
        bounds[:, 1] = np.minimum(bounds[:, 1], clip[1])
    return bounds


def _grid_cell_area_m2_from_centres(lats, lons):
    """Calculate grid-cell areas from 1D lat/lon centre coordinates."""
    earth_radius_m = 6371000.0
    lat_bounds = np.deg2rad(_infer_cell_bounds(lats, clip=(-90.0, 90.0)))
    lon_bounds = np.deg2rad(_infer_cell_bounds(lons))

    lat_term = np.abs(np.sin(lat_bounds[:, 1]) - np.sin(lat_bounds[:, 0]))
    lon_term = np.abs(lon_bounds[:, 1] - lon_bounds[:, 0])
    return (earth_radius_m ** 2) * lat_term[:, None] * lon_term[None, :]


def _build_region_mask(lats, lons, region):
    """Build a boolean mask for a rectangular region on the target grid."""
    lat_min = region["lat_min"]
    lat_max = region["lat_max"]
    lon_min = region["lon_min"]
    lon_max = region["lon_max"]

    lat_arr = np.asarray(lats, dtype=np.float64)
    lon_arr = np.asarray(lons, dtype=np.float64)

    if lat_min > lat_max or lon_min > lon_max:
        raise ValueError("region bounds must satisfy lat_min <= lat_max and lon_min <= lon_max.")

    if lat_min < float(np.min(lat_arr)) or lat_max > float(np.max(lat_arr)):
        raise ValueError("region latitude bounds extend outside the footprint domain.")
    if lon_min < float(np.min(lon_arr)) or lon_max > float(np.max(lon_arr)):
        raise ValueError("region longitude bounds extend outside the footprint domain.")

    lat_mask = (lat_arr >= lat_min) & (lat_arr <= lat_max)
    lon_mask = (lon_arr >= lon_min) & (lon_arr <= lon_max)
    region_mask = lat_mask[:, None] & lon_mask[None, :]
    if not np.any(region_mask):
        raise ValueError("region does not include any target grid cells.")
    return region_mask


def _default_proxy_path(base_data_dir, relative_path):
    """Resolve a proxy path relative to the current run's base data directory."""
    if base_data_dir:
        return Path(base_data_dir) / relative_path
    return Path(get_data_path(data_path / relative_path))


def _compute_weights(method, lats, lons, area_m2, nightlights_path=None, population_path=None, base_data_dir=None):
    """Compute normalized distribution weights for the requested method."""
    method = method.strip().lower()
    if method not in ("nightlights", "population", "uniform", "uniform_over_land"):
        raise ValueError(
            "method must be one of: nightlights, population, uniform, uniform_over_land"
        )

    area_km2 = area_m2 / 1e6
    if method == "uniform":
        weights = area_km2
    elif method == "uniform_over_land":
        codes = get_countries_for_grid(lons, lats, base_data_dir=base_data_dir).values
        weights = np.where(codes != "OCN", area_km2, 0.0)
    elif method == "nightlights":
        path = Path(nightlights_path) if nightlights_path else _default_proxy_path(base_data_dir, "masks/reference/nightlights_0.1deg.nc")
        if not path.exists():
            raise FileNotFoundError(
                f"Night lights proxy file not found: {path}. Generate proxy file first."
            )
        proxy = _load_proxy(path, "stable_lights", lats, lons)
        weights = np.where(np.isfinite(proxy) & (proxy > 0), proxy, 0.0) * area_km2
    else:
        path = Path(population_path) if population_path else _default_proxy_path(base_data_dir, "masks/reference/population_0.1deg.nc")
        if not path.exists():
            raise FileNotFoundError(
                f"Population proxy file not found: {path}. Generate proxy file first."
            )
        proxy = _load_proxy(path, "population_density", lats, lons)
        weights = np.where(np.isfinite(proxy) & (proxy > 0), proxy, 0.0) * area_km2

    weight_sum = float(np.nansum(weights))
    if weight_sum <= 0:
        raise ValueError(f"{method} produced no positive weights on the selected grid.")
    return weights / weight_sum


def generate_emissions_distribution(
    total_Gg,
    method,
    year=None,
    lats=None,
    lons=None,
    nightlights_path=None,
    population_path=None,
    base_data_dir=None,
    region=None,
    region_portion=1.0,
    outside_method=None,
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
    nightlights_path : str or Path, optional
        Path to night lights NetCDF (global 0.1°).
        Default: data_path / "masks/reference/nightlights_0.1deg.nc"
    population_path : str or Path, optional
        Path to population NetCDF (global 0.1°).
        Default: data_path / "masks/reference/population_0.1deg.nc"
    base_data_dir : str or Path, optional
        Base data directory used to resolve default proxy paths when explicit
        proxy files are not provided.
    region : dict, optional
        Rectangular sub-region with keys lat_min, lat_max, lon_min, lon_max.
        Distribution is done over this region and, optionally, its complement.
    region_portion : float, optional
        Fraction of total emissions assigned to the region. Only used when
        region is provided. Defaults to 1.0.
    outside_method : str, optional
        Distribution method for the region complement when region is provided
        and region_portion < 1.
    out_path : str or Path, optional
        If set, write the emissions dataset to this NetCDF.

    Returns
    -------
    xarray.Dataset
        With variables:
        - emissions: (latitude, longitude), flux in g/m2/s
        - grid_cell_area_m2: (latitude, longitude), area in m2
    """
    if lats is None:
        lats = TARGET_LAT
    if lons is None:
        lons = TARGET_LON
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    if lats.ndim != 1 or lons.ndim != 1:
        raise ValueError("lats and lons must be 1D arrays of target grid centres.")
    if lats.size < 2 or lons.size < 2:
        raise ValueError("lats and lons must each contain at least 2 grid centres.")

    lat_da = xr.DataArray(lats, dims=["latitude"], attrs={"units": "degrees_north"})
    lon_da = xr.DataArray(lons, dims=["longitude"], attrs={"units": "degrees_east"})
    area_m2 = _grid_cell_area_m2_from_centres(lats, lons)

    if region is None:
        weights = _compute_weights(
            method,
            lats,
            lons,
            area_m2,
            nightlights_path=nightlights_path,
            population_path=population_path,
            base_data_dir=base_data_dir,
        )
    else:
        if not isinstance(region, dict):
            raise ValueError("region must be a dictionary with lat/lon bounds.")
        if not 0.0 <= float(region_portion) <= 1.0:
            raise ValueError("region_portion must be between 0 and 1.")

        region_mask = _build_region_mask(lats, lons, region)
        outside_mask = ~region_mask

        if float(region_portion) < 1.0 and not np.any(outside_mask):
            raise ValueError("region_portion is less than 1, but the region covers the entire domain.")

        inside_weights = _compute_weights(
            method,
            lats,
            lons,
            area_m2,
            nightlights_path=nightlights_path,
            population_path=population_path,
            base_data_dir=base_data_dir,
        )
        inside_weights = np.where(region_mask, inside_weights, 0.0)
        inside_sum = float(np.nansum(inside_weights))
        if float(region_portion) > 0.0:
            if inside_sum <= 0:
                raise ValueError("Selected region has no positive weights for the requested method.")
            inside_weights = inside_weights / inside_sum

        outside_weights = np.zeros_like(inside_weights)
        if float(region_portion) < 1.0:
            if outside_method is None:
                raise ValueError("outside_method must be provided when region_portion is less than 1.")
            outside_weights = _compute_weights(
                outside_method,
                lats,
                lons,
                area_m2,
                nightlights_path=nightlights_path,
                population_path=population_path,
                base_data_dir=base_data_dir,
            )
            outside_weights = np.where(outside_mask, outside_weights, 0.0)
            outside_sum = float(np.nansum(outside_weights))
            if outside_sum <= 0:
                raise ValueError("Outside-region cells have no positive weights for the requested outside_method.")
            outside_weights = outside_weights / outside_sum

        weights = float(region_portion) * inside_weights + (1.0 - float(region_portion)) * outside_weights

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
            "outside_distribution_method": outside_method,
            "region_portion": None if region is None else float(region_portion),
        },
    )
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out_path, format="NETCDF4", engine="netcdf4")
    return ds
