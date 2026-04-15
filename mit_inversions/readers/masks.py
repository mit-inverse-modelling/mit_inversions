import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
from ..config import data_path, get_data_path


def _mask_path(base_data_dir=None):
    """Resolve the country mask path for the current run."""
    if base_data_dir:
        return get_data_path(base_data_dir) / "masks/countries/world_countries.gpkg"
    return get_data_path(data_path / "masks/countries/world_countries.gpkg")


def get_countries_for_grid(lons_1d, lats_1d, base_data_dir=None):
    """
    Get ISO 3-letter country code (ADM0_A3) for each grid cell.

    Returns an xarray DataArray of shape (n_lat, n_lon) with string values
    like 'CHN', 'USA', 'GBR', or 'OCN' for ocean cells.
    """
    mask_path = _mask_path(base_data_dir)
    world = gpd.read_file(mask_path)
    lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
    geometry = [Point(xy) for xy in zip(lon_grid.ravel(), lat_grid.ravel())]
    gdf_points = gpd.GeoDataFrame({'geometry': geometry}, crs='EPSG:4326')
    result = gpd.sjoin(gdf_points, world[['geometry', 'ADM0_A3']], how='left', predicate='within')

    codes = result['ADM0_A3'].fillna('OCN').to_numpy().reshape(lon_grid.shape)
    return xr.DataArray(
        codes,
        coords={'latitude': lats_1d, 'longitude': lons_1d},
        dims=['latitude', 'longitude'],
        name='country_code'
    )


def get_country_info(base_data_dir=None):
    """
    Load the full country table from world_countries.gpkg.

    Returns a DataFrame with columns: ADM0_A3, NAME, CONTINENT, etc.
    Includes an 'OCN' row for ocean.
    """
    import pandas as pd
    mask_path = _mask_path(base_data_dir)
    world = gpd.read_file(mask_path)
    info = world[['ADM0_A3', 'NAME', 'CONTINENT']].copy()
    info = pd.concat([info, pd.DataFrame([{
        'ADM0_A3': 'OCN', 'NAME': 'Ocean', 'CONTINENT': 'Ocean',
    }])], ignore_index=True)
    return info
