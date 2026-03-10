import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from ..config import data_path, get_data_path

def get_countries_for_grid(lons_1d, lats_1d):
    """Get country for each grid cell."""
    mask_path = get_data_path(data_path / "mask/countries/world_countries.gpkg")
    # Load cached geometries 
    world = gpd.read_file(mask_path)
    # Create grid
    lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
    geometry = [Point(xy) for xy in zip(lon_grid.ravel(), lat_grid.ravel())]
    gdf_points = gpd.GeoDataFrame({'geometry': geometry}, crs='EPSG:4326')
    # Spatial join
    result = gpd.sjoin(gdf_points, world[['geometry', 'NAME']], how='left', predicate='within')

    # Return as 2D array matching grid shape, filling missing values with 'Ocean'
    return result['NAME'].fillna('Ocean').to_numpy().reshape(lon_grid.shape)