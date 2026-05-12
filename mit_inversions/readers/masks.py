import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
import json
from pathlib import Path
from ..config import data_path, get_data_path


_REGION_DEFINITIONS_PATH = Path(__file__).with_name("region_definitions.json")


def _mask_path(base_data_dir=None):
    """Resolve the country mask path for the current run."""
    if base_data_dir:
        return get_data_path(base_data_dir) / "masks/countries/world_countries.gpkg"
    return get_data_path(data_path / "masks/countries/world_countries.gpkg")


def _admin1_mask_path(base_data_dir=None):
    """Resolve the admin-1 mask path for subnational region lookups."""
    if base_data_dir:
        return get_data_path(base_data_dir) / "masks/admin1/world_admin1.gpkg"
    return get_data_path(data_path / "masks/admin1/world_admin1.gpkg")


def _load_region_definitions():
    """Load named ADMIN-1 region definitions from readers/region_definitions.json."""
    if not _REGION_DEFINITIONS_PATH.exists():
        raise FileNotFoundError(
            f"Region definitions file not found: {_REGION_DEFINITIONS_PATH}"
        )

    with _REGION_DEFINITIONS_PATH.open("r", encoding="utf-8") as f:
        definitions = json.load(f)

    if not isinstance(definitions, dict):
        raise ValueError("region_definitions.json must contain a JSON object (dictionary).")

    for key, value in definitions.items():
        if not isinstance(key, str):
            raise ValueError("All region definition keys must be strings.")
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(
                f"Region '{key}' must map to a list of ADMIN-1 name strings."
            )

    return definitions


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


def get_admin1_for_grid(lons_1d, lats_1d, base_data_dir=None, country_iso_a3=None):
    """
    Get ADMIN-1 names (e.g., province/state) for each grid cell.

    Expected file: masks/admin1/world_admin1.gpkg
    Required fields in that file: an ADMIN-1 name column and optional ADM0_A3.
    """
    mask_path = _admin1_mask_path(base_data_dir)
    admin1 = gpd.read_file(mask_path)

    name_candidates = ["name", "NAME", "NAME_1", "admin1_name", "province"]
    name_col = next((c for c in name_candidates if c in admin1.columns), None)
    if name_col is None:
        raise KeyError(
            "Could not find an ADMIN-1 name column in admin1 mask. "
            f"Tried: {name_candidates}"
        )

    if country_iso_a3 is not None and "ADM0_A3" in admin1.columns:
        admin1 = admin1.loc[admin1["ADM0_A3"] == country_iso_a3]

    lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
    geometry = [Point(xy) for xy in zip(lon_grid.ravel(), lat_grid.ravel())]
    gdf_points = gpd.GeoDataFrame({"geometry": geometry}, crs="EPSG:4326")
    result = gpd.sjoin(gdf_points, admin1[["geometry", name_col]], how="left", predicate="within")

    names = result[name_col].fillna("NONE").to_numpy().reshape(lon_grid.shape)
    return xr.DataArray(
        names,
        coords={"latitude": lats_1d, "longitude": lons_1d},
        dims=["latitude", "longitude"],
        name="admin1_name",
    )


def get_regions_for_grid(
    region_name,
    lons_1d,
    lats_1d,
    region_definitions=None,
    base_data_dir=None,
    country_iso_a3=None,
):
    """
    Get region names for each grid cell.

    Returns an xarray DataArray of shape (n_lat, n_lon) with string values
    like the region_name for cells within the region, or 'NONE' for cells outside.
    Analogous to get_countries_for_grid.

    Parameters
    ----------
    region_name : str
        Name of region definition key, e.g., "eastern china".
    region_definitions : dict[str, list[str]]
        Mapping like {"eastern china": ["Anhui", "Beijing", ...]}.
        If omitted, definitions are loaded from readers/region_definitions.json.
    """
    if region_definitions is None:
        region_definitions = _load_region_definitions()

    lookup = {k.lower(): v for k, v in region_definitions.items()}
    key = str(region_name).lower().strip()
    if key not in lookup:
        raise KeyError(
            f"Unknown region '{region_name}'. "
            f"Available regions: {sorted(region_definitions.keys())}"
        )

    admin1_grid = get_admin1_for_grid(
        lons_1d,
        lats_1d,
        base_data_dir=base_data_dir,
        country_iso_a3=country_iso_a3,
    )
    region_admin1_names = np.asarray(lookup[key], dtype=object)
    in_region = np.isin(admin1_grid.values, region_admin1_names)
    
    # Return region_name for cells in region, 'NONE' for cells outside
    result = np.where(in_region, region_name, 'NONE')

    return xr.DataArray(
        result,
        coords=admin1_grid.coords,
        dims=admin1_grid.dims,
        name=f"region_{key.replace(' ', '_')}",
    )
