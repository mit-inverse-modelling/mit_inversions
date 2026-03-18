# -*- coding: utf-8 -*-
"""
Aggregate emissions flux (g/m2/s) to country and region totals (Gg/yr).

Country codes are ISO 3-letter (ADM0_A3): 'CHN', 'USA', 'GBR', 'OCN' (ocean), etc.
Default region mapping uses CONTINENT from world_countries.gpkg.
Custom region mapping via CSV with columns: code, region.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from ..readers.masks import get_countries_for_grid, get_country_info
from ..data.utils import seconds_per_year, GG_TO_G


def get_country_emissions(emissions_ds, year=None, flux_var="flux", area_var="grid_cell_area_m2"):
    """
    Integrate emissions flux (g/m2/s) over area and time to get Gg/yr by country.

    Parameters
    ----------
    emissions_ds : xarray.Dataset
        Dataset with emissions flux (g/m2/s) and grid cell area (m2).
    year : int, optional
        Calendar year for seconds-per-year (leap year aware).
    flux_var : str
        Name of the flux variable (default "flux").
    area_var : str
        Name of the area variable (default "grid_cell_area_m2").

    Returns
    -------
    pandas.DataFrame
        Columns: code, name, emissions_Gg, share_of_global.
        code is ADM0_A3 (e.g. 'CHN', 'USA', 'OCN').
    """
    lat = np.asarray(emissions_ds["latitude"].values)
    lon = np.asarray(emissions_ds["longitude"].values)
    codes = get_countries_for_grid(lon, lat).values
    flux = np.asarray(emissions_ds[flux_var].values)
    area = np.asarray(emissions_ds[area_var].values)
    spy = seconds_per_year(year)
    em_Gg = flux * area * spy / GG_TO_G

    info = get_country_info().set_index('ADM0_A3')
    total_global = float(np.nansum(em_Gg))
    unique_codes = np.unique(codes)
    rows = []
    for code in unique_codes:
        code_str = str(code).strip()
        if code_str == "":
            continue
        mask = codes == code
        name = info.loc[code_str, 'NAME'] if code_str in info.index else code_str
        rows.append({
            "code": code_str,
            "name": name,
            "emissions_Gg": float(np.nansum(em_Gg[mask])),
        })
    df = pd.DataFrame(rows)
    if total_global > 0 and "emissions_Gg" in df.columns:
        df["share_of_global"] = df["emissions_Gg"] / total_global
    return df.sort_values("emissions_Gg", ascending=False).reset_index(drop=True)


def get_region_emissions(emissions_ds, year=None, flux_var="flux", area_var="grid_cell_area_m2", csv_path=None):
    """
    Integrate emissions flux (g/m2/s) to Gg/yr by region.

    Without csv_path, uses CONTINENT from world_countries.gpkg as region.
    With csv_path, loads a CSV with columns (code, region) where code is ADM0_A3.
    See emissions/data/country_region_sample.csv for the expected format.

    Parameters
    ----------
    emissions_ds : xarray.Dataset
        Dataset with emissions flux (g/m2/s) and grid cell area (m2).
    year : int, optional
        Calendar year for seconds-per-year (leap year aware).
    flux_var : str
        Name of the flux variable (default "flux").
    area_var : str
        Name of the area variable (default "grid_cell_area_m2").
    csv_path : str or Path, optional
        Path to a CSV with columns (code, region).
        If None, uses CONTINENT from gpkg.

    Returns
    -------
    pandas.DataFrame
        Columns: region, emissions_Gg, share_of_global.
    """
    country_df = get_country_emissions(emissions_ds, year=year, flux_var=flux_var, area_var=area_var)

    if csv_path is not None:
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(f"Region CSV not found: {p}")
        mapping = pd.read_csv(p)
        if "code" not in mapping.columns or "region" not in mapping.columns:
            raise ValueError(f"CSV must have 'code' and 'region' columns. Found: {list(mapping.columns)}")
        code_to_region = dict(zip(mapping["code"].astype(str).str.strip(), mapping["region"].astype(str).str.strip()))
        country_df["region"] = country_df["code"].map(lambda c: code_to_region.get(c, "Other"))
    else:
        info = get_country_info().set_index('ADM0_A3')
        country_df["region"] = country_df["code"].map(lambda c: info.loc[c, 'CONTINENT'] if c in info.index else "Other")

    region_df = (
        country_df.groupby("region", as_index=False)["emissions_Gg"]
        .sum()
        .sort_values("emissions_Gg", ascending=False)
    )
    total = region_df["emissions_Gg"].sum()
    if total > 0:
        region_df["share_of_global"] = region_df["emissions_Gg"] / total
    return region_df.reset_index(drop=True)
