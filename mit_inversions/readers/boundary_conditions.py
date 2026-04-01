import re
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from ..config import data_path, get_data_path

class BoundaryConditions():
    """
    Class to store the boundary conditions for a given case.
    Also produces a sensitivity matrix for the boundary conditions.
    """
    def __init__(self, species, fp_obj):
        self.bc_path = get_data_path(data_path / "/home/lwestern/agage/12box_model/")
        self.species = species
        self.fp_obj = fp_obj

    def _species_string_format(self) -> str:
        if "br" in self.species:
            return self.species.upper().replace("BR", "Br")
        elif "cl" in self.species:
            return self.species.upper().replace("CL", "Cl")
        else:
            return re.sub(r'^[a-zA-Z]+', lambda m: m.group().upper(), self.species)

    @staticmethod
    def _latitudinal_scale(latitudes: xr.DataArray, box0: float, box1: float, box2: float, box3: float) -> xr.DataArray:
        """Return latitude-dependent scale factors for 4 semi-hemispheric bands."""
        return xr.where(
            latitudes <= -30,
            box3,
            xr.where(
                (latitudes > -30) & (latitudes <= 0),
                box2,
                xr.where((latitudes > 0) & (latitudes <= 30), box1, box0),
            ),
        )

    @staticmethod
    def _band_value(latitude_value: float, box0: float, box1: float, box2: float, box3: float) -> float:
        """Return a single box value for a given boundary latitude."""
        if latitude_value <= -30:
            return box3
        if latitude_value <= 0:
            return box2
        if latitude_value <= 30:
            return box1
        return box0

    def get_boundary_conditions(self):
        bc_dict = {}

        var_name = "Semihemispheric_modelled_mole_fractions"
        species_in = self._species_string_format()
        csv_path = Path(self.bc_path) / species_in / "outputs" / f"{species_in}_{var_name}.csv"
        csv = pd.read_csv(
            csv_path,
            comment="#",
        )

        if isinstance(self.fp_obj, dict):
            site_iter = self.fp_obj.items()
        else:
            site_iter = ((str(i), site_ds) for i, site_ds in enumerate(self.fp_obj))

        for key, fp_obs_site in site_iter:
            lats = fp_obs_site.latitude
            minlat = float(lats.min())
            maxlat = float(lats.max())

            years = np.unique(fp_obs_site.time.dt.year.values.astype(int))
            monthly_bcs = []

            for year in years:
                yind = np.floor(csv["Year"].values).astype(int) == int(year)
                year_rows = csv.loc[yind]

                if len(year_rows) < 12:
                    raise ValueError(f"Expected 12 monthly rows for {year}, got {len(year_rows)}")

                for month_idx in range(12):
                    month_num = month_idx + 1
                    month_mask = (
                        (fp_obs_site.time.dt.year == year)
                        & (fp_obs_site.time.dt.month == month_num)
                    )
                    fp_obs_month = fp_obs_site.sel(time=month_mask)

                    if fp_obs_month.sizes.get("time", 0) == 0:
                        continue

                    box0 = float(year_rows[f"{var_name}_box0"].iloc[month_idx])*1e-12
                    box1 = float(year_rows[f"{var_name}_box1"].iloc[month_idx])*1e-12
                    box2 = float(year_rows[f"{var_name}_box2"].iloc[month_idx])*1e-12
                    box3 = float(year_rows[f"{var_name}_box3"].iloc[month_idx])*1e-12
                    south_value = self._band_value(minlat, box0, box1, box2, box3)
                    north_value = self._band_value(maxlat, box0, box1, box2, box3)

                    lat_scale = self._latitudinal_scale(lats, box0, box1, box2, box3)

                    bc_s = fp_obs_month.particle_locations_n * south_value
                    bc_n = fp_obs_month.particle_locations_s * north_value
                    bc_e = fp_obs_month.particle_locations_e * lat_scale
                    bc_w = fp_obs_month.particle_locations_w * lat_scale

                    bcds = xr.Dataset(
                        data_vars={
                            "bc_s": bc_s,
                            "bc_n": bc_n,
                            "bc_e": bc_e,
                            "bc_w": bc_w,
                        }
                    )
                    monthly_bcs.append(bcds)

            if not monthly_bcs:
                raise ValueError(f"No boundary-condition time slices generated for site {key}")

            site_ds = xr.concat(monthly_bcs, dim="time").sortby("time")
            site_ds.attrs.update(
                {
                    "source": "AGAGE 12-box model",
                    "species": self.species,
                    "title": f"{self.species} mixing ratio at domain edges",
                    "units": "ppt",
                }
            )

            bc_dict[key] = site_ds

        return bc_dict
    
    def get_sensitivity_matrix(self, frequency="monthly"):
        """
        Build a boundary-condition sensitivity matrix.

        Output shape is (n_time, n_periods * 4), where each period contributes
        4 columns corresponding to boundary edges [S, N, E, W].

        Values are edge-summed sensitivities at each time step, assigned to
        the column block of that row's period.
        """
        bc_sensitivity_dict = {}
        bc_dict = self.get_boundary_conditions()

        for key, bc_ds in bc_dict.items():
            if frequency == "monthly":
                period_labels = pd.DatetimeIndex(bc_ds.time.values).to_period("M").astype(str)
            elif frequency == "annual":
                period_labels = pd.DatetimeIndex(bc_ds.time.values).to_period("Y").astype(str)
            else:
                raise ValueError(f"Unsupported frequency: {frequency}")

            unique_periods = pd.Index(period_labels).unique().tolist()
            n_time = bc_ds.sizes["time"]
            n_periods = len(unique_periods)
            n_cols = n_periods * 4
            bc_sensitivity = np.zeros((n_time, n_cols), dtype=float)

            period_to_col = {period: idx * 4 for idx, period in enumerate(unique_periods)}
            edge_vars = (("S", "bc_s"), ("N", "bc_n"), ("E", "bc_e"), ("W", "bc_w"))

            for i in range(n_time):
                period = period_labels[i]
                col0 = period_to_col[period]

                for edge_offset, (_, var_name) in enumerate(edge_vars):
                    edge_data = np.asarray(bc_ds[var_name].isel(time=i).values, dtype=float)
                    bc_sensitivity[i, col0 + edge_offset] = np.nansum(edge_data)

            period_edge_labels = []
            for period in unique_periods:
                for edge_name, _ in edge_vars:
                    period_edge_labels.append(f"{period}_{edge_name}")

            bc_sensitivity_dict[key] = xr.DataArray(
                bc_sensitivity,
                coords={
                    "time": bc_ds.time.values,
                    "period_edge": period_edge_labels,
                },
                dims=["time", "period_edge"],
                name=f"{self.species}_boundary_condition_sensitivity",
            )

        return bc_sensitivity_dict
        