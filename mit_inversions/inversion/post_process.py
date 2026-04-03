import json
import warnings
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import xarray as xr

try:
    from . import mcmc_builder as _mcmc_builder_module
except ImportError:  # pragma: no cover - fallback for direct execution
    from geoschem.inversion import mcmc_builder as _mcmc_builder_module

"""Generic post-processing utilities.

This module only keeps format-agnostic building blocks:
- save/load inversion inputs and raw outputs
- convert raw analytical or MCMC outputs into basic state-level results
- compute annual totals from already prepared state results

Experiment-specific interpretation should stay out of this file. Examples:
- expanding subset states back to the full state vector
Those belong in post_process_specific.py.
"""

__all__ = [
    "build_mcmc_metadata",
    "build_analytical_metadata",
    "save_mcmc_outputs",
    "save_analytical_outputs",
    "compute_annual_totals",
    "postprocess_analytical_outputs",
    "postprocess_mcmc_outputs",
    "plot_state_outputs",
    "plot_analytical_diagnostics",
]




# Public API: metadata builders
def build_mcmc_metadata(x_prior, R_prior, *, n_samples, n_tune, n_chains,
                        cores=None, nuts_sampler="pymc", target_accept=0.9,
                        random_seed=None, obs_index=None, state_index=None,
                        dropped_rows=None, dropped_cols=None,
                        obs_group=None, obs_group_name="group",
                        obs_site_year_group=None, x_var_name="x_prior",
                        other_specific_data=None, extra=None):
    metadata = {
        "x_var_name": x_var_name,
        "x_prior": _infer_prior_spec(x_prior, fallback_type="custom_callable"),
        "R_prior": _infer_prior_spec(R_prior, fallback_type="fixed_variance"),
        "nuts_sampler": nuts_sampler,
        "n_chains": int(n_chains),
        "n_samples": int(n_samples),
        "n_tune": int(n_tune),
        "target_accept": float(target_accept),
        "cores": int(cores if cores is not None else n_chains),
        "random_seed": random_seed,
        "obs_index": _index_to_strings(obs_index) if obs_index is not None else [],
        "state_index": _index_to_strings(state_index) if state_index is not None else [],
    }
    packed_specific = _pack_other_specific_data(
        other_specific_data,
        dropped_rows=dropped_rows,
        dropped_cols=dropped_cols,
        obs_group=obs_group,
        obs_group_name=obs_group_name,
        obs_site_year_group=obs_site_year_group,
    )
    if packed_specific is not None:
        metadata["other_specific_data"] = packed_specific
    if extra:
        metadata.update(_jsonify(extra))
    return metadata


def build_analytical_metadata(*, obs_index=None, state_index=None,
                              dropped_rows=None, dropped_cols=None,
                              other_specific_data=None, extra=None):
    metadata = {
        "obs_index": _index_to_strings(obs_index) if obs_index is not None else [],
        "state_index": _index_to_strings(state_index) if state_index is not None else [],
    }
    packed_specific = _pack_other_specific_data(
        other_specific_data,
        dropped_rows=dropped_rows,
        dropped_cols=dropped_cols,
    )
    if packed_specific is not None:
        metadata["other_specific_data"] = packed_specific
    if extra:
        metadata.update(_jsonify(extra))
    return metadata


# Public API: raw output writers
def save_mcmc_outputs(output_path, idata, H_used, obs_used,
                      metadata=None, R_used=None, P_used=None, xa=None, prior_add_on=None):
    if metadata is None:
        raise ValueError("metadata is required")
    paths = _ensure_output_dirs(output_path)
    file_map = _save_input_tables(
        paths["inputs"],
        H_used,
        obs_used,
        R_used=R_used,
        P_used=P_used,
        xa=xa,
        prior_add_on=prior_add_on,
    )

    results_path = paths["results"] / "mcmc_results.nc"
    idata.to_netcdf(results_path)

    metadata = dict(metadata or {})
    metadata["result_type"] = "mcmc"
    metadata["files"] = {
        "inputs": file_map,
        "results": {"mcmc_results": str(results_path.name)},
    }

    metadata_path = paths["results"] / "mcmc_run_metadata.json"
    _write_metadata(metadata_path, metadata)
    return paths


def save_analytical_outputs(output_path, xhat, ak, shat, H_used, obs_used,
                            metadata=None, R_used=None, P_used=None, xa=None, prior_add_on=None):
    if metadata is None:
        raise ValueError("metadata is required")
    paths = _ensure_output_dirs(output_path)
    file_map = _save_input_tables(
        paths["inputs"],
        H_used,
        obs_used,
        R_used=R_used,
        P_used=P_used,
        xa=xa,
        prior_add_on=prior_add_on,
    )

    xa_df = _coerce_dataframe(xa, value_name="xa")
    state_labels = _index_to_strings(xa_df.index)

    ds = xr.Dataset(
        data_vars={
            "xhat": (("state",), np.asarray(xhat, dtype=float).reshape(-1)),
            "ak": (("state", "state_out"), np.asarray(ak, dtype=float)),
            "shat": (("state", "state_out"), np.asarray(shat, dtype=float)),
        },
        coords={
            "state": state_labels,
            "state_out": state_labels,
        },
    )

    results_path = paths["results"] / "analytical_results.nc"
    ds.to_netcdf(results_path)

    metadata = dict(metadata or {})
    metadata["result_type"] = "analytical"
    metadata["files"] = {
        "inputs": file_map,
        "results": {"analytical_results": str(results_path.name)},
    }

    metadata_path = paths["results"] / "analytical_run_metadata.json"
    _write_metadata(metadata_path, metadata)
    return paths


# Public API: generic downstream aggregation
def compute_annual_totals(results, *, prior_cov_full=None, posterior_cov_full=None):
    """Aggregate prepared state results into annual totals and annual uncertainties.

    This function expects state-level prior/posterior values to already be in their
    final interpretation. In other words, if a workflow needs to expand subset states
    or add prior means back to increments, that should happen before calling this
    function.
    """
    results = results.copy()
    results.index = results.index.map(str)
    results["year"] = results.index.map(lambda idx: idx.rsplit("_", 1)[0])

    annual = pd.DataFrame(index=sorted(results["year"].unique()))
    annual["prior_total"] = results.groupby("year")["prior"].sum()
    annual["posterior_total"] = results.groupby("year")["posterior"].sum()
    annual["posterior_sigma_if_states_independent"] = results.groupby("year")["state_posterior_sigma"].apply(
        lambda s: np.sqrt(np.sum(np.square(np.nan_to_num(s.astype(float).values, nan=0.0))))
    )

    prior_cov_df = _coerce_square_dataframe(prior_cov_full, results.index, "prior_cov_full")
    posterior_cov_df = _coerce_square_dataframe(posterior_cov_full, results.index, "posterior_cov_full")

    if prior_cov_df is None:
        annual["prior_sigma_if_states_independent"] = np.nan
        annual["prior_sigma_for_total"] = np.nan
    if posterior_cov_df is None:
        annual["posterior_sigma_for_total"] = annual["posterior_sigma_if_states_independent"]

    for year in annual.index:
        year_index = results.index[results["year"] == year]
        if prior_cov_df is not None:
            p_sub = np.nan_to_num(
                prior_cov_df.loc[year_index, year_index].values.astype(float),
                nan=0.0,
            )
            annual.loc[year, "prior_sigma_if_states_independent"] = np.sqrt(np.trace(p_sub))
            annual.loc[year, "prior_sigma_for_total"] = np.sqrt(np.sum(p_sub))
        if posterior_cov_df is not None:
            s_sub = np.nan_to_num(
                posterior_cov_df.loc[year_index, year_index].values.astype(float),
                nan=0.0,
            )
            annual.loc[year, "posterior_sigma_for_total"] = np.sqrt(np.sum(s_sub))

    if {"prior_sigma_if_states_independent", "posterior_sigma_if_states_independent"}.issubset(annual.columns):
        prior_var = annual["prior_sigma_if_states_independent"] ** 2
        posterior_var = annual["posterior_sigma_if_states_independent"] ** 2
        annual["error_reduction_if_states_independent"] = np.where(
            prior_var > 0,
            1.0 - posterior_var / prior_var,
            np.nan,
        )

    if {"prior_sigma_for_total", "posterior_sigma_for_total"}.issubset(annual.columns):
        prior_var = annual["prior_sigma_for_total"] ** 2
        posterior_var = annual["posterior_sigma_for_total"] ** 2
        annual["error_reduction_for_total"] = np.where(
            prior_var > 0,
            1.0 - posterior_var / prior_var,
            np.nan,
        )

    return annual


def plot_state_outputs(output_path, results, annual):
    """Make presentation-style plots from already prepared state and annual results."""
    paths = _ensure_output_dirs(output_path)

    results_with_year = results.copy()
    results_with_year["year"] = results_with_year.index.map(lambda idx: idx.rsplit("_", 1)[0])
    for year in sorted(results_with_year["year"].unique()):
        year_df = results_with_year.loc[
            results_with_year["year"] == year,
            ["prior", "posterior", "delta_x", "state_posterior_sigma"],
        ]

        plt.figure(figsize=(10, 5))
        labels = year_df.index.tolist()
        plt.errorbar(labels, year_df["posterior"], yerr=year_df["state_posterior_sigma"], fmt="o", label="Posterior")
        plt.plot(labels, year_df["prior"], "s--", label="Prior")
        plt.title(f"{year} State Posterior Summary")
        plt.ylabel("Emission")
        plt.xticks(rotation=45, ha="right")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths["yearly_diag"] / f"{year}_state_summary.png", dpi=200, bbox_inches="tight")
        plt.close()

    annual_plot = annual.copy()
    annual_plot.index = annual_plot.index.astype(int)
    plt.figure(figsize=(10, 6))
    plt.plot(annual_plot.index, annual_plot["prior_total"], marker="o", label="Prior total", color="tab:blue")
    plt.fill_between(
        annual_plot.index,
        annual_plot["prior_total"] - annual_plot["prior_sigma_for_total"],
        annual_plot["prior_total"] + annual_plot["prior_sigma_for_total"],
        color="tab:blue",
        alpha=0.18,
        label="Prior total uncertainty",
    )
    plt.plot(annual_plot.index, annual_plot["posterior_total"], marker="o", label="Posterior total", color="tab:orange")
    plt.fill_between(
        annual_plot.index,
        annual_plot["posterior_total"] - annual_plot["posterior_sigma_for_total"],
        annual_plot["posterior_total"] + annual_plot["posterior_sigma_for_total"],
        color="tab:orange",
        alpha=0.18,
        label="Posterior total uncertainty",
    )
    plt.xlabel("Year")
    plt.ylabel("Annual total emission")
    plt.title("Annual Prior and Posterior Total Emissions")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths["figures"] / "annual_prior_posterior_timeseries.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_analytical_diagnostics(output_path, results, posterior_cov_full, ak_full):
    """Plot analytical diagnostics from the provided state results and matrices."""
    paths = _ensure_output_dirs(output_path)
    results_with_year = results.copy()
    results_with_year["year"] = results_with_year.index.map(lambda idx: idx.rsplit("_", 1)[0])

    posterior_cov_df = _coerce_square_dataframe(posterior_cov_full, results.index, "posterior_cov_full")
    ak_df = _coerce_square_dataframe(ak_full, results.index, "ak_full") if ak_full is not None else None

    diag_vals = np.clip(np.diag(posterior_cov_df.values.astype(float)), a_min=0.0, a_max=None)
    denom = np.sqrt(np.outer(diag_vals, diag_vals))
    corr = np.divide(
        posterior_cov_df.values.astype(float),
        denom,
        out=np.zeros_like(posterior_cov_df.values.astype(float)),
        where=denom > 0,
    )
    corr_df = pd.DataFrame(corr, index=posterior_cov_df.index, columns=posterior_cov_df.columns)

    for year in sorted(results_with_year["year"].unique()):
        labels = results_with_year.index[results_with_year["year"] == year]
        cov_sub = posterior_cov_df.loc[labels, labels]
        corr_sub = corr_df.loc[labels, labels]
        if ak_df is not None:
            ak_sub = ak_df.loc[labels, labels]
        else:
            ak_sub = pd.DataFrame(0.0, index=labels, columns=labels)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.heatmap(ak_sub, cmap="coolwarm", center=0, ax=axes[0], xticklabels=labels, yticklabels=labels)
        axes[0].set_title(f"{year} AK")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].tick_params(axis="y", rotation=0)

        sns.heatmap(cov_sub, cmap="coolwarm", center=0, ax=axes[1], xticklabels=labels, yticklabels=labels)
        axes[1].set_title(f"{year} Posterior Covariance")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].tick_params(axis="y", rotation=0)

        sns.heatmap(
            corr_sub, cmap="coolwarm", center=0, vmin=-1, vmax=1,
            ax=axes[2], xticklabels=labels, yticklabels=labels
        )
        axes[2].set_title(f"{year} Posterior Correlation")
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].tick_params(axis="y", rotation=0)

        fig.suptitle(f"{year} Inversion Diagnostics", fontsize=14)
        fig.tight_layout()
        fig.savefig(paths["yearly_diag"] / f"{year}_diagnostics.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


# Public API: raw output readers and generic post-processing
def postprocess_analytical_outputs(output_path):
    """Load saved analytical outputs and build the basic raw state-level products.

    The job of this function is intentionally narrow: it reconstructs the direct
    analytical inversion outputs (`xhat`, `ak`, `shat`) into a consistent raw table
    and saves raw matrices. It does not do experiment-specific interpretation such as
    subset expansion, extra plotting, or final annual summaries.
    """
    paths = _ensure_output_dirs(output_path)
    metadata = _read_metadata(paths["results"] / "analytical_run_metadata.json")
    input_files = metadata["files"]["inputs"]

    prior_add_on_df = _read_saved_table(paths, input_files, "prior_add_on")
    xa_df = _read_saved_table(paths, input_files, "xa")
    P_df = _read_saved_table(paths, input_files, "P_used")
    ds = xr.load_dataset(paths["results"] / metadata["files"]["results"]["analytical_results"])

    state_index = pd.Index([str(v) for v in ds["state"].values], name="state")
    if xa_df is None:
        raise ValueError("Saved analytical inputs must include xa")
    prior_add_on = None if prior_add_on_df is None else _first_column(
        prior_add_on_df, index=state_index, fill_value=0.0, name="prior_add_on"
    )
    xa = _first_column(xa_df, index=state_index, fill_value=0.0, name="xa")

    xhat = np.asarray(ds["xhat"].values, dtype=float).reshape(-1)
    ak = _coerce_square_dataframe(ds["ak"].values, state_index, "ak")
    shat = _coerce_square_dataframe(ds["shat"].values, state_index, "shat")
    prior_sigma = _state_sigma_from_covariance(P_df, state_index, "prior_cov")
    posterior_sigma = _state_sigma_from_covariance(shat, state_index, "posterior_cov")

    results = _build_interpreted_state_results(
        state_index,
        xa,
        xhat,
        prior_sigma=prior_sigma,
        posterior_sigma=posterior_sigma,
        prior_add_on=prior_add_on,
    )
    results["xhat"] = xhat
    results["ak"] = np.diag(ak.values)
    results.to_csv(paths["tables"] / "state_results_raw.csv")

    shat_diag = np.clip(np.diag(shat.values), a_min=0.0, a_max=None)
    denom = np.sqrt(np.outer(shat_diag, shat_diag))
    corr = np.divide(shat.values, denom, out=np.zeros_like(shat.values), where=denom > 0)
    corr_df = pd.DataFrame(corr, index=state_index, columns=state_index)

    ak.to_csv(paths["tables"] / "ak_raw.csv")
    shat.to_csv(paths["tables"] / "posterior_covariance_raw.csv")
    corr_df.to_csv(paths["tables"] / "posterior_correlation_raw.csv")

    return {
        "results": results,
        "prior_add_on": prior_add_on_df,
        "xa": xa_df,
        "prior_cov": P_df,
        "posterior_cov": shat,
        "ak_matrix": ak,
        "metadata": metadata,
    }


def postprocess_mcmc_outputs(
    output_path,
    hdi_prob=0.95,
    *,
    prior_handling_mode="sampled",
    prior_handling_sample_size=1000,
    prior_handling_random_seed=42,
):
    """Load saved MCMC outputs and build the basic raw state-level products.

    This function summarizes saved sampling results and directly interprets them
    using one shared rule across analytical and MCMC workflows:
    - if prior_add_on exists, it is added back to both xa and sampled values
    - if prior_add_on does not exist, results are interpreted directly from xa and
      sampled values

    When prior_handling_mode="sampled", the prior mean and prior covariance are
    rebuilt from the saved x_prior builder metadata. In that case, the sampled
    prior mean overrides any xa values passed in at save time.
    """
    paths = _ensure_output_dirs(output_path)
    metadata = _read_metadata(paths["results"] / "mcmc_run_metadata.json")
    input_files = metadata["files"]["inputs"]
    x_var_name = metadata.get("x_var_name", "x_prior")

    prior_add_on_df = _read_saved_table(paths, input_files, "prior_add_on")
    xa_df = _read_saved_table(paths, input_files, "xa")
    P_df = _read_saved_table(paths, input_files, "P_used")

    idata = az.from_netcdf(paths["results"] / metadata["files"]["results"]["mcmc_results"])
    out_dic = _mcmc_post_process(idata, x_names=[x_var_name], hdi_prob=hdi_prob)

    if xa_df is None:
        raise ValueError("Saved MCMC inputs must include xa")
    state_index = pd.Index([str(v) for v in xa_df.index], name="state")
    prior_add_on = None if prior_add_on_df is None else _first_column(
        prior_add_on_df, index=state_index, fill_value=0.0, name="prior_add_on"
    )
    xa = _first_column(xa_df, index=state_index, fill_value=0.0, name="xa")
    posterior_cov = _coerce_square_dataframe(
        _posterior_cov_from_idata(idata, x_var_name),
        state_index,
        "posterior_covariance",
    )

    prior_mode = str(prior_handling_mode).lower()
    if prior_mode == "sampled":
        warnings.warn(
            "prior_handling_mode='sampled' rebuilds the prior from x_prior metadata; "
            "sampled prior mean and uncertainty will override the saved xa values.",
            stacklevel=2,
        )

    prior_mean_sampled, prior_cov_for_sigma = _resolve_mcmc_prior_statistics(
        metadata,
        state_index=state_index,
        saved_cov=P_df,
        prior_handling_mode=prior_handling_mode,
        prior_handling_sample_size=prior_handling_sample_size,
        prior_handling_random_seed=prior_handling_random_seed,
    )
    if prior_mean_sampled is not None:
        xa = prior_mean_sampled.copy()
        xa_df = xa.to_frame(name="xa")
    prior_sigma = _state_sigma_from_covariance(prior_cov_for_sigma, state_index, "prior_cov")
    posterior_sigma = _state_sigma_from_covariance(posterior_cov, state_index, "posterior_cov")

    results = _build_interpreted_state_results(
        state_index,
        xa,
        np.asarray(out_dic[x_var_name]["x_mean"], dtype=float).reshape(-1),
        prior_sigma=prior_sigma,
        posterior_sigma=posterior_sigma,
        prior_add_on=prior_add_on,
    )
    results["x_mean"] = np.asarray(out_dic[x_var_name]["x_mean"], dtype=float).reshape(-1)
    results["x_sd"] = posterior_sigma.values
    results["x_hdi_low"] = np.asarray(out_dic[x_var_name]["x_hdi_low"], dtype=float).reshape(-1)
    results["x_hdi_high"] = np.asarray(out_dic[x_var_name]["x_hdi_high"], dtype=float).reshape(-1)
    results.to_csv(paths["tables"] / "state_results_raw.csv")

    diag_var_names = [x_var_name]
    if "R_prior_extra_variance" in idata.posterior:
        diag_var_names.append("R_prior_extra_variance")

    diag = _mcmc_diagnostics(idata, var_names=diag_var_names)
    diag_df = pd.DataFrame.from_dict(diag, orient="index", columns=["value"])
    diag_df.index.name = "metric"
    diag_df.to_csv(paths["tables"] / "mcmc_diagnostics.csv")

    summary_df = az.summary(idata, var_names=diag_var_names)
    rename_map = {f"{x_var_name}[{i}]": str(label) for i, label in enumerate(state_index)}
    summary_df = summary_df.rename(index=rename_map)
    summary_df.to_csv(paths["tables"] / "mcmc_posterior_summary.csv")

    return {
        "results": results,
        "prior_add_on": prior_add_on_df,
        "xa": xa_df,
        "prior_cov_saved": P_df,
        "prior_cov_for_sigma": prior_cov_for_sigma,
        "prior_handling_mode": str(prior_handling_mode).lower(),
        "posterior_cov": posterior_cov,
        "diagnostics": diag_df,
        "summary": summary_df,
        "metadata": metadata,
    }

# Internal helper: MCMC diagnostics and minimal summaries
def _mcmc_diagnostics(
    idata,
    var_names=("x_prior",),
    ess_threshold=400,
    rhat_threshold=1.01,
):
    """Basic MCMC diagnostics for inversion results."""
    diag = {}

    ess = az.ess(idata, var_names=var_names, method="bulk")
    ess_vals = np.asarray(ess.to_array())
    diag["ess_min"] = float(np.nanmin(ess_vals))
    diag["ess_median"] = float(np.nanmedian(ess_vals))
    diag["ess_ok"] = diag["ess_min"] >= ess_threshold

    rhat = az.rhat(idata, var_names=var_names)
    rhat_vals = np.asarray(rhat.to_array())
    diag["rhat_max"] = float(np.nanmax(rhat_vals))
    diag["rhat_ok"] = diag["rhat_max"] <= rhat_threshold

    div = idata.sample_stats.get("diverging", None)
    if div is not None:
        n_div = int(div.sum().values)
    else:
        n_div = 0
    diag["n_divergences"] = n_div

    n_draws = idata.posterior.sizes.get("draw", 1)
    n_chains = idata.posterior.sizes.get("chain", 1)
    diag["divergence_frac"] = n_div / (n_draws * n_chains)

    try:
        bfmi = az.bfmi(idata)
        diag["bfmi_min"] = float(np.nanmin(bfmi))
        diag["bfmi_ok"] = diag["bfmi_min"] > 0.2
    except Exception:
        diag["bfmi_min"] = None
        diag["bfmi_ok"] = None

    return diag


def _mcmc_post_process(idata, x_names=("x_prior",), hdi_prob=0.95):
    """Minimal MCMC post-processing for sampled state variables."""
    out_dic = {}
    for x_name in x_names:
        if x_name in getattr(idata, "posterior", {}):
            x = np.asarray(idata.posterior[x_name])
            x_s = x.reshape((-1, x.shape[-1]))

            x_mean = x_s.mean(axis=0)
            x_sd = x_s.std(axis=0, ddof=1)
            x_hdi = az.hdi(x_s, hdi_prob=hdi_prob)

            out_dic[x_name] = {
                "x_mean": x_mean,
                "x_sd": x_sd,
                "x_hdi_low": x_hdi[:, 0],
                "x_hdi_high": x_hdi[:, 1],
            }

    return out_dic

# Internal helpers: filesystem, serialization, and table coercion
def _ensure_output_dirs(output_path):
    output_dir = Path(output_path)
    inputs_dir = output_dir / "inputs"
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    yearly_diag_dir = figures_dir / "yearly_diagnostics"
    yearly_matrix_dir = tables_dir / "yearly_matrices"

    for path in (
        output_dir,
        inputs_dir,
        results_dir,
        figures_dir,
        tables_dir,
        yearly_diag_dir,
        yearly_matrix_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "output": output_dir,
        "inputs": inputs_dir,
        "results": results_dir,
        "figures": figures_dir,
        "tables": tables_dir,
        "yearly_diag": yearly_diag_dir,
        "yearly_matrix": yearly_matrix_dir,
    }


def _index_to_strings(index_like):
    return [str(v) for v in index_like]


def _jsonify(value):
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (pd.Index, pd.Series)):
        return _jsonify(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _write_metadata(path, metadata):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(metadata), f, indent=2, ensure_ascii=False)


def _read_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_dataframe(data, index=None, columns=None, value_name="value"):
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return data.to_frame(name=data.name or value_name)

    arr = np.asarray(data)
    if arr.ndim == 1:
        return pd.DataFrame(arr, index=index, columns=[value_name])
    if arr.ndim == 2:
        return pd.DataFrame(arr, index=index, columns=columns)
    raise ValueError(f"Unsupported array shape for DataFrame conversion: {arr.shape}")


def _coerce_square_dataframe(matrix_like, index, name):
    if matrix_like is None:
        return None

    index = pd.Index([str(v) for v in index], name="state")
    if isinstance(matrix_like, pd.DataFrame):
        df = matrix_like.copy()
        df.index = df.index.map(str)
        df.columns = df.columns.map(str)
        return df.loc[index, index]

    arr = np.asarray(matrix_like, dtype=float)
    n_state = len(index)
    if arr.shape != (n_state, n_state):
        raise ValueError(f"{name} has shape {arr.shape}, expected {(n_state, n_state)}")
    return pd.DataFrame(arr, index=index, columns=index)


def _summarize_metadata_value(value, *, max_preview=8):
    if isinstance(value, dict):
        return {str(k): _summarize_metadata_value(v, max_preview=max_preview) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) <= max_preview:
            return [_summarize_metadata_value(v, max_preview=max_preview) for v in value]
        return {
            "kind": type(value).__name__,
            "length": len(value),
            "preview": [_summarize_metadata_value(v, max_preview=max_preview) for v in value[:max_preview]],
        }
    if isinstance(value, (pd.Index, pd.Series)):
        return _summarize_metadata_value(value.tolist(), max_preview=max_preview)

    arr = np.asarray(value)
    if arr.ndim == 0:
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    flat = arr.ravel()
    if flat.size <= max_preview:
        return flat.tolist()
    return {
        "kind": "ndarray",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "preview": flat[:max_preview].tolist(),
    }


def _infer_prior_spec(prior, *, fallback_type):
    if callable(prior) and hasattr(prior, "_prior_spec"):
        return _jsonify(prior._prior_spec)

    arr = np.asarray(prior)
    if arr.ndim == 0:
        return {"type": fallback_type, "parameters": {"value": _summarize_metadata_value(arr.item())}}
    return {
        "type": fallback_type,
        "parameters": _summarize_metadata_value(arr),
    }


def _pack_other_specific_data(
    other_specific_data=None,
    *,
    dropped_rows=None,
    dropped_cols=None,
    obs_group=None,
    obs_group_name="group",
    obs_site_year_group=None,
):
    packed = {}
    if other_specific_data:
        packed.update(_jsonify(other_specific_data))

    if dropped_rows is not None:
        if isinstance(dropped_rows, (int, np.integer)):
            packed["dropped_rows"] = int(dropped_rows)
        else:
            packed["dropped_rows"] = _index_to_strings(dropped_rows)
    if dropped_cols is not None:
        packed["dropped_cols"] = _index_to_strings(dropped_cols)

    if obs_group is None and obs_site_year_group is not None:
        obs_group = obs_site_year_group
        obs_group_name = "site_year"
    if obs_group is not None:
        packed["obs_group"] = _index_to_strings(obs_group)
        packed["obs_group_name"] = str(obs_group_name)

    return packed or None

# Internal helpers: persist inversion inputs
def _save_input_tables(inputs_dir, H_used, obs_used, *, R_used=None, P_used=None, xa=None, prior_add_on=None):
    file_map = {}

    H_df = _coerce_dataframe(H_used)
    H_path = inputs_dir / "H_used.csv"
    H_df.to_csv(H_path)
    file_map["H_used"] = str(H_path.name)

    obs_df = _coerce_dataframe(obs_used)
    obs_path = inputs_dir / "obs_used.csv"
    obs_df.to_csv(obs_path)
    file_map["obs_used"] = str(obs_path.name)

    xa_df = _coerce_dataframe(xa, value_name="xa")
    if xa_df is None:
        raise ValueError("xa is required and cannot be None")
    state_index = xa_df.index

    prior_df = _coerce_dataframe(prior_add_on, index=state_index, value_name="prior_add_on")
    if prior_df is not None:
        prior_path = inputs_dir / "prior_add_on.csv"
        prior_df.to_csv(prior_path)
        file_map["prior_add_on"] = str(prior_path.name)

    obs_index = obs_df.index if obs_df is not None else None

    if R_used is not None:
        R_df = _coerce_dataframe(R_used, index=obs_index, columns=obs_index, value_name="R")
        R_path = inputs_dir / "R_used.csv"
        R_df.to_csv(R_path)
        file_map["R_used"] = str(R_path.name)

    if P_used is not None:
        P_df = _coerce_dataframe(P_used, index=state_index, columns=state_index, value_name="P")
        P_path = inputs_dir / "P_used.csv"
        P_df.to_csv(P_path)
        file_map["P_used"] = str(P_path.name)

    xa_df = _coerce_dataframe(xa, index=state_index, value_name="xa")
    xa_path = inputs_dir / "xa.csv"
    xa_df.to_csv(xa_path)
    file_map["xa"] = str(xa_path.name)

    return file_map

# Internal helpers: load normalized saved inputs and posterior samples
def _read_saved_table(paths, input_files, *keys):
    for key in keys:
        if key in input_files:
            path = paths["inputs"] / input_files[key]
            return pd.read_csv(path, index_col=0)
    return None


def _first_column(df_like, *, index, fill_value=0.0, name="value"):
    if df_like is None:
        return pd.Series(fill_value, index=index, dtype=float, name=name)
    if isinstance(df_like, pd.Series):
        series = df_like.copy()
    else:
        series = df_like.iloc[:, 0].copy()
    series.index = series.index.map(str)
    return series.reindex(index, fill_value=fill_value).astype(float).rename(name)


def _posterior_cov_from_idata(idata, var_name):
    samples = np.asarray(idata.posterior[var_name], dtype=float)
    if samples.ndim < 3:
        raise ValueError(f"Posterior variable {var_name!r} must have chain/draw/state dimensions")
    samples_2d = samples.reshape((-1, samples.shape[-1]))
    if samples_2d.shape[1] == 1:
        return np.array([[np.var(samples_2d[:, 0], ddof=1)]], dtype=float)
    return np.cov(samples_2d, rowvar=False, ddof=1)


def _state_sigma_from_covariance(cov_like, index, name):
    cov_df = _coerce_square_dataframe(cov_like, index, name)
    if cov_df is None:
        return None
    sigma = np.sqrt(np.clip(np.diag(cov_df.values.astype(float)), a_min=0.0, a_max=None))
    return pd.Series(sigma, index=cov_df.index, name=f"{name}_sigma")


def _build_interpreted_state_results(
    state_index,
    xa,
    sampled_values,
    *,
    prior_sigma=None,
    posterior_sigma=None,
    prior_add_on=None,
):
    xa = _first_column(xa, index=state_index, fill_value=0.0, name="xa")
    sampled_values = np.asarray(sampled_values, dtype=float).reshape(-1)

    results = pd.DataFrame(index=state_index)
    results["xa"] = xa.values

    if prior_add_on is not None:
        prior_add_on = _first_column(prior_add_on, index=state_index, fill_value=0.0, name="prior_add_on")
        results["prior_add_on"] = prior_add_on.values
        results["prior"] = xa.values + prior_add_on.values
        results["delta_x"] = sampled_values - xa.values
        results["posterior"] = sampled_values + prior_add_on.values
    else:
        results["prior"] = xa.values
        results["delta_x"] = sampled_values - xa.values
        results["posterior"] = sampled_values

    if prior_sigma is not None:
        results["state_prior_sigma"] = pd.Series(prior_sigma, index=state_index).astype(float).values
    if posterior_sigma is not None:
        results["state_posterior_sigma"] = pd.Series(posterior_sigma, index=state_index).astype(float).values
    return results



def _mcmc_prior_builder_from_metadata(metadata):
    prior_spec = metadata.get("x_prior", {})
    prior_type = prior_spec.get("type")
    params = dict(prior_spec.get("parameters", {}))
    if not prior_type:
        raise NotImplementedError("MCMC prior sampling requires metadata['x_prior']['type']")

    builder_factory = getattr(_mcmc_builder_module, str(prior_type), None)
    if not callable(builder_factory):
        raise NotImplementedError(f"Prior sampling is not implemented for x_prior type {prior_type!r}")

    try:
        return builder_factory(**params)
    except TypeError as exc:
        raise NotImplementedError(
            f"Could not rebuild x_prior builder {prior_type!r} from metadata parameters"
        ) from exc


def _sample_mcmc_prior_statistics(metadata, *, state_index, n_samples=1000, random_seed=42):
    builder = _mcmc_prior_builder_from_metadata(metadata)

    with pm.Model():
        bundle = builder()
        prior_samples = pm.sample_prior_predictive(
            samples=int(n_samples),
            var_names=[bundle.main.name],
            random_seed=random_seed,
            return_inferencedata=False,
        )

    draws = np.asarray(prior_samples[bundle.main.name], dtype=float)
    if draws.ndim == 1:
        samples_2d = draws.reshape(-1, 1)
    else:
        samples_2d = draws.reshape(draws.shape[0], -1)

    state_index = pd.Index([str(v) for v in state_index], name="state")
    if samples_2d.shape[1] != len(state_index):
        raise ValueError(
            f"Sampled prior has {samples_2d.shape[1]} states, expected {len(state_index)} from state_index"
        )

    prior_mean = pd.Series(samples_2d.mean(axis=0), index=state_index, name="xa")
    cov = np.cov(samples_2d, rowvar=False, ddof=1)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=float)
    prior_cov = pd.DataFrame(cov, index=state_index, columns=state_index)
    return prior_mean, prior_cov


def _resolve_mcmc_prior_statistics(
    metadata,
    *,
    state_index,
    saved_cov=None,
    prior_handling_mode="sampled",
    prior_handling_sample_size=1000,
    prior_handling_random_seed=42,
):
    mode = str(prior_handling_mode).lower()
    state_index = pd.Index([str(v) for v in state_index], name="state")

    if mode == "none":
        return None, None

    if mode == "saved":
        if saved_cov is None:
            return None, None
        return None, _coerce_square_dataframe(saved_cov, state_index, "prior_cov_saved")

    if mode != "sampled":
        raise ValueError("prior_handling_mode must be one of: 'sampled', 'saved', 'none'")

    return _sample_mcmc_prior_statistics(
        metadata,
        state_index=state_index,
        n_samples=prior_handling_sample_size,
        random_seed=prior_handling_random_seed,
    )
