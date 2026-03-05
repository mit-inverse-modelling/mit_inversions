#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================

Created:     2026-02-01
Last update: 2026-03-05

===============================================================
"""

import arviz as az
import numpy as np

def mcmc_diagnostics(
    idata,
    var_names=("x_prior",),
    ess_threshold=400,
    rhat_threshold=1.01,
):
    """
    Basic MCMC diagnostics for inversion results.

    Checks:
    - Effective sample size (ESS)
    - R-hat convergence
    - NUTS divergences
    - Energy/BFMI (optional warning)

    Returns
    -------
    diag : dict
        Summary statistics and boolean flags.
    """

    diag = {}

    # -------------------------------------------------
    # 1) ESS (effective sample size)
    # -------------------------------------------------
    ess = az.ess(idata, var_names=var_names, method="bulk")
    ess_vals = np.asarray(ess.to_array())

    diag["ess_min"] = float(np.nanmin(ess_vals))
    diag["ess_median"] = float(np.nanmedian(ess_vals))
    diag["ess_ok"] = diag["ess_min"] >= ess_threshold

    # -------------------------------------------------
    # 2) R-hat (chain convergence)
    # -------------------------------------------------
    rhat = az.rhat(idata, var_names=var_names)
    rhat_vals = np.asarray(rhat.to_array())

    diag["rhat_max"] = float(np.nanmax(rhat_vals))
    diag["rhat_ok"] = diag["rhat_max"] <= rhat_threshold

    # -------------------------------------------------
    # 3) Divergences (NUTS only, informational)
    # -------------------------------------------------
    div = idata.sample_stats.get("diverging", None)
    if div is not None:
        n_div = int(div.sum().values)
    else:
        n_div = 0

    diag["n_divergences"] = n_div
    #diag["divergence_ok"] = (n_div == 0)
    n_draws = idata.posterior.sizes.get("draw", 1)
    n_chains = idata.posterior.sizes.get("chain", 1)
    diag["divergence_frac"] = n_div / (n_draws * n_chains)

    # -------------------------------------------------
    # 4) Energy / BFMI (quick health check)
    # -------------------------------------------------
    try:
        bfmi = az.bfmi(idata)
        diag["bfmi_min"] = float(np.nanmin(bfmi))
        diag["bfmi_ok"] = diag["bfmi_min"] > 0.2
    except Exception:
        diag["bfmi_min"] = None
        diag["bfmi_ok"] = None



    return diag


def mcmc_post_process(idata, H, y, x_names=["x_prior"], hdi_prob=0.95):
    """
    Minimal post-processing:
    - x posterior mean/sd/HDI
    - y_hat mean (H @ x_mean)
    - residuals (y - y_hat)
    - (optional) R summary if present in posterior
    """
    H = np.asarray(H)
    y = np.asarray(y).ravel()

    out_dic = {}
    for x_name in x_names:
        if x_name in getattr(idata,"posterior",{}):
            # -------- x samples: (chains, draws, n) -> (S, n)
            x = np.asarray(idata.posterior[x_name])
            x_s = x.reshape((-1, x.shape[-1]))

            x_mean = x_s.mean(axis=0)
            x_sd   = x_s.std(axis=0, ddof=1)

            #lo_q = (1 - hdi_prob) / 2
            #hi_q = 1 - lo_q
            #x_hdi = np.quantile(x_s, [lo_q, hi_q], axis=0)  # (2, n)
            x_hdi = az.hdi(x_s, hdi_prob=hdi_prob)  # shape: (n, 2)

            # -------- predicted y using posterior mean (fast + stable)
            y_hat = H @ x_mean
            resid = y - y_hat

            out = {
                "x_mean": x_mean,
                "x_sd": x_sd,
                "x_hdi_low": x_hdi[:,0],
                "x_hdi_high": x_hdi[:,1],
                "y_hat": y_hat,
                "residual": resid,
            }
            out_dic[x_name] = out

    return out_dic
