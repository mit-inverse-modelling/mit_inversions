#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================

Created:     2026-02-01
Last update: 2026-03-05

===============================================================
"""
import numpy as np
from typing import Dict, List
import pymc as pm
from dataclasses import dataclass

@dataclass
class PriorBundle:
    main: pm.TensorVariable                 
    rvs: Dict[str, List[pm.TensorVariable]] 

# ---------- x priors ----------
FUNCTION_DICT = {
    "uniform": pm.Uniform,
    "flat": pm.Flat,
    "halfflat": pm.HalfFlat,
    "normal": pm.Normal,
    "truncatednormal": pm.TruncatedNormal,
    "halfnormal": pm.HalfNormal,
    "skewnormal": pm.SkewNormal,
    "beta": pm.Beta,
    "kumaraswamy": pm.Kumaraswamy,
    "exponential": pm.Exponential,
    "laplace": pm.Laplace,
    "studentt": pm.StudentT,
    "halfstudentt": pm.HalfStudentT,
    "cauchy": pm.Cauchy,
    "halfcauchy": pm.HalfCauchy,
    "gamma": pm.Gamma,
    "inversegamma": pm.InverseGamma,
    "weibull": pm.Weibull,
    "lognormal": pm.LogNormal,   # <- fix name
    "chisquared": pm.ChiSquared,
    "wald": pm.Wald,
    "pareto": pm.Pareto,
    "exgaussian": pm.ExGaussian,
    "vonmises": pm.VonMises,
    "triangular": pm.Triangular,
    "gumbel": pm.Gumbel,
    "rice": pm.Rice,
    "logistic": pm.Logistic,
    "logitnormal": pm.LogitNormal,
    "interpolated": pm.Interpolated,
}
def make_x_prior_normal(n, mu=0.0, sigma=1.0, name="x_prior"):
    """Independent Normal prior for x (shape n)."""
    mu_arr = np.asarray(mu)
    if mu_arr.ndim == 0:
        mu_val = mu_arr
    else:
        mu_arr = np.squeeze(mu_arr)
        if mu_arr.ndim != 1 or mu_arr.shape[0] != n:
            raise ValueError(f"mu must be scalar or array of length {n}")
        mu_val = mu_arr
        
    
    sigma_arr = np.asarray(sigma)
    if np.any(sigma_arr <=0):
        raise ValueError("sigma must be > 0")
    if sigma_arr.ndim == 0:
        sigma_val = sigma_arr
    else:
        sigma_arr = np.squeeze(sigma_arr)
        if sigma_arr.ndim != 1 or sigma_arr.shape[0] != n:
            raise ValueError(f"sigma must be scalar or array of length {n}")
        sigma_val = sigma_arr
    
    
    def builder():
        x = pm.Normal(name, mu=mu_val, sigma=sigma_val, shape=n)
        return PriorBundle(
            main = x,
            rvs={'NUTS':[x]}
        )
    

    return builder


def make_x_prior_mvnormal(n, mu=0.0, P=None, name="x_prior"):
    """Correlated MVN prior with fixed covariance P (shape n x n)."""

    P = np.asarray(P)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix (n,n).")
    if P.shape[0] != n:
        raise ValueError(f"P must have shape ({n},{n}).")

    # ---- handle mu: scalar or array-like ----
    mu_arr = np.asarray(mu)
    if mu_arr.ndim == 0:
        mu_val = np.full(n, mu_arr)
    else:
        mu_arr = np.squeeze(mu_arr)
        if mu_arr.ndim != 1 or mu_arr.shape[0] != n:
            raise ValueError(f"mu must be scalar or array of length {n}")
        mu_val =  mu_arr


    def builder():
        x = pm.MvNormal(name, mu=mu_val, cov=P, shape=n)
        return PriorBundle(
            main = x,
            rvs={'NUTS':[x]}
        )

    return builder

def make_x_prior_scaling(n, x0=0.0, name="x_prior", scaling_prior = None ):
    """build the sample of x with a scaling factor."""
    

    if scaling_prior is None:
        scaling_prior = {"pdf":"lognormal","mu":1.0,"sigma":1.0}
        
    if not isinstance(scaling_prior, dict):
        raise ValueError("scaling_prior must be a diction include key 'pdf'")    
      
    x0_arr = np.asarray(x0)

    if x0_arr.ndim == 0:
        x0_val = x0_arr          # scalar
    else:
        x0_arr = np.squeeze(x0_arr)
        if x0_arr.ndim != 1 or x0_arr.shape[0] != n:
            raise ValueError("x0 must be scalar or array of length n")
        x0_val = x0_arr          # (n,)        
        
     # ---- prior spec ----
 
    

    
    nuts_vars = []
    slice_vars = []
    def builder():

        scale,sampled_RV,solv_dist = solve_rv(f"{name}_scale", scaling_prior, shape=n)
        if solv_dist:
            sampled_RV.append(scale)
            #print(f"Added {scale} to sampled_RV, now {sampled_RV}")
        for rv in sampled_RV:
            if rv.name.endswith("_scale") or rv.ndim > 0:
                nuts_vars.append(rv)
            else:
                slice_vars.append(rv)
        x = pm.Deterministic(name, x0_val * scale)
        
        return PriorBundle(
            main = x,
            rvs={'NUTS':nuts_vars,'Slice':slice_vars}
        )

    return builder


# ---------- R priors ----------

def make_R_prior_sigma(m, sigma=1.0, name="R_prior"):
    """Independent observation error sigma vector (shape m)."""
    sigma_arr = np.asarray(sigma)
    if sigma_arr.ndim == 0:
        sigma_val = sigma_arr
    else:
        sigma_arr = np.squeeze(sigma_arr)
        if sigma_arr.ndim != 1 or sigma_arr.shape[0] != m:
            raise ValueError(f"sigma must be scalar or array of length {m}")
        sigma_val = sigma_arr
        
    def builder():
        sigma = pm.HalfNormal(name, sigma=sigma_val, shape=m)
        return PriorBundle(
            main = sigma,
            rvs={'Slice':[sigma]}
        )
    return builder


def make_R_prior_cov_lkj(m, eta=2.0, sd_sigma=1.0,
                         chol_name="R_prior_chol"):
    """Full covariance for obs errors via LKJCholeskyCov; returns cov matrix."""
    
    def builder():
        L, corr, sigmas = pm.LKJCholeskyCov(
            chol_name,
            n=m,
            eta=eta,
            sd_dist=pm.HalfNormal.dist(sd_sigma),
            compute_corr=True,
        )
        sigma = pm.Deterministic("R_prior", pm.math.dot(L, L.T))
        return PriorBundle(
            main = sigma,
            rvs={'Slice':[L, sigmas]}
        )
    return builder


def solve_rv(param_path, v, *,  shape=None):
    """
    v can be:
      - constant
      - RV spec dict: {"pdf": "...", <params...>}
    param_path: string used to build unique RV names (e.g. "x_prior.scale.sigma")
    shape: only applied to the RV created at *this level* (typically only top-level)
    """
    # constant
    solv_distribution = False
    sampled_RV = []
    if not isinstance(v, dict):
        #print(f"Returned sampled_RV {sampled_RV}")
        return v, sampled_RV, solv_distribution
    solv_distribution = True
    if "pdf" not in v:
        raise ValueError(f"{param_path} RV spec must be a diction include 'pdf'")

    pdf = str(v["pdf"]).lower()
    if pdf not in FUNCTION_DICT:
        raise ValueError(f"Unknown pdf '{pdf}' for {param_path}")
    #print(f"Solving RV for {param_path} with pdf {pdf}")
    # recursively resolve parameters
    params = {}
    for k, vv in v.items():
        if k == "pdf":
            continue
        #print(f"Solving RV for {param_path}.{k} with value {vv}")
        params[k],sampled_RV2, solv_distribution_step = solve_rv(f"{param_path}.{k}", vv)
        sampled_RV.extend(sampled_RV2)
        #print(f"get sampled_RV from {sampled_RV2}")
        if solv_distribution_step:
            sampled_RV.append(params[k])
            #print(f"Added {params[k]} to sampled_RV, now {sampled_RV}")

   
    rv_name = param_path.replace(".", "_")  # pytensor/pymc name safe

    # only the RV created at this level gets `shape`
    #print(f"Returned sampled_RV {sampled_RV}")
    if shape is None:
        return FUNCTION_DICT[pdf](rv_name, **params), sampled_RV, solv_distribution
    else:
        return FUNCTION_DICT[pdf](rv_name, **params, shape=shape), sampled_RV, solv_distribution