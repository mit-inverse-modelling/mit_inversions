import numpy as np
from typing import Dict, List
import pymc as pm
from dataclasses import dataclass

@dataclass
class PriorBundle:
    main: pm.TensorVariable                 
    rvs: Dict[str, List[pm.TensorVariable]] 


def _summarize_spec_value(value, *, max_preview=8):
    if isinstance(value, dict):
        return {str(k): _summarize_spec_value(v, max_preview=max_preview) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize_spec_value(v, max_preview=max_preview) for v in value]

    arr = np.asarray(value)
    if arr.ndim == 0:
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    return arr.tolist()


def _attach_prior_spec(builder, prior_type, parameters):
    builder._prior_spec = {
        "type": prior_type,
        "parameters": _summarize_spec_value(parameters),
    }
    return builder

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
    

    return _attach_prior_spec(
        builder,
        "make_x_prior_normal",
        {"n": n, "mu": mu_val, "sigma": sigma_val, "name": name},
    )


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

    return _attach_prior_spec(
        builder,
        "make_x_prior_mvnormal",
        {"n": n, "mu": mu_val, "P": P, "name": name},
    )

def make_x_prior_scaling(n, x0=0.0, name="x_prior", scaling_prior = None ):
    """
    Build the sampled state x itself using a multiplicative scaling prior.

    The latent scaling factor is sampled and the returned state is

        x = x0 * scale

    so the quantity entering the forward model is the absolute scaled state,
    not the increment relative to x0.

    This is the right builder when H is meant to act directly on the full
    sampled state.
    """
    

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

    return _attach_prior_spec(
        builder,
        "make_x_prior_scaling",
        {"n": n, "x0": x0_val, "name": name, "scaling_prior": scaling_prior},
    )

def make_x_prior_scaling_increment(n, x0=0.0, name="x_prior", scaling_prior=None):
    """
    Build the sampled state increment x - x0 using a multiplicative scaling prior.

    The latent scaling factor is sampled as usual, but the quantity returned to
    the forward model is the increment:

        x - x0 = x0 * (scale - 1)

    This is useful when the inversion is written in terms of state increments,
    while the underlying physical state is still interpreted as

        x = x0 * scale

    In other words:
      - physical state: x = x0 * scale
      - quantity returned to the forward model: x - x0 = x0 * (scale - 1)

    This is the right builder when H maps state perturbations / increments to y.
    """
    

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
        x = pm.Deterministic(name, x0_val * (scale-1))
        
        return PriorBundle(
            main = x,
            rvs={'NUTS':nuts_vars,'Slice':slice_vars}
        )

    return _attach_prior_spec(
        builder,
        "make_x_prior_scaling_increment",
        {"n": n, "x0": x0_val, "name": name, "scaling_prior": scaling_prior},
    )


def make_x_prior_scaling_diff(n, x0=0.0, name="x_prior", scaling_prior=None):
    """
    Backward-compatible alias for make_x_prior_scaling_increment.

    Kept so older notebooks/scripts continue to run, but new code should prefer
    the clearer name make_x_prior_scaling_increment.
    """
    return make_x_prior_scaling_increment(
        n,
        x0=x0,
        name=name,
        scaling_prior=scaling_prior,
    )
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
    return _attach_prior_spec(
        builder,
        "make_R_prior_sigma",
        {"m": m, "sigma": sigma_val, "name": name},
    )


def make_R_prior_sigma_additive_variance(m, R=0.0, extra_prior=None,
                                         group_index=None, name="R_prior"):
    """
    Observation-error sigma prior with additive variance inflation:

        sigma_i = sqrt(R_i + A_{g(i)})

    where:
      - R is a known variance floor (scalar or length-m array)
      - A is a sampled non-negative variance increment
      - g(i) maps each observation i to a shared increment group

    Parameters
    ----------
    m : int
        Number of observations.
    R : float or array-like, default 0.0
        Known variance floor. Can be a scalar or a length-m array.
        This is the "minimum" variance term; when A = 0, sigma = sqrt(R).
    extra_prior : dict or constant, optional
        Prior specification for the additional variance term A.
        Defaults to {"pdf": "halfnormal", "sigma": 1.0}.
        For valid sqrt(R + A), this should usually have support on [0, inf).
    group_index : array-like of length m, optional
        Observations with the same group label share one sampled A value.
        If None, all m observations are independent.
    name : str, default "R_prior"
        Name of the sigma variable returned to the likelihood.
    """
    if int(m) <= 0:
        raise ValueError("m must be > 0")
    m = int(m)

    R_arr = np.asarray(R, dtype=float)
    if R_arr.ndim == 0:
        R_val = np.full(m, float(R_arr))
    else:
        R_arr = np.squeeze(R_arr)
        if R_arr.ndim != 1 or R_arr.shape[0] != m:
            raise ValueError(f"R must be scalar or array of length {m}")
        R_val = R_arr
    if np.any(~np.isfinite(R_val)):
        raise ValueError("R must contain only finite values")
    if np.any(R_val < 0):
        raise ValueError("R must be >= 0 because sigma = sqrt(R + A)")

    if group_index is None:
        group_codes = np.arange(m, dtype=int)
        n_groups = m
    else:
        group_arr = np.asarray(group_index)
        group_arr = np.squeeze(group_arr)
        if group_arr.ndim != 1 or group_arr.shape[0] != m:
            raise ValueError(f"group_index must be a 1D array of length {m}")
        _, group_codes = np.unique(group_arr, return_inverse=True)
        n_groups = int(group_codes.max()) + 1

    if extra_prior is None:
        extra_prior = {"pdf": "halfnormal", "sigma": 1.0}

    def builder():
        nuts_vars = []
        slice_vars = []

        extra_variance_group, sampled_RV, solv_dist = solve_rv(
            f"{name}_extra_variance",
            extra_prior,
            shape=n_groups,
        )

        if solv_dist:
            sampled_RV.append(extra_variance_group)
        else:
            extra_arr = np.asarray(extra_variance_group, dtype=float)
            if extra_arr.ndim == 0:
                extra_variance_group = np.full(n_groups, float(extra_arr))
            else:
                extra_arr = np.squeeze(extra_arr)
                if extra_arr.ndim != 1 or extra_arr.shape[0] != n_groups:
                    raise ValueError(
                        f"Constant extra_prior must be scalar or array of length {n_groups}"
                    )
                extra_variance_group = extra_arr
            if np.any(~np.isfinite(extra_variance_group)):
                raise ValueError("Constant extra_prior must contain only finite values")

        for rv in sampled_RV:
            if rv.name.endswith("_extra_variance") or rv.ndim > 0:
                nuts_vars.append(rv)
            else:
                slice_vars.append(rv)

        sigma = pm.Deterministic(
            name,
            pm.math.sqrt(R_val + extra_variance_group[group_codes]),
        )

        return PriorBundle(
            main=sigma,
            rvs={"NUTS": nuts_vars, "Slice": slice_vars},
        )

    return _attach_prior_spec(
        builder,
        "make_R_prior_sigma_additive_variance",
        {
            "m": m,
            "R": R_val,
            "extra_prior": extra_prior,
            "group_index": group_index,
            "name": name,
        },
    )


def make_R_prior_sigma_additive(m, R=0.0, extra_prior=None,
                                group_index=None, name="R_prior"):
    """
    Backward-compatible alias for make_R_prior_sigma_additive_variance.
    """
    return make_R_prior_sigma_additive_variance(
        m,
        R=R,
        extra_prior=extra_prior,
        group_index=group_index,
        name=name,
    )


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
    return _attach_prior_spec(
        builder,
        "make_R_prior_cov_lkj",
        {"m": m, "eta": eta, "sd_sigma": sd_sigma, "chol_name": chol_name},
    )


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
