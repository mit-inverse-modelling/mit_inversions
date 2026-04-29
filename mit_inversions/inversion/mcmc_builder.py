import pymc as pm
import numpy as np
import pytensor.tensor as pt

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PriorBundle:
    main: pt.TensorVariable
    rvs: Dict[str, List[pt.TensorVariable]]


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
    "lognormal": pm.LogNormal,
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


# ---------- x priors ----------
def make_x_prior_scaling(n, x0=0.0, name="x_prior", scaling_prior=None):
    """Build an absolute state prior as x = x0 * scale."""
    if scaling_prior is None:
        scaling_prior = {"pdf": "lognormal", "mu": 1.0, "sigma": 1.0}

    if not isinstance(scaling_prior, dict):
        raise ValueError("scaling_prior must be a dictionary including key 'pdf'")

    x0_arr = np.asarray(x0)
    if x0_arr.ndim == 0:
        x0_val = x0_arr
    else:
        x0_arr = np.squeeze(x0_arr)
        if x0_arr.ndim == 0:
            x0_val = x0_arr
            return_scalar = True
        else:
            return_scalar = False
        if not return_scalar and (x0_arr.ndim != 1 or x0_arr.shape[0] != n):
            raise ValueError("x0 must be scalar or array of length n")
        if not return_scalar:
            x0_val = x0_arr

    def builder():
        nuts_vars = []
        slice_vars = []

        scale, sampled_rv, solved_distribution = solve_rv(
            f"{name}_scale",
            scaling_prior,
            shape=n,
        )
        if solved_distribution:
            sampled_rv.append(scale)

        for rv in sampled_rv:
            if rv.name.endswith("_scale") or rv.ndim > 0:
                nuts_vars.append(rv)
            else:
                slice_vars.append(rv)

        x = pm.Deterministic(name, x0_val * scale)
        return PriorBundle(main=x, rvs={"NUTS": nuts_vars, "Slice": slice_vars})

    return _attach_prior_spec(
        builder,
        "make_x_prior_scaling",
        {"n": n, "x0": x0_val, "name": name, "scaling_prior": scaling_prior},
    )


# ---------- R priors ----------
def make_R_prior_sigma_additive(m, R=0.0, extra_prior=None, group_index=None, name="R_prior"):
    """Build sigma_i = sqrt(R_i + A_{g(i)}) with optional grouped additive variance."""
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
        group_codes = np.zeros(m, dtype=int)
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

        extra_variance_group, sampled_rv, solved_distribution = solve_rv(
            f"{name}_extra_variance",
            extra_prior,
            shape=n_groups,
        )

        if solved_distribution:
            sampled_rv.append(extra_variance_group)
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

        for rv in sampled_rv:
            if rv.name.endswith("_extra_variance") or rv.ndim > 0:
                nuts_vars.append(rv)
            else:
                slice_vars.append(rv)

        sigma = pm.Deterministic(
            name,
            pm.math.sqrt(R_val + extra_variance_group[group_codes]),
        )
        return PriorBundle(main=sigma, rvs={"NUTS": nuts_vars, "Slice": slice_vars})

    return _attach_prior_spec(
        builder,
        "make_R_prior_sigma_additive",
        {
            "m": m,
            "R": R_val,
            "extra_prior": extra_prior,
            "group_index": group_index,
            "name": name,
        },
    )


def solve_rv(param_path, v, *, shape=None):
    """Resolve nested random-variable specifications into PyMC RVs."""
    solved_distribution = False
    sampled_rv = []

    if not isinstance(v, dict):
        return v, sampled_rv, solved_distribution

    solved_distribution = True
    if "pdf" not in v:
        raise ValueError(f"{param_path} RV spec must be a dictionary including 'pdf'")

    pdf = str(v["pdf"]).lower()
    if pdf not in FUNCTION_DICT:
        raise ValueError(f"Unknown pdf '{pdf}' for {param_path}")

    params = {}
    for k, vv in v.items():
        if k == "pdf":
            continue
        params[k], sampled_rv_step, solved_step = solve_rv(f"{param_path}.{k}", vv)
        sampled_rv.extend(sampled_rv_step)
        if solved_step:
            sampled_rv.append(params[k])

    rv_name = param_path.replace(".", "_")
    if shape is None:
        return FUNCTION_DICT[pdf](rv_name, **params), sampled_rv, solved_distribution
    return FUNCTION_DICT[pdf](rv_name, **params, shape=shape), sampled_rv, solved_distribution
