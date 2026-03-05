#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================

Created:     2026-02-01
Last update: 2026-03-05

===============================================================
"""

import numpy as np
import pymc as pm
import arviz as az
from geoschem.inversion.mcmc_utils import mcmc_diagnostics, mcmc_post_process

    

def analytical_inversion(H, y, R, xa, P):
    '''
    Define analytical inversion function
    
    Parameters:
    -----------
    H: Sensitivity matrix (Jacobian matrix) - shape (m, n)
    y: Observations - shape (m,) or (m, 1)
    R: Observation error covariance matrix - shape (m, m)
    xa: Prior estimates - shape (n,) or (n, 1)
    P: Prior error covariance matrix - shape (n, n)
    
    Returns:
    --------
    xhat: Posterior estimates - shape (n,) or (n, 1)
    ak: Averaging kernel matrix - shape (n, n)
    shat: Posterior error covariance matrix - shape (n, n)
    
    Raises:
    -------
    TypeError: If inputs cannot be converted to numpy arrays
    ValueError: If matrix dimensions are inconsistent or matrices are not positive definite
    numpy.linalg.LinAlgError: If matrix inversion fails
    
    -------
    The analytical inversion only supports Gaussian Distribution.
    The xa are the mean values of the prior.
    The P is the covariance matrix of the prior.
    
    '''
    
    # Convert to numpy arrays
    try:
        H = np.asarray(H, dtype=float)
        y = np.asarray(y, dtype=float)
        R = np.asarray(R, dtype=float)
        xa = np.asarray(xa, dtype=float)
        P = np.asarray(P, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Failed to convert inputs to numpy arrays: {str(e)}")
    
    # Check for NaN or Inf values
    inputs = {'H': H, 'y': y, 'R': R, 'xa': xa, 'P': P}
    for name, value in inputs.items():
        if np.any(np.isnan(value)):
            raise ValueError(f"{name} contains NaN values")
        if np.any(np.isinf(value)):
            raise ValueError(f"{name} contains Inf values")
    
    # Ensure proper dimensions
    if H.ndim != 2:
        raise ValueError(f"H must be 2-dimensional, got {H.ndim} dimensions")
    
    m, n = H.shape
    
    # Reshape y and xa if needed
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim == 2 and y.shape[1] != 1:
        raise ValueError(f"y must be a column vector or 1D array, got shape {y.shape}")
    
    if xa.ndim == 1:
        xa = xa.reshape(-1, 1)
    elif xa.ndim == 2 and xa.shape[1] != 1:
        raise ValueError(f"xa must be a column vector or 1D array, got shape {xa.shape}")
    
    # Dimension consistency checks
    if y.shape[0] != m:
        raise ValueError(f"y dimension {y.shape[0]} does not match H rows {m}")
    
    if xa.shape[0] != n:
        raise ValueError(f"xa dimension {xa.shape[0]} does not match H columns {n}")
    
    if R.shape != (m, m):
        raise ValueError(f"R must be ({m}, {m}), got {R.shape}")
    
    if P.shape != (n, n):
        raise ValueError(f"P must be ({n}, {n}), got {P.shape}")
    
    # Check if R and P are square and symmetric
    if not np.allclose(R, R.T):
        raise ValueError("R must be symmetric")
    
    if not np.allclose(P, P.T):
        raise ValueError("P must be symmetric")
    
    # Check if R and P are positive definite
    try:
        eigvals_R = np.linalg.eigvalsh(R)
        if np.any(eigvals_R <= 0):
            raise ValueError(f"R is not positive definite. Minimum eigenvalue: {eigvals_R.min()}")
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Failed to compute eigenvalues of R: {str(e)}")
    
    try:
        eigvals_P = np.linalg.eigvalsh(P)
        if np.any(eigvals_P <= 0):
            raise ValueError(f"P is not positive definite. Minimum eigenvalue: {eigvals_P.min()}")
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Failed to compute eigenvalues of P: {str(e)}")
    
    # Perform inversion with error handling
    try:
        # Compute H@P@H.T + R
        HPHt_R = H @ P @ H.T + R
        
        # Check condition number for numerical stability
        cond_HPHt_R = np.linalg.cond(HPHt_R)
        if cond_HPHt_R > 1e12:
            print(f"Warning: H@P@H.T+R is ill-conditioned (condition number: {cond_HPHt_R:.2e})")
        
        # Gain matrix
        inv_HPHt_R = np.linalg.inv(HPHt_R)
        G = P @ H.T @ inv_HPHt_R
        
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(f"Failed to compute gain matrix: {str(e)}")
    
    # Posterior estimates
    try:
        xhat = xa + G @ (y - H @ xa)
    except Exception as e:
        raise RuntimeError(f"Failed to compute posterior estimates: {str(e)}")
    
    # Posterior error covariance
    try:
        # Check condition number of R
        cond_R = np.linalg.cond(R)
        if cond_R > 1e12:
            print(f"Warning: R is ill-conditioned (condition number: {cond_R:.2e})")
        
        # Check condition number of P
        cond_P = np.linalg.cond(P)
        if cond_P > 1e12:
            print(f"Warning: P is ill-conditioned (condition number: {cond_P:.2e})")
        
        inv_R = np.linalg.inv(R)
        inv_P = np.linalg.inv(P)
        shat = H.T @ inv_R @ H + inv_P
        shat = np.linalg.inv(shat)
        
        # Verify shat is positive definite
        eigvals_shat = np.linalg.eigvalsh(shat)
        if np.any(eigvals_shat <= 0):
            print(f"Warning: Posterior covariance has non-positive eigenvalues. Min: {eigvals_shat.min():.2e}")
        
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(f"Failed to compute posterior error covariance: {str(e)}")
    
    # Averaging kernel
    try:
        ak = G @ H
        
        # Check averaging kernel properties
        trace_ak = np.trace(ak)
        dofs = trace_ak  # Degrees of freedom for signal
        if dofs < 0 or dofs > n:
            print(f"Warning: Unusual DOFS value: {dofs:.2f} (expected 0 to {n})")
        
    except Exception as e:
        raise RuntimeError(f"Failed to compute averaging kernel: {str(e)}")
    
    return xhat, ak, shat

def hbmcmc_inversion(H, y, R_prior, x_prior_builder, 
                     n_samples=1e5, 
                     n_tune=4e4, 
                     n_chains=4,
                     #samplers = {"NUTS":["x_prior"], "Slice":["R_prior"]},
                     target_accept=0.9,
                     return_trace=False,
                     random_seed=None,
                     use_mvnormal_if_matrix=True):


    """
    Hierarchical Bayesian MCMC inversion for linear forward models:

        y = H x + ε

    where
        x ~ p(x)        prior
        ε ~ N(0, R)     observation error

    This function is designed to be *model-agnostic*:
    - The prior for x is fully defined by `x_prior_builder`
    - The observation error structure R can be either fixed or inferred

    Parameters
    ----------
    H : array-like, shape (m, n)
        Forward model / sensitivity (Jacobian) matrix.

    y : array-like, shape (m,)
        Observations.

    x_prior_builder : callable
        Function that constructs the prior for x **inside** a PyMC model context.
        Must return a PyMC random variable with shape (n,).

        Examples
        --------
        Independent prior:
            def x_prior_builder():
                return pm.Normal("x_prior", mu=0, sigma=1, shape=n)

        Correlated prior:
            def x_prior_builder():
                return pm.MvNormal("x_prior", mu=np.zeros(n), cov=B, shape=n)

    R_prior : callable or array-like
        Observation error specification.

        If callable:
            Must return a PyMC RV representing either:
              - sigma (scalar or vector, independent errors), or
              - covariance matrix (m,m), correlated errors
            Example sigma: 
            def R_prior_builder(): 
                return pm.HalfNormal("R_prior", sigma=1.0, shape=m) 
            Example covariance (advanced): 
            def R_prior_builder(): 
                sd = pm.HalfNormal("sd", 1.0, shape=m) 
                L, corr, sigmas = pm.LKJCholeskyCov("R_prior_chol", n=m, eta=2, sd_dist=pm.HalfNormal.dist(1.0), compute_corr=True) 
                # Return covariance matrix: 
                return pm.math.dot(L, L.T)
        If float or 1-d array:
            Must contain data of variances/co-variances

        If array-like (constant):
            * scalar        -> homoscedastic independent errors
            * shape (m,)    -> heteroscedastic independent errors
            * shape (m,m)   -> fixed correlated covariance matrix

    n_samples : int
        Number of posterior samples (draws).

    n_tune : int
        Number of tuning (warmup) steps.

    n_chains : int
        Number of MCMC chains.

    samplers : dict
        Mapping of sampler name -> list of variable names.
        Example:
            {"NUTS": ["x_prior"], "Slice": ["R_prior"]}

        Variables corresponding to constants are automatically skipped.

    target_accept : float
        Target acceptance probability for NUTS.

    return_trace : bool
        If True, also return the PyMC model and variable handles.

    random_seed : int or None
        Random seed for reproducibility.

    use_mvnormal_if_matrix : bool
        If True, automatically use MvNormal likelihood when R is a matrix.

    Returns
    -------
    idata : arviz.InferenceData
        Posterior samples.

    (optional)
    model : pm.Model
        The PyMC model object.

    rv_dict : dict
        Mapping of variable names to PyMC RVs.
    """ 
    
    
    # ------------------------------------------------------------------
    # 0) Input validation and basic consistency checks
    # ------------------------------------------------------------------
    H = np.asarray(H)
    if H.ndim != 2:
        raise ValueError(f"H must be 2D (m,n), got ndim={H.ndim}")

    m, n = H.shape
    if m <= 0 or n <= 0:
        raise ValueError(f"H has invalid shape {H.shape}")

    y = np.asarray(y).ravel()
    if y.shape[0] != m:
        raise ValueError(f"y length {y.shape[0]} does not match H rows {m}")

    if not np.isfinite(H).all():
        raise ValueError("H contains NaN or Inf")
    if not np.isfinite(y).all():
        raise ValueError("y contains NaN or Inf")

    if not callable(x_prior_builder):
        raise TypeError("x_prior_builder must be callable")

    R_is_callable = callable(R_prior)


    n_samples = int(n_samples)
    n_tune = int(n_tune)
    n_chains = int(n_chains)

    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if n_tune < 0:
        raise ValueError("n_tune must be >= 0")
    if n_chains <= 0:
        raise ValueError("n_chains must be > 0")
    if not (0 < target_accept < 1):
        raise ValueError("target_accept must be in (0,1)")

    
    # ------------------------------------------------------------------
    # 1) Sampler registry
    # ------------------------------------------------------------------
    SAMPLER_REGISTRY = {
        "NUTS": pm.NUTS,
        "Slice": pm.Slice,
        # Metropolis / DEMetropolisZ can be added later if needed
    }
    samplers = {'NUTS':[],"Slice":[]}

    #if samplers is None:
        #samplers = {"NUTS": ["x_prior"]}

    #if not isinstance(samplers, dict):
        #raise TypeError("samplers must be a dict")
    

    # ------------------------------------------------------------------
    # 2) Model construction
    # ------------------------------------------------------------------
    with pm.Model() as model:

        # ---- Prior for x (possibly correlated) ----
        x_bundle = x_prior_builder()

        # Forward model: y_pred = H x
        try:
            y_pred = pm.math.dot(H, x_bundle.main)
        except Exception as e:
            raise TypeError(
                "Failed to compute y_pred = dot(H, x_prior). "
                "x_prior_builder() likely returned an incompatible object."
            ) from e

        # ------------------------------------------------------------------
        # 3) Observation error model R
        # ------------------------------------------------------------------
        if R_is_callable:
            # Hierarchical / stochastic R
            R_bundle = R_prior()
            R_obj = R_bundle.main
        else:
            # Fixed (deterministic) R
            R_arr = np.asarray(R_prior)
            #R_arr = np.square(R_arr)  
            
            if not np.isfinite(R_arr).all():
                raise ValueError("R_prior contains NaN or Inf")

            if R_arr.ndim == 0:
                # Scalar sigma
                if R_arr <= 0:
                    raise ValueError("Scalar sigma must be > 0")
                R_obj = float(np.sqrt(R_arr))
                print("Be careful!! R_prior is sqrted to be used as sigma in the likelihood.")
                
            elif R_arr.ndim == 1:
                # Independent heteroscedastic errors
                if R_arr.shape[0] != m:
                    raise ValueError("Sigma vector must have length m")
                if np.any(R_arr <= 0):
                    raise ValueError("All sigma values must be > 0")
                R_obj = np.sqrt(R_arr)
                print("Be careful!! R_prior is sqrted to be used as sigma in the likelihood.")
                
            elif R_arr.ndim == 2:
                # Full covariance matrix
                if R_arr.shape != (m, m):
                    raise ValueError("Covariance matrix must be shape (m,m)")
                if not np.allclose(R_arr, R_arr.T, atol=0, rtol=1e-7):
                    raise ValueError("Covariance matrix must be symmetric")
                R_obj = R_arr

            else:
                raise ValueError("R_prior must be scalar, (m,), or (m,m)")

        # ------------------------------------------------------------------
        # 4) Likelihood
        # ------------------------------------------------------------------
        use_mv = False
        if use_mvnormal_if_matrix:
            if isinstance(R_obj, np.ndarray) and R_obj.ndim == 2:
                use_mv = True
            elif R_is_callable and R_obj.ndim == 2:
                use_mv = True
            '''
            if isinstance(R_obj, float):
                use_mv = False
            elif isinstance(R_obj, np.ndarray) and R_obj.ndim == 1:
                use_mv = False
            elif R_is_callable and R_obj.ndim == 1:
                use_mv = False
            '''
        

        if use_mv:
            print(f"using multi-variable normal for the likelihood")
            pm.MvNormal(
                "y_obs",
                mu=y_pred,
                cov=R_obj,
                observed=y,
                shape=m,
            )
        else:
            print(f"using normal for the likelihood")
            pm.Normal(
                "y_obs",
                mu=y_pred,
                sigma=R_obj,
                observed=y,
                shape=m,
            )

        # ------------------------------------------------------------------
        # 5) Step methods
        # ------------------------------------------------------------------

        # merge sampler option
        
        samplers['NUTS']  += x_bundle.rvs.get("NUTS", [])
        samplers['Slice'] += x_bundle.rvs.get("Slice", [])
        if R_is_callable:
            samplers['NUTS']  += R_bundle.rvs.get("NUTS", [])
            samplers['Slice'] += R_bundle.rvs.get("Slice", [])
            
        steps = []
        for sampler_name, vars in samplers.items():
            if not vars:
                continue

            StepClass = SAMPLER_REGISTRY[sampler_name]



            if sampler_name == "NUTS":
                steps.append(StepClass(vars=vars,
                                       target_accept=target_accept))
            else:
                steps.append(StepClass(vars=vars))

        if not steps:
            raise ValueError("No valid step methods were created.")

        # ------------------------------------------------------------------
        # 6) Sampling
        # ------------------------------------------------------------------
        idata = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            step=steps,
            random_seed=random_seed,
            cores=n_chains,
            return_inferencedata=True,
            progressbar=True,
        )

    # ------------------------------------------------------------------
    # 7) Return
    # ------------------------------------------------------------------
    if return_trace:
        return idata, model
    else:
        return idata


