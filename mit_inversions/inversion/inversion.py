# inversion.py
# Created: 10 April 2026
# Authors: Eric Saboya, Minde An, Luke Western
# Copyright (c) 2026. All rights reserved.
# License: MIT License
# 
# Description:
#  This module contains the different inverse methods that can be used in ARTEMIS

import warnings
import numpy as np
import pymc as pm
import arviz as az
from scipy.stats import truncnorm
    

def analytical_inversion(K, y, So, xa, Sa):
    '''
    Define analytical inversion function using the notation from
    Jacobs et al
    
    Parameters:
    -----------
    K: Sensitivity matrix (Jacobian matrix) - shape (m, n)
    y: Observations - shape (m,) or (m, 1)
    So: Observation error covariance matrix - shape (m, m),
       or the diagonal variances with shape (m,)
    xa: Prior estimates - shape (n,) or (n, 1)
    Sa: Prior error covariance matrix - shape (n, n),
       or the diagonal variances with shape (n,)
    
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
        K = np.asarray(K, dtype=float)
        y = np.asarray(y, dtype=float)
        So = np.asarray(So, dtype=float)
        xa = np.asarray(xa, dtype=float)
        Sa = np.asarray(Sa, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Failed to convert inputs to numpy arrays: {str(e)}")
    
    # Check for NaN or Inf values
    inputs = {'K': K, 'y': y, 'So': So, 'xa': xa, 'Sa': Sa}
    for name, value in inputs.items():
        if np.any(np.isnan(value)):
            raise ValueError(f"{name} contains NaN values")
        if np.any(np.isinf(value)):
            raise ValueError(f"{name} contains Inf values")
    
    # Ensure proper dimensions
    if K.ndim != 2:
        raise ValueError(f"K must be 2-dimensional, got {K.ndim} dimensions")
    
    m, n = K.shape
    
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
        raise ValueError(f"y dimension {y.shape[0]} does not match K rows {m}")
    
    if xa.shape[0] != n:
        raise ValueError(f"xa dimension {xa.shape[0]} does not match K columns {n}")
    
    # Interpret So either as a full covariance matrix or as diagonal variances.
    if So.ndim == 1:
        if So.shape[0] != m:
            raise ValueError(f"Diagonal So must have length {m}, got {So.shape[0]}")
        warnings.warn(
            "So was provided as a 1D array; interpreting it as diagonal variances.",
            stacklevel=2,
        )
        so_diag = So
        So_full = None
    elif So.ndim == 2:
        if So.shape != (m, m):
            raise ValueError(f"So must be ({m}, {m}), got {So.shape}")
        if not np.allclose(So, So.T):
            raise ValueError("So must be symmetric")
        diag_So = np.diag(So)
        if np.allclose(So, np.diag(diag_So)):
            so_diag = diag_So
            So_full = None
        else:
            so_diag = None
            So_full = So
    else:
        raise ValueError(f"So must be 1D or 2D, got {So.ndim} dimensions")

    # Interpret Sa either as a full covariance matrix or as diagonal variances.
    if Sa.ndim == 1:
        if Sa.shape[0] != n:
            raise ValueError(f"Diagonal Sa must have length {n}, got {Sa.shape[0]}")
        warnings.warn(
            "Sa was provided as a 1D array; interpreting it as diagonal variances.",
            stacklevel=2,
        )
        sa_diag = Sa
        Sa_full = None
    elif Sa.ndim == 2:
        if Sa.shape != (n, n):
            raise ValueError(f"Sa must be ({n}, {n}), got {Sa.shape}")
        if not np.allclose(Sa, Sa.T):
            raise ValueError("Sa must be symmetric")
        diag_Sa = np.diag(Sa)
        if np.allclose(Sa, np.diag(diag_Sa)):
            sa_diag = diag_Sa
            Sa_full = None
        else:
            sa_diag = None
            Sa_full = Sa
    else:
        raise ValueError(f"Sa must be 1D or 2D, got {Sa.ndim} dimensions")

    # Interpret Sa either as a full covariance matrix or as diagonal variances.
    if Sa.ndim == 1:
        if Sa.shape[0] != n:
            raise ValueError(f"Diagonal Sa must have length {n}, got {Sa.shape[0]}")
        warnings.warn(
            "Sa was provided as a 1D array; interpreting it as diagonal variances.",
            stacklevel=2,
        )
        sa_diag = Sa
        Sa = np.diag(Sa)
    elif Sa.ndim == 2:
        if Sa.shape != (n, n):
            raise ValueError(f"Sa must be ({n}, {n}), got {Sa.shape}")
        sa_diag = np.diag(Sa) if np.allclose(Sa, np.diag(np.diag(Sa))) else None
    else:
        raise ValueError(f"Sa must be 1D or 2D, got {Sa.ndim} dimensions")

    # Check if Sa is square and symmetric
    if not np.allclose(Sa, Sa.T):
        raise ValueError("Sa must be symmetric")
    
    # Check if So and Sa are positive definite
    if so_diag is not None:
        if np.any(so_diag <= 0):
            raise ValueError(f"So is not positive definite. Minimum diagonal entry: {so_diag.min()}")
    else:
        try:
            np.linalg.cholesky(So_full)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"So is not positive definite: {str(e)}")
    
    try:
        np.linalg.cholesky(Sa)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Sa is not positive definite: {str(e)}")

    resid = y - K @ xa

    # Choose the cheaper solve direction based on matrix structure and size.
    use_state_space = (n <= m)
    try:
        if sa_diag is not None:
            if np.any(sa_diag < 0):
                raise ValueError(f"Sa is not positive definite. Minimum diagonal entry: {sa_diag.min()}")
            inv_sa = np.nan_to_num(np.diag(1.0 / sa_diag))
        else:
            inv_sa = np.nan_to_num(np.linalg.inv(Sa))
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(f"Failed to invert Sa: {str(e)}")

    try:
        if so_diag is not None and use_state_space:
            inv_so = 1.0 / so_diag
            weighted_K = K * inv_so[:, None]
            system = K.T @ weighted_K + inv_sa
            shat = np.linalg.inv(system)
            xhat = xa + shat @ (K.T @ (inv_so[:, None] * resid))
            ak = shat @ (K.T @ weighted_K)

        elif so_diag is not None:
            SaKt = Sa @ K.T
            KSaKt_So = K @ SaKt + So_full
            G = SaKt @ np.linalg.inv(KSaKt_So)
            xhat = xa + G @ resid
            weighted_K = K * (1.0 / so_diag)[:, None]
            shat = np.linalg.inv(K.T @ weighted_K + inv_sa)
            ak = G @ K

        elif use_state_space:
            solve_R_H = np.linalg.solve(So_full, K)
            system = K.T @ solve_R_H + inv_sa
            shat = np.linalg.inv(system)
            xhat = xa + shat @ (K.T @ np.linalg.solve(So_full, resid))
            ak = shat @ (K.T @ solve_R_H)

        else:
            SaKt = Sa @ K.T
            KSaKt_So = K @ SaKt + So_full
            G = SaKt @ np.linalg.inv(KSaKt_So)
            xhat = xa + G @ resid
            shat = np.linalg.inv(K.T @ np.linalg.solve(So_full, K) + inv_sa)
            ak = G @ K

    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(f"Failed to solve analytical inversion system: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to compute analytical inversion: {str(e)}")

    # Check posterior covariance and averaging kernel properties
    eigvals_shat = np.linalg.eigvalsh(shat)
    if np.any(eigvals_shat <= 0):
        print(f"Warning: Posterior covariance has non-positive eigenvalues. Min: {eigvals_shat.min():.2e}")

    trace_ak = np.trace(ak)
    dofs = trace_ak
    if dofs < 0 or dofs > n:
        print(f"Warning: Unusual DOFS value: {dofs:.2f} (expected 0 to {n})")
    
    return xhat, ak, shat


def gen_ensemble(H, xb_bar, Pin, N, dist_type='gaussian', dist_params=None, random_seed=None):
    """Generate an ensemble of state vectors and their corresponding observations.
       Intended for input to ETKF.
       For the lognormal, input is the mean and std of the distribution (which 
       will be converted to the underlying log-normal parameters)

    Parameters:
    -----------
    dist_type : str
        Distribution to sample from: 'gaussian', 'multivariate_normal', 'lognormal', 
        'truncated_normal', 'uniform'
    dist_params : dict
        Optional parameters:
        - truncated_normal: {'lower_bound': 0, 'upper_bound': np.inf}
        - uniform: {'bounds': (lower, upper)} or use default ±√3·σ
    """
    if dist_params is None:
        dist_params = {}

    rng = np.random.default_rng(random_seed)
    
    nx = H.shape[1]
    
    sigma = np.expand_dims(np.diag(Pin)**0.5, axis=1)
    mean = np.expand_dims(xb_bar, 1)
    
    # Sample state ensemble
    if dist_type == 'gaussian':
        # gaussian mode samples independent components from diag(Pin).
        # If Pin has off-diagonal structure, prefer multivariate_normal.
        if not np.allclose(Pin, np.diag(np.diag(Pin))):
            warnings.warn(
                "dist_type='gaussian' ignores off-diagonal prior covariance; "
                "falling back to 'multivariate_normal'.",
                stacklevel=2,
            )
            xb = rng.multivariate_normal(mean=xb_bar, cov=Pin, size=N).T
        else:
            xb = rng.normal(loc=mean, scale=sigma, size=(nx, N))
    
    elif dist_type == 'multivariate_normal':
        xb = rng.multivariate_normal(mean=xb_bar, cov=Pin, size=N).T
    
    elif dist_type == 'lognormal':
        mu = np.log(mean) - 0.5 * np.log(1 + (sigma / mean)**2)
        sig = np.sqrt(np.log(1 + (sigma / mean)**2))
        xb = rng.lognormal(mean=mu, sigma=sig, size=(nx, N))
    
    elif dist_type == 'truncated_normal':
        # Default bounds: [0, inf]
        lower_bound = dist_params.get('lower_bound', 0)
        upper_bound = dist_params.get('upper_bound', np.inf)
        
        xb = np.zeros((nx, N))
        for i in range(nx):
            a = (lower_bound - mean[i, 0]) / sigma[i, 0]
            b = (upper_bound - mean[i, 0]) / sigma[i, 0]
            xb[i, :] = truncnorm.rvs(a=a, b=b,
                                     loc=mean[i, 0],
                                     scale=sigma[i, 0],
                                     size=N,
                                     random_state=rng)
    
    elif dist_type == 'uniform':
        # Check if custom bounds provided, otherwise use ±√3·σ
        if 'bounds' in dist_params:
            bounds = dist_params['bounds']
            low = np.full_like(mean, bounds[0])
            high = np.full_like(mean, bounds[1])
        else:
            width = np.sqrt(3) * sigma
            low = mean - width
            high = mean + width
        
        xb = rng.uniform(low=low, high=high, size=(nx, N))
    
    else:
        raise ValueError(f"Unknown distribution: {dist_type}")
    
    # Enforce consistency with the prescribed prior mean for finite ensembles.
    xb = xb - np.mean(xb, axis=1, keepdims=True) + mean

    # Compute perturbations and simulated-observation anomalies around xb_bar.
    yb_bar = H @ xb_bar
    Yb = (H @ xb) - np.expand_dims(yb_bar, 1)
    Xb = xb - np.expand_dims(xb_bar, 1)

    return Xb, yb_bar, Yb

def ETKF_step(Xb, Yb, R,y,xb_bar,yb_bar):
    """An Ensemble Transform Kalman Filter"""
    N = Yb.shape[1]
    I = np.identity(N)
    # Ensemble-space ETKF solve with symmetric eigendecomposition for stability.
    Rinv_Yb = np.linalg.solve(R, Yb)
    system = Yb.T @ Rinv_Yb + (N - 1) * I
    system = 0.5 * (system + system.T)

    innovation = y - yb_bar
    rhs = Yb.T @ np.linalg.solve(R, innovation)
    w = np.linalg.solve(system, rhs)
    xa_bar = xb_bar + Xb @ w

    evals, evecs = np.linalg.eigh(system)
    eps = 1e-12 * np.max(np.abs(evals)) if evals.size else 1e-12
    evals = np.clip(evals, eps, None)
    sqrt_transform = np.sqrt(N - 1.0) * (evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T)

    Xa = Xb @ sqrt_transform
    Pa = Xa@Xa.T / (N-1)
    return xa_bar, Pa

def ETKF_inversion(H, y, R, xb_bar, Pin, N=1000, dist_type='gaussian', dist_params=None, random_seed=None):
    """Perform ETKF inversion for linear forward model y = Hx + ε, where ε ~ N(0,R).
       The prior is represented by an ensemble of state vectors generated from 
       the specified distribution.

    Parameters:
    -----------
    H: Sensitivity matrix (Jacobian) - shape (m, n)
    y: Observations - shape (m,) or (m, 1)
    R: Observation error covariance matrix - shape (m, m) or diagonal variances (m,)
    xb_bar: Prior mean state vector - shape (n,) or (n, 1)
    Pin: Prior error covariance matrix - shape (n, n) or diagonal variances (n,)
    N: Ensemble size
    dist_type: Distribution type for generating the ensemble ('gaussian', 'multivariate_normal', 'lognormal', 'truncated_normal', 'uniform')
    dist_params: Additional parameters for the distribution (e.g., bounds for truncated_normal)

    Returns:
    --------
    xa_bar: Posterior mean state vector - shape (n,) or (n, 1)
    Pa: Posterior error covariance matrix - shape (n, n)
    
    Note:
    -----
    The ETKF is an approximate method that relies on the ensemble representation of the prior. 
    The quality of the results can depend on the choice of distribution and ensemble size.
    """
    
    Xb, yb_bar, Yb = gen_ensemble(
        H,
        xb_bar,
        Pin,
        N,
        dist_type=dist_type,
        dist_params=dist_params,
        random_seed=random_seed,
    )
    
    xa_bar, Pa = ETKF_step(Xb, Yb, R, y, xb_bar, yb_bar)
    
    return xa_bar, Pa

def hbmcmc_inversion(H_emis, y, R_prior, x_prior_builder,
                     H_bc=None,
                     bc_prior_builder=None,
                     n_samples=1e4, 
                     n_tune=1e4, 
                     n_chains=4,
                     #samplers = {"NUTS":["x_prior"], "Slice":["R_prior"]},
                     target_accept=0.9,
                     nuts_sampler="pymc",
                     nuts_sampler_kwargs=None,
                     progressbar=True,
                     cores=None,
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
    H_emis : array-like, shape (m, n)
        Emission sensitivity (Jacobian) matrix.

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

    nuts_sampler : str
        Which NUTS implementation to use. One of
        ``"pymc"``, ``"nutpie"``, ``"blackjax"``, or ``"numpyro"``.
        The default ``"pymc"`` preserves the existing behavior.

    nuts_sampler_kwargs : dict or None
        Extra keyword arguments forwarded to the external NUTS backend when
        ``nuts_sampler`` is not ``"pymc"``.

    progressbar : bool
        Whether to display the PyMC sampling progress bar.

    cores : int or None
        Number of worker processes used by ``pm.sample``. If None, defaults to
        ``n_chains``.

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
    H_emis = np.asarray(H_emis)
    if H_emis.ndim != 2:
        raise ValueError(f"H_emis must be 2D (m,n), got ndim={H_emis.ndim}")

    m, n = H_emis.shape
    if m <= 0 or n <= 0:
        raise ValueError(f"H_emis has invalid shape {H_emis.shape}")

    y = np.asarray(y).ravel()
    if y.shape[0] != m:
        raise ValueError(f"y length {y.shape[0]} does not match H rows {m}")

    if not np.isfinite(H_emis).all():
        raise ValueError("H_emis contains NaN or Inf")
    if not np.isfinite(y).all():
        raise ValueError("y contains NaN or Inf")

    if not callable(x_prior_builder):
        raise TypeError("x_prior_builder must be callable")

    if (H_bc is None) != (bc_prior_builder is None):
        raise ValueError("H_bc and bc_prior_builder must be provided together")

    if H_bc is not None:
        H_bc = np.asarray(H_bc)
        if H_bc.ndim != 2:
            raise ValueError(f"H_bc must be 2D (m,n_bc), got ndim={H_bc.ndim}")
        if H_bc.shape[0] != m:
            raise ValueError(f"H_bc rows {H_bc.shape[0]} do not match y length {m}")
        if not callable(bc_prior_builder):
            raise TypeError("bc_prior_builder must be callable")

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
    valid_nuts_samplers = {"pymc", "nutpie", "blackjax", "numpyro"}
    if nuts_sampler not in valid_nuts_samplers:
        raise ValueError(
            f"nuts_sampler must be one of {sorted(valid_nuts_samplers)}, got {nuts_sampler!r}"
        )
    if nuts_sampler_kwargs is None:
        nuts_sampler_kwargs = {}
    elif not isinstance(nuts_sampler_kwargs, dict):
        raise TypeError("nuts_sampler_kwargs must be a dict or None")
    if not isinstance(progressbar, bool):
        raise TypeError("progressbar must be a bool")
    if cores is None:
        cores = n_chains
    else:
        cores = int(cores)
        if cores <= 0:
            raise ValueError("cores must be > 0")

    
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

        # Forward model: y_pred = H_emis x + H_bc x_bc
        try:
            y_pred = pm.math.dot(H_emis, x_bundle.main)
        except Exception as e:
            raise TypeError(
                "Failed to compute y_pred = dot(H_emis, x_prior). "
                "x_prior_builder() likely returned an incompatible object."
            ) from e

        bc_bundle = None
        if bc_prior_builder is not None:
            bc_bundle = bc_prior_builder()
            try:
                y_pred = y_pred + pm.math.dot(H_bc, bc_bundle.main)
            except Exception as e:
                raise TypeError(
                    "Failed to compute y_pred = dot(H_emis, x_prior) + dot(H_bc, bc_prior). "
                    "bc_prior_builder() likely returned an incompatible object."
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
        if bc_bundle is not None:
            samplers['NUTS']  += bc_bundle.rvs.get("NUTS", [])
            samplers['Slice'] += bc_bundle.rvs.get("Slice", [])
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

        use_external_nuts = nuts_sampler != "pymc"
        if use_external_nuts and samplers["Slice"]:
            raise ValueError(
                "External nuts_sampler backends require all sampled variables "
                "to be handled by NUTS only; found Slice-assigned variables."
            )
        if use_external_nuts and not samplers["NUTS"]:
            raise ValueError(
                "External nuts_sampler backends require at least one NUTS-sampled variable."
            )

        # ------------------------------------------------------------------
        # 6) Sampling
        # ------------------------------------------------------------------
        print(
            f"Starting MCMC sampling: chains={n_chains}, draws={n_samples}, "
            f"tune={n_tune}, cores={cores}, progressbar={progressbar}, "
            f"nuts_sampler={nuts_sampler}"
        )
        sample_kwargs = dict(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            random_seed=random_seed,
            cores=cores,
            return_inferencedata=True,
            progressbar=progressbar,
        )

        if use_external_nuts:
            sample_kwargs["nuts_sampler"] = nuts_sampler
            sample_kwargs["nuts_sampler_kwargs"] = nuts_sampler_kwargs
            idata = pm.sample(**sample_kwargs)
        else:
            sample_kwargs["step"] = steps
            idata = pm.sample(**sample_kwargs)

    # ------------------------------------------------------------------
    # 7) Return
    # ------------------------------------------------------------------
    if return_trace:
        return idata, model
    else:
        return idata
