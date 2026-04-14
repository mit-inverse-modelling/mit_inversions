import numpy as np
import warnings
from scipy.stats import truncnorm
import pymc as pm
import arviz as az

    

def analytical_inversion(H, y, R, xa, P):
    '''
    Define analytical inversion function
    
    Parameters:
    -----------
    H: Sensitivity matrix (Jacobian matrix) - shape (m, n)
    y: Observations - shape (m,) or (m, 1)
    R: Observation error covariance matrix - shape (m, m),
       or the diagonal variances with shape (m,)
    xa: Prior estimates - shape (n,) or (n, 1)
    P: Prior error covariance matrix - shape (n, n),
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
    
    # Interpret R either as a full covariance matrix or as diagonal variances.
    if R.ndim == 1:
        if R.shape[0] != m:
            raise ValueError(f"Diagonal R must have length {m}, got {R.shape[0]}")
        warnings.warn(
            "R was provided as a 1D array; interpreting it as diagonal variances.",
            stacklevel=2,
        )
        r_diag = R
        R_full = None
    elif R.ndim == 2:
        if R.shape != (m, m):
            raise ValueError(f"R must be ({m}, {m}), got {R.shape}")
        if not np.allclose(R, R.T):
            raise ValueError("R must be symmetric")
        diag_R = np.diag(R)
        if np.allclose(R, np.diag(diag_R)):
            r_diag = diag_R
            R_full = None
        else:
            r_diag = None
            R_full = R
    else:
        raise ValueError(f"R must be 1D or 2D, got {R.ndim} dimensions")

    # Interpret P either as a full covariance matrix or as diagonal variances.
    if P.ndim == 1:
        if P.shape[0] != n:
            raise ValueError(f"Diagonal P must have length {n}, got {P.shape[0]}")
        warnings.warn(
            "P was provided as a 1D array; interpreting it as diagonal variances.",
            stacklevel=2,
        )
        p_diag = P
        P = np.diag(P)
    elif P.ndim == 2:
        if P.shape != (n, n):
            raise ValueError(f"P must be ({n}, {n}), got {P.shape}")
        p_diag = np.diag(P) if np.allclose(P, np.diag(np.diag(P))) else None
    else:
        raise ValueError(f"P must be 1D or 2D, got {P.ndim} dimensions")

    # Check if P is square and symmetric
    if not np.allclose(P, P.T):
        raise ValueError("P must be symmetric")
    
    # Check if R and P are positive definite
    if r_diag is not None:
        if np.any(r_diag <= 0):
            raise ValueError(f"R is not positive definite. Minimum diagonal entry: {r_diag.min()}")
    else:
        try:
            np.linalg.cholesky(R_full)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"R is not positive definite: {str(e)}")
    
    try:
        np.linalg.cholesky(P)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"P is not positive definite: {str(e)}")

    resid = y - H @ xa

    # Choose the cheaper solve direction based on matrix structure and size.
    use_state_space = (n <= m)
    try:
        if p_diag is not None:
            if np.any(p_diag <= 0):
                raise ValueError(f"P is not positive definite. Minimum diagonal entry: {p_diag.min()}")
            inv_P = np.diag(1.0 / p_diag)
        else:
            inv_P = np.linalg.inv(P)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(f"Failed to invert P: {str(e)}")

    try:
        if r_diag is not None and use_state_space:
            inv_r = 1.0 / r_diag
            weighted_H = H * inv_r[:, None]
            system = H.T @ weighted_H + inv_P
            shat = np.linalg.inv(system)
            xhat = xa + shat @ (H.T @ (inv_r[:, None] * resid))
            ak = shat @ (H.T @ weighted_H)
        elif r_diag is not None:
            PHt = P @ H.T
            HPHt_R = H @ PHt
            HPHt_R[np.diag_indices_from(HPHt_R)] += r_diag
            G = PHt @ np.linalg.inv(HPHt_R)
            xhat = xa + G @ resid
            weighted_H = H * (1.0 / r_diag)[:, None]
            shat = np.linalg.inv(H.T @ weighted_H + inv_P)
            ak = G @ H
        elif use_state_space:
            solve_R_H = np.linalg.solve(R_full, H)
            system = H.T @ solve_R_H + inv_P
            shat = np.linalg.inv(system)
            xhat = xa + shat @ (H.T @ np.linalg.solve(R_full, resid))
            ak = shat @ (H.T @ solve_R_H)
        else:
            PHt = P @ H.T
            HPHt_R = H @ PHt + R_full
            G = PHt @ np.linalg.inv(HPHt_R)
            xhat = xa + G @ resid
            shat = np.linalg.inv(H.T @ np.linalg.solve(R_full, H) + inv_P)
            ak = G @ H
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


def gen_ensemble(H, xb_bar, Pin, N, dist_type='gaussian', dist_params=None):
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
    
    nm = H.shape[0]
    nx = H.shape[1]
    yb_bar = H @ xb_bar
    
    sigma = np.expand_dims(np.diag(Pin)**0.5, axis=1)
    mean = np.expand_dims(xb_bar, 1)
    
    # Sample state ensemble
    if dist_type == 'gaussian':
        xb = np.random.normal(loc=mean, scale=sigma, size=(nx, N))
    
    elif dist_type == 'multivariate_normal':
        xb = np.random.multivariate_normal(mean=xb_bar, cov=Pin, size=N).T
    
    elif dist_type == 'lognormal':
        mu = np.log(mean) - 0.5 * np.log(1 + (sigma / mean)**2)
        sig = np.sqrt(np.log(1 + (sigma / mean)**2))
        xb = np.random.lognormal(mean=mu, sigma=sig, size=(nx, N))
    
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
                                     size=N)
    
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
        
        xb = np.random.uniform(low=low, high=high, size=(nx, N))
    
    else:
        raise ValueError(f"Unknown distribution: {dist_type}")
    
    # Compute observations and perturbations
    Yb = (H @ xb) - np.expand_dims(yb_bar, 1)
    Xb = xb - np.expand_dims(xb_bar, 1)
    
    return Xb, yb_bar, Yb

def ETKF_step(Xb, Yb, R,y,xb_bar,yb_bar):
    """An Ensemble Transform Kalman Filter"""
    N = Yb.shape[1]
    I = np.identity(N)
    inv_R = np.linalg.inv(R)
    Pas = np.linalg.inv(Yb.T @ inv_R @ Yb + (N-1)*I)
    xa_bar = xb_bar + Xb @ Pas @ Yb.T @ inv_R @ (y-yb_bar)
    Xa = Xb @ np.linalg.cholesky((N-1)*Pas)
    Pa = Xa@Xa.T / (N-1)
    return xa_bar, Pa

def ETKF_inversion(H, y, R, xb_bar, Pin, N=1000, dist_type='gaussian', dist_params=None):
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
    
    Xb, yb_bar, Yb = gen_ensemble(H, xb_bar, Pin, N, dist_type=dist_type, dist_params=dist_params)
    
    xa_bar, Pa = ETKF_step(Xb, Yb, R,y,xb_bar,yb_bar)
    
    return xa_bar, Pa

def hbmcmc_inversion(H, y, R_prior, x_prior_builder, 
                     n_samples=1e5, 
                     n_tune=4e4, 
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
