import torch
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scores.probability import crps_cdf
from probability import to_log_probability_density
def bayesian_information_criterion(model, X_validate, Y_validate, n):
    """   
    Compute the Bayesian information criterion for GMM. Models with lower BIC are generally preferred. 
    The idea is that adding parameters increases the likelihood, but also increases the complexity of
    and thus leads to overfitting. 

    BIC = k * ln(n) - 2 * ln(hat{L})
    where \hat{L} is the max value of the likelihood function of model M. Here it is just 
    the conditional probability on validation set using the fitted GMM model after EM algorithm.
          n is the number of the data points in x / number of obs / number of samples
          k is the number of parameters in the model M
    """
    
    
    # total number of params: 2 * number of GMM components (for mean and covariance) + number of GMM components (for weights)
    n_component = model.gmm.n_components
    n_dim = model.ndim_reduced_total
    k = n_component * (n_dim + n_dim*(n_dim+1)/2 + 1) -1

    L = prediction_error(model, X_validate, Y_validate)
    kln = k * np.log(n)
    neg2L = -2 * L
    BIC = kln + neg2L
    print("k*ln(n):", kln, " - 2*L:", neg2L, "BIC:", BIC)
    return BIC, kln, neg2L

def ensemble_crps(
    posterior_samples,
    ground_truth,
    domain_mask,
    n_thresholds=256,
    chunk_size=512,
):
    """
    Compute pointwise CRPS using scores.probability.crps_cdf for an ensemble of samples.
    A chunked evaluation is performed to handle large datasets efficiently.

    Parameters
    ----------
    posterior_samples : array, shape (n_draws, ny, nx)
        Posterior enthalpy samples in physical units.
    ground_truth : array, shape (ny, nx)
        Ground-truth enthalpy in the same units.
    domain_mask : boolean array, shape (ny, nx)
        Cells included in evaluation.

    Returns
    -------
    crps_map : array, shape (ny, nx)
        CRPS inside the domain; NaN outside.
    mean_crps : float
        Domain-averaged CRPS.
    """
    samples = np.asarray(posterior_samples)
    truth = np.asarray(ground_truth)
    mask = np.asarray(domain_mask, dtype=bool)

    if samples.ndim != 3 or samples.shape[1:] != truth.shape:
        raise ValueError("Expected samples with shape (n_draws, ny, nx).")
    if mask.shape != truth.shape:
        raise ValueError("domain_mask must match ground_truth.")
    if samples.shape[0] == 0 or n_thresholds < 2:
        raise ValueError("Need posterior draws and at least two thresholds.")

    indices = np.flatnonzero(mask.ravel())
    if indices.size == 0:
        raise ValueError("The evaluation domain is empty.")

    draws = samples.reshape(samples.shape[0], -1)
    observations = truth.ravel()

    # Check only cells being evaluated.
    if not np.isfinite(observations[indices]).all():
        raise ValueError("Ground truth contains nonfinite domain values.")

    lower = observations[indices].min()
    upper = observations[indices].max()

    for start in range(0, indices.size, chunk_size):
        cells = indices[start:start + chunk_size]
        values = draws[:, cells]

        if not np.isfinite(values).all():
            raise ValueError("Posterior samples contain nonfinite domain values.")

        lower = min(lower, values.min())
        upper = max(upper, values.max())

    result = np.full(truth.size, np.nan, dtype=float)

    # All predictions and observations coincide: CRPS is zero.
    if lower == upper:
        result[indices] = 0.0
        return result.reshape(truth.shape), 0.0

    thresholds = np.linspace(lower, upper, n_thresholds)

    for start in range(0, indices.size, chunk_size):
        cells = indices[start:start + chunk_size]
        values = draws[:, cells]

        cdf = np.empty((cells.size, n_thresholds), dtype=float)

        for k, threshold in enumerate(thresholds):
            cdf[:, k] = (values <= threshold).mean(axis=0)

        forecast = xr.DataArray(
            cdf,
            dims=("cell", "threshold"),
            coords={"cell": cells, "threshold": thresholds},
        )
        observed = xr.DataArray(
            observations[cells],
            dims=("cell",),
            coords={"cell": cells},
        )

        score = crps_cdf(
            forecast,
            observed,
            threshold_dim="threshold",
            preserve_dims=["cell"],
        )

        result[cells] = score["total"].transpose("cell").values


    crps_map = result.reshape(truth.shape)
    return crps_map, float(result[indices].mean())


def prediction_error(model, X, Y):
    """
    Compute the mean conditional log-likelihood: mean over i of log p(x_i | y_i)
    This is the correct quantity for BIC: ln(hat{L}) = sum_i log p(x_i | y_i)
    """
    n_samples = X.shape[0]
    log_prob = np.zeros(n_samples)

    for i in range(n_samples):
        xy = np.hstack((Y[i], X[i]))
        # Evaluate log p(x_i | y_i) at the OBSERVED x_i
        log_prob[i] = to_log_probability_density(model.gmm, xy)

    # Return TOTAL log-likelihood (not mean), so it scales with n
    total_log_likelihood = np.sum(log_prob)
    return total_log_likelihood

def lcurve_analysis(opt_func, beta_prior_list):
    """
    Perform L-curve analysis 
    """
    solution_norm_list = []
    residual_norm_list = []

    for beta_prior in beta_prior_list:
        z_optimal, _, residual, _ = opt_func(beta=beta_prior)
        solution_norm_list.append(np.linalg.norm(z_optimal, ord=2))
        residual_norm_list.append(np.linalg.norm(residual, ord=2))

    # Convert to arrays
    sol = np.array(solution_norm_list)
    res = np.array(residual_norm_list)

    # --- Normalize to [0, 1] ---
    res_norm = (res - res.min()) / (res.max() - res.min())
    sol_norm = (sol - sol.min()) / (sol.max() - sol.min())

    plt.figure(figsize=(8, 6))
    plt.plot(res_norm, sol_norm, marker='o')

    for i, beta in enumerate(beta_prior_list):
        plt.annotate(
            f'{beta:.2e}',
            (res_norm[i], sol_norm[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )

    plt.xlabel('Normalized Residual Norm ||Ax - b||')
    plt.ylabel('Normalized Solution Norm ||x||')
    plt.title('L-curve Analysis (Normalized)')
    plt.grid()
    plt.show()
    return 

def split_evidence(evidence, k=20):
    """
    Splitting the basal evidence into k non-overlapping subsets for cross-validation.
    """
    # shuffle evidence
    np.random.seed(42)
    np.random.shuffle(evidence)
    subsets = np.array_split(evidence, k)
    return subsets


