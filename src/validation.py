import torch
import numpy as np
import matplotlib.pyplot as plt
from src.amortization import to_log_probability_density

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


