from tabnanny import verbose
import numpy as np
from gmr import MVN
from gmr.gmm import _safe_probability_density
from src.amortization import to_log_probability_density
from src.ice import enthalpy_to_temperature
# import pytorch for AD
import torch

# functions related to finding Maximum A Posteriori (MAP) in the latent space of a trained GMR model
def loglikelihood_thawed(beta, Tb, Tpmp, dw):
    """
    Compute the log likelihood 1 (the thawed base evidence) given the basal temperature and pressure melting point.

    L1 = (1/beta) * \Sum_i^N (Tb - Tpmp) * dw

    Parameters:
    -------
    beta: scalar
        temperature scale (K) for the exponential parameterization
    Tb: array
        basal temperature
    Tpmp: array
        pressure melting point 
    dw: array
        area of thawed base at each pixel

    Returns:
    -------
    L1: scalar
        log likelihood 1 value
    """
    # shape check first
    shape_check(Tb, Tpmp, dw)
    return -(1.0/beta) * torch.sum((Tpmp-Tb) * dw)

def loglikelihood_frozen(beta, Tb, Tpmp, df, eps = 0.01):
    """
    Compute the log likelihood 2 (the frozen base evidence) given the basal temperature and pressure melting point.

    L2 = (1/beta) * \Sum_i^N (Tpmp - Tb) * df

    Parameters:
    -------
    beta: scalar
        temperature scale (K) for the exponential parameterization
    Tb: array
        basal temperature
    Tpmp: array
        pressure melting point 
    df: array
        area of frozen base at each pixel

    Returns:
    -------
    L2: scalar
        log likelihood 2 value
    """
    shape_check(Tb, Tpmp, df)
    return torch.sum(torch.log(1 + (eps-1)*torch.exp(-(1/beta)*(Tpmp-Tb))) * df) 

def log_prior(Eb, gmm):
    """   
    Compute the log prior probability of the basal enthalpy field under the GMM model.

    Parameters:
    -------
    Eb: array
        basal enthalpy field (n_features,)
    gmm: GaussianMixture
        trained GMM model representing the prior distribution over basal enthalpy fields
    """

    return to_log_probability_density(gmm, Eb)

def log_prior_gradient(Eb, gmm):
    """   
    Compute the gradient of the log prior probability with respect to Eb.
    
    For GMM: p(x) = sum_k pi_k * N(x | mu_k, Sigma_k)
    Gradient: nabla log p(x) = [sum_k r_k * nabla log N_k(x)] 
    where r_k is the responsibility (posterior weight) of component k
    
    Parameters:
    -----------
    Eb: array, shape (n_features,)
        Input vector (e.g., basal enthalpy field)
    gmm: GMM object
        Trained GMM model
    
    Returns:
    --------
    grad: array, shape (n_features,)
        Gradient of log p(Eb) with respect to Eb
    """

    n_features = Eb.shape[0]
    n_components = gmm.n_components
    
    # Step 1: Compute probability of Eb under each component
    component_log_probs = np.zeros(n_components)
    component_grads = np.zeros((n_components, n_features))
    
    for k in range(n_components):
        mvn = MVN(mean=gmm.means[k], 
                  covariance=gmm.covariances[k],
                  random_state=gmm.random_state)
        
        # Get normalization factor and exponent
        # norm_factor is the 1/sqrt((2pi)^d |Sigma|) term
        # exponent is the -0.5 * (x-mu)^T Sigma^{-1} (x-mu) term
        norm_factor, exponent = mvn.to_norm_factor_and_exponents(Eb)
        
        # Log probability of component k (including prior weight)
        component_log_probs[k] = np.log(gmm.priors[k]) + np.log(norm_factor) + exponent[0]
        
        # Gradient of log N(x | mu_k, Sigma_k) = -Sigma_k^{-1} (x - mu_k)
        cov_inv = np.linalg.inv(gmm.covariances[k])
        # print('shape of cov_inv:', cov_inv.shape)
        # print('shape of Eb.T.reshape(-1,1):', Eb.T.reshape(-1,1).shape)
        # print('shape of gmm.means[k].reshape(-1,1):', gmm.means[k].reshape(-1,1).shape)
        component_grads[k,:] = (-cov_inv @ (Eb.T.reshape(-1,1) - gmm.means[k].reshape(-1,1))).flatten()
    
    # Step 2: Compute responsibilities (posterior weights) using log-sum-exp trick
    max_log_prob = np.max(component_log_probs)
    log_probs_stable = component_log_probs - max_log_prob
    
    # Responsibilities: r_k = p(k|x) = pi_k * N(x|mu_k,Sigma_k) / p(x)
    responsibilities = np.exp(log_probs_stable)
    responsibilities /= np.sum(responsibilities)
    
    # Step 3: Weighted sum of gradients
    grad = np.sum(responsibilities[:, np.newaxis] * component_grads, axis=0)
    
    return grad

def log_likelihoods_sum(beta, Tb, Tpmp, dw, df, eps=0.01):
    L1 = loglikelihood_thawed(beta, Tb, Tpmp, dw)
    L2 = loglikelihood_frozen(beta, Tb, Tpmp, df, eps)
    return L1 + L2

def log_posterior(Eb_star, V, gmm, beta, Tpmp, Eb_mean, Eb_std, dw, df, verbose = False):
    """   
    Compute the log posterior probability 
    
    log p(Eb | evidence) \propto likelihood(thawed_evidence | Eb) + likelihood(frozen_evidence | Eb) + log_prior(Eb)
    leads to negative log posterior for minimization
    - log p(Eb | evidence) =  (1/beta) * (Tpmp - Tb(Eb)) * dw
                              -\Sum_i^N log(1+(eps-1)*exp(-1/beta*(Tpmp - Tb(Eb)))) * df
                              - log P(Eb)

    Parameters:
    -------
    Eb_star: array
        latent enthalpy at base in the reduced space (n_feature_latent,)
    V: array
        right singular vectors from PCA (n_features, n_feature_latent)
    gmm: GaussianMixture
        trained GMM model representing the prior distribution over basal enthalpy fields
    beta: scalar
        temperature scale (K) for the exponential parameterization
    Tpmp: array
        pressure melting point at each pixel (n_features,)
    Eb_mean: array
        mean of Eb in the training data for reverse standardization (n_features,)
    Eb_std: array
        std of Eb in the training data for reverse standardization (n_features,)
    dw: array
        fractional area of thawed base at each pixel (n_features,)
    df: array
        fractional area of frozen base at each pixel (n_features,)

    """
    if isinstance(Eb_star, np.ndarray):
        Eb_star = torch.from_numpy(Eb_star)
    if isinstance(V, np.ndarray):
        V = torch.from_numpy(V)
    if isinstance(Tpmp, np.ndarray):
        Tpmp = torch.from_numpy(Tpmp)
    if isinstance(Eb_mean, np.ndarray):
        Eb_mean = torch.from_numpy(Eb_mean)
    if isinstance(Eb_std, np.ndarray):
        Eb_std = torch.from_numpy(Eb_std)
    if isinstance(dw, np.ndarray):
        dw = torch.from_numpy(dw)
    if isinstance(df, np.ndarray):
        df = torch.from_numpy(df)

    Eb = V.T @ Eb_star # map from latent space to original space
    Eb_ori = reverse_standardize(Eb, Eb_mean, Eb_std) # reverse standardization
    Tb = enthalpy_to_temperature(Eb_ori, Tpmp)
    # check that no Tb is above Tpmp 
    if torch.any(Tb > Tpmp):
        raise ValueError('Tb should not be above Tpmp, but found some Tb > Tpmp')

    L1 = loglikelihood_thawed(beta, Tb, Tpmp, dw)
    L2 = loglikelihood_frozen(beta, Tb, Tpmp, df)

    # Eb_star back to numpy for log_prior computation
    Eb_star_np = Eb_star.detach().numpy()
    log_prior_val = log_prior(Eb_star_np, gmm)
    
    if verbose:
        print(f"Log Likelihood Thawed: {L1.item():.3f}")
        print(f"Log Likelihood Frozen: {L2.item():.3f}")
        print(f"Log Prior: {log_prior_val.item():.3f}")

    return L1 + L2 + log_prior_val

def log_posterior_gradient(Eb_star, V, gmm, beta, Tpmp, Eb_mean, Eb_std, dw, df):
    """   
    Compute the gradient of the log posterior prob wrt Eb.
    
    Here we use Torch AD for the two likelihood terms and use the analytical gradient for the log prior term

    Parameters:
    -------
    Eb_star: array
        basal enthalpy field in the latent space (n_latent_features,)
    V: array
        right singular vectors from PCA (n_features, n_latent_feature)
    gmm: GaussianMixture
        trained GMM model
    Tpmp: array
        pressure melting point at each pixel (n_features,)
    Eb_mean: array
        mean of Eb in the training data for reverse standardization (n_features,)
    Eb_std: array
        std of Eb in the training data for reverse standardization (n_features,)
    dw: array
        fractional area of thawed base at each pixel (n_features,)
    df: array
        fractional area of frozen base at each pixel (n_features,)
    beta: scalar
        temperature scale (K) for the exponential parameterization
    
    """
    if isinstance(V, np.ndarray):
        V = torch.from_numpy(V)
    if isinstance(Eb_star, np.ndarray):
        Eb_star = torch.from_numpy(Eb_star)
    if isinstance(Tpmp, np.ndarray):
        Tpmp = torch.from_numpy(Tpmp)
    if isinstance(Eb_mean, np.ndarray):
        Eb_mean = torch.from_numpy(Eb_mean)
    if isinstance(Eb_std, np.ndarray):
        Eb_std = torch.from_numpy(Eb_std)
    if isinstance(dw, np.ndarray):
        dw = torch.from_numpy(dw)
    if isinstance(df, np.ndarray):
        df = torch.from_numpy(df)

    Eb_star_tensor = torch.tensor(Eb_star, requires_grad=True)
    Eb_tensor = V.T @ Eb_star_tensor
    Eb_ori = reverse_standardize(Eb_tensor, Eb_mean, Eb_std)
    Tb = enthalpy_to_temperature(Eb_ori, Tpmp)

    # forward compute of the likelihood terms
    llsum = log_likelihoods_sum(beta, Tb, Tpmp, dw, df)

    # Compute gradients using AD
    llsum.backward()
    likelihood_grad = Eb_star_tensor.grad.detach().numpy()

    # Compute gradient of log prior
    prior_grad = log_prior_gradient(Eb_star_tensor.detach().numpy(), gmm)

    # Total gradient is the sum of likelihood and prior gradients
    total_grad = likelihood_grad + prior_grad
    
    return total_grad

def finite_difference_check(Eb_star, V, gmm, beta, Tpmp, Eb_mean, Eb_std, dw, df, epsilon=1e-5):
    """  
    Perform finite difference check for the log posterior gradient.

    Parameters:
    -------
    Eb_star: array
        basal enthalpy field in the latent space (n_latent_features,)
    V: array
        right singular vectors from PCA (n_features, n_latent_feature)
    gmm: GaussianMixture
        trained GMM model
    Tpmp: array
        pressure melting point at each pixel (n_features,)
    Eb_mean: array
        mean of Eb in the training data for reverse standardization (n_features,)
    Eb_std: array
        std of Eb in the training data for reverse standardization (n_features,)
    dw: array
        fractional area of thawed base at each pixel (n_features,)
    df: array
        fractional area of frozen base at each pixel (n_features,)
    beta: scalar
        temperature scale (K) for the exponential parameterization
    epsilon: scalar
        small perturbation for finite difference

    Returns:
    -------
    finite_diff_grad: array
        Gradient computed using finite difference approximation
    """
    finite_diff_grad = np.zeros_like(Eb_star)
    
    for i in range(len(Eb_star)):
        Eb_star_plus = np.copy(Eb_star)
        Eb_star_minus = np.copy(Eb_star)
        
        Eb_star_plus[i] += epsilon
        Eb_star_minus[i] -= epsilon
        
        log_post_plus = log_posterior(Eb_star_plus, V, gmm, beta, Tpmp, Eb_mean, Eb_std, dw, df)
        log_post_minus = log_posterior(Eb_star_minus, V, gmm, beta, Tpmp, Eb_mean, Eb_std, dw, df)
        
        finite_diff_grad[i] = (log_post_plus - log_post_minus) / (2 * epsilon)
    
    return finite_diff_grad

def reverse_standardize(X, mean, std, method='standard'):
    """
    reverse z-score standardization (standard)
    or with relaxation 

    """
    # first check X, mean, and std have the same shape
    # or it will do outer product and our computer will EXPLODE SIR
    shape_check(X, mean, std)
    if method == 'standard':
        return X * std + mean
    elif method == 'relaxation':
        return X * std + mean
    else:
        raise ValueError("Unknown method: {}".format(method))
    
def shape_check(*arrays):
    """  
    Check that all input arrays have the same shape.

    Parameters:
    -------
    *arrays: list of arrays
        list of arrays to check

    Raises:
    -------
    ValueError: if any two arrays have different shapes
    """
    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) > 1:
        raise ValueError("All input arrays must have the same shape, but got shapes: {}".format(shapes))