import numpy as np
from gmr import MVN
from gmr.gmm import _safe_probability_density
# import pytorch for AD
import torch

# functions related to finding Maximum A Posteriori (MAP) in the latent space of a trained GMR model

def dTb_dEb(Eb, Tpmp, Cp=2093.0):
    """  
    Compute a smooth gradient from enthalpy to temperature.

    Tb = Tpmp if Eb > Cp*Tpmp
    Tb = (Eb + Cp*To)/Cp if Eb <= Cp*Tpmp

    similar to differentiating a ReLU function, 
    we can compute the gradient dTb/dEb as follows:
    dTb/dEb = 0 if Eb > Cp*Tpmp
    dTb/dEb = 1/Cp if Eb <= Cp*Tpmp
    
    """
    if Eb > Cp * Tpmp:
        return 0.0
    else:
        return 1.0 / Cp

def dL1_dEb(beta, Eb, Tpmp, dw, Cp=2093.0):
    """  
    Compute the gradient of log likelihood 1 (the thawed base evidence) with respect to the
      latent enthalpy at base (Eb asterisk)

    dL1/dEb* = -(1/beta) * \Sum_i^N dTb/dEb * dw

    Parameters:
    -------
    beta: scalar
        temperature scale (K) for the exponential parameterization
    Eb: array
        basal enthalpy
    Tpmp: array
        pressure melting point 
    dw: array
        area of thawed base at each pixel
    Cp: scalar
        specific heat capacity of ice (default 2093 J/kg/K)

    Returns:
    -------
    dL1_dEb: array
        gradient of log likelihood 1 with respect to Eb
    """
    return -(1.0/beta) * np.sum(dTb_dEb(Eb, Tpmp, Cp) * dw)

def dEb_dEbstar(dL1_dEb, V):
    """
    Gradient of enthalpy at base Eb wrt its latent values.
    For PCA-based model reduction, it's just the right singular vectors

    since dL1_dEb is a scalar, whereas V is (n_features, n_components),
    we augment dL1_dEb to be (n_features,) by repeating it, 
    then multiply with V to get the gradient in the latent space (n_components,)
    """
    dL1_dEb = np.full(V.shape[0], dL1_dEb)  # shape (n_features,)
    return dL1_dEb @ V

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

    return -torch.sum(torch.log(1 + (eps-1)*torch.exp(-(1/beta)*(Tpmp-Tb))) * df) 

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

    return np.log(gmm.to_probability_density(Eb))

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
    n_features = Eb.shape[1]
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

        component_grads[k,:] = (-cov_inv @ (Eb.T - gmm.means[k].reshape(-1,1))).flatten()
    
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

def log_posterior(Eb_star, V, gmm, beta, Tpmp, dw, df):
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
    if isinstance(dw, np.ndarray):
        dw = torch.from_numpy(dw)
    if isinstance(df, np.ndarray):
        df = torch.from_numpy(df)

    Eb = V.T @ Eb_star # map from latent space to original space
    Tb = enthalpy_to_temperature(Eb, Tpmp)

    L1 = loglikelihood_thawed(beta, Tb, Tpmp, dw)
    L2 = loglikelihood_frozen(beta, Tb, Tpmp, df)

    # Eb_star back to numpy for log_prior computation
    Eb_star_np = Eb_star.detach().numpy()
    log_prior_val = log_prior(Eb_star_np, gmm)
    
    return -(L1 + L2 + log_prior_val)

def log_posterior_gradient(Eb, gmm, beta, Tpmp, dw, df):
    """   
    Compute the gradient of the log posterior prob wrt Eb.
    
    Here we use Torch AD for the two likelihood terms and use the analytical gradient for the log prior term
    
    """
    Eb_tensor = torch.tensor(Eb, requires_grad=True)
    Tb = enthalpy_to_temperature(Eb_tensor, Tpmp)

    # forward compute of the likelihood terms
    llsum = log_likelihoods_sum(beta, Tb, Tpmp, dw, df)

    # Compute gradients using AD
    llsum.backward()
    likelihood_grad = Eb_tensor.grad.detach().numpy()

    # Compute gradient of log prior
    prior_grad = log_prior_gradient(Eb, gmm)

    # Total gradient is the sum of likelihood and prior gradients
    total_grad = likelihood_grad + prior_grad
    
    return total_grad

def enthalpy_to_temperature(Eb, Tpmp, Cp=2093.0, C0=223.15):
    """  
    Compute basal temperature from enthalpy

    Tb = Tpmp if Eb > Cp*Tpmp
    Tb = (Eb + Cp*To)/Cp if Eb <= Cp*Tpmp

    """

    Tb = torch.where(Eb > Cp * Tpmp, Tpmp, (Eb + Cp*C0) / Cp)
    return Tb