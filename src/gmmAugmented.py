from gmr import MVN, gmm
import numpy as np
import scipy as sp

class gmmAugmented:
    """
    A wrapper class for a Gaussian Mixture Model (GMM) 
    that provides additional functionality, built on top of 
    the gmr package (https://github.com/AlexanderFabisch/gmr)
    """
    def __init__(self, gmm):
        self.gmm = gmm

    def log_prior_gmm_gradient(self, Eb):
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

        expected_features = self.gmm.means.shape[1]

        if Eb.size != expected_features:
            raise ValueError(
                f"GMM expects {expected_features} latent features, "
                f"but received {Eb.size}."
            )

        n_features = Eb.shape[0]
        n_components = self.gmm.n_components
        
        # Step 1: Compute probability of Eb under each component
        component_log_probs = np.zeros(n_components)
        component_grads = np.zeros((n_components, n_features))
        
        for k in range(n_components):
            mvn = MVN(mean=self.gmm.means[k], 
                    covariance=self.gmm.covariances[k],
                    random_state=self.gmm.random_state)
            
            # Get normalization factor and exponent
            norm_factor, exponent = mvn.to_norm_factor_and_exponents(Eb)
            
            # Log probability of component k (including prior weight)
            component_log_probs[k] = np.log(self.gmm.priors[k]) + np.log(norm_factor) + exponent[0]
            
            # Gradient of log N(x | mu_k, Sigma_k) = -Sigma_k^{-1} (x - mu_k)
            cov_inv = np.linalg.inv(self.gmm.covariances[k])
            # print('shape of cov_inv:', cov_inv.shape)
            # print('shape of Eb.T.reshape(-1,1):', Eb.T.reshape(-1,1).shape)
            # print('shape of self.gmm.means[k].reshape(-1,1):', self.gmm.means[k].reshape(-1,1).shape)
            component_grads[k,:] = (-cov_inv @ (Eb.T.reshape(-1,1) - self.gmm.means[k].reshape(-1,1))).flatten()
        
        # Compute responsibilities (posterior weights) using log-sum-exp trick
        max_log_prob = np.max(component_log_probs)
        log_probs_stable = component_log_probs - max_log_prob
        
        # Responsibilities: r_k = p(k|x) = pi_k * N(x|mu_k,Sigma_k) / p(x)
        responsibilities = np.exp(log_probs_stable)
        responsibilities /= np.sum(responsibilities)
        
        # Weighted sum of gradients
        grad = np.sum(responsibilities[:, np.newaxis] * component_grads, axis=0)
        
        return grad
    
    def to_log_probability_density(self, X):
        """
        Compute the log probability density for each sample in X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.
        
        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log probability density for each sample.
        """
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        
        # Store log probabilities for each component and sample
        log_prob_components = np.zeros((n_samples, self.gmm.n_components))
        
        for k in range(self.gmm.n_components):
            mean = self.gmm.means[k]
            covariance = self.gmm.covariances[k]
            
            # Cholesky decomposition
            try:
                L = sp.linalg.cholesky(covariance, lower=True)
            except np.linalg.LinAlgError:
                L = sp.linalg.cholesky(
                    covariance + 1e-3 * np.eye(n_features), lower=True)
            
            # Log normalization constant: log(1/sqrt((2π)^d * |Σ|))
            log_det_L = np.sum(np.log(np.diag(L)))  # log|L| = sum(log(L_ii))
            log_norm = -0.5 * n_features * np.log(2.0 * np.pi) - log_det_L
            
            # Mahalanobis distance
            X_minus_mean = X - mean
            X_normalized = sp.linalg.solve_triangular(
                L, X_minus_mean.T, lower=True).T
            log_exponent = -0.5 * np.sum(X_normalized ** 2, axis=1)
            
            # Log probability for component k: log(π_k) + log(N(x|μ_k, Σ_k))
            log_prob_components[:, k] = np.log(self.gmm.priors[k]) + log_norm + log_exponent
        
        # Log-sum-exp trick: log(Σ exp(x_i)) = c + log(Σ exp(x_i - c))
        c = np.max(log_prob_components, axis=1, keepdims=True)
        log_prob = c.squeeze() + np.log(np.sum(np.exp(log_prob_components - c), axis=1))
        
        return log_prob