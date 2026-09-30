import torch
from src.observationOperator import latent_operator_enthalpy, binary_operator

class PriorCorrection:
    def __init__(self):
        pass

    def load_simulation_var(self, V, Eb_star, Eb_mean, Eb_std, Tpmp, method, epsilon, mask, dT_cutoff=1):
        self.V         = V
        self.Eb_star   = Eb_star
        self.Eb_mean   = Eb_mean
        self.Eb_std    = Eb_std
        self.Tpmp      = Tpmp
        self.method    = method # standardization method
        self.epsilon   = epsilon # standardization relaxation constant

        return

    def load_observation(self, base, dT_cutoff=1):
        """ 
        Load data associated with observations

        Parameters
        ----------
        base : torch.Tensor
            Observed binary base (thawed or frozen).
                1: thawed
                0: frozen
                nan: not observed
        dT_cutoff : float, optional
            Threshold for classifying as thawed or frozen. Default is 1.
        """
        self.base      = base
        self.mask      = ~torch.isnan(base) # binary observation mask. if observed -> 1
        self.dT_cutoff = dT_cutoff
        self.n_obs     = self.mask.sum() # total number of obs
        return

    def _initialize_alpha(self):
        """
        initialize the alpha vector for the prior correction
        """
        self.alpha = torch.zeros_like(self.Eb_star)

        return
    def _simulated_obs(self, alpha):
        
        self.Eb_star_p = self.Eb_star * torch.exp(alpha)
        delta_T = latent_operator_enthalpy(self.V, 
                                           self.Eb_star_p, 
                                           self.Eb_mean, 
                                           self.Eb_std, 
                                           self.Tpmp,
                                           self.method,
                                           self.epsilon)

        binary_base = binary_operator(delta_T, self.dT_cutoff)
        return binary_base

    def log_likelihood(self, sigma_sq, alpha):
        """
        Compute the log-likelihood of the observed binary base
        given the simulated binary base from the current model parameters.
        
        Parameters
        ----------
        sigma_sq : torch.Tensor
            The variance of the observation noise in the MVN likelihood, assuming isotropy.
        alpha : torch.Tensor
            The current estimate of the alpha vector for the prior correction.
            
        Returns
        -------
        log_likelihood : torch.Tensor
            Log-likelihood of the observed binary base given the simulated binary base.
        """

        # the scaling constants for MVN
        mvn_constant = -0.5 * self.n_obs * torch.log(sigma_sq) \
                       - self.n_obs / 2 * torch.log(2 * torch.pi)

        mu = self._simulated_obs(alpha)
        mu = mu[self.mask]
        obs =self.base[self.mask]

        mvn_ll = mvn_constant - 0.5 * torch.sum((obs - mu) ** 2) / sigma_sq
        return mvn_ll

    def lasso(self, lasso_lambda, alpha):
        """
        Lasso-type regularization with a lass_lambda weight parameter

        """
        return lasso_lambda * torch.sum(torch.abs(alpha))

    def solve_MAP(self, lasso_lambda, sigma_sq):
        """
        Solve this maximum a posteriori (MAP) estimation problem for alpha.
        """

        self._initialize_alpha()

        def objective(alpha):
            return -self.log_likelihood(sigma_sq, alpha) + self.lasso(lasso_lambda, alpha)

        result = torch.optim.minimize(objective, self.alpha)
        self.alpha = result.x
        return self.alpha

    