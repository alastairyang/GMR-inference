import torch
import numpy as np
from src.utilities import shape_check
import scipy as sp
from gmr import MVN

class enthalpyPosterior:

    """
    Bayesian inference over Gaussian Process and Gaussian Mixture Model
    latent representations. 

    This class contains the essential methods for computing log likelihood, prior, and
    posterior (As well as the gradients) of the latent representations.

    """

    def __init__(
            self,
            forward_model,
            evidence,
            gmm,
            gp,
            config,
    ):
        self.forward_model = forward_model
        self.evidence = evidence
        self.gmm = gmm # gaussian mixture class, "gmmAugmented"
        self.gp  = gp  # gaussian process class, "lowRankGP"
        self.config = config


    def thawed_log_likelihood(self, delta_T):
        beta = self.config.beta
        dw   = self.evidence.thawed_fraction

        shape_check(delta_T, dw)
        return -(1.0 / beta) * torch.sum(delta_T * dw)

    def frozen_log_likelihood(self, delta_T):
        beta = self.config.beta
        eps  = self.config.eps
        df   = self.evidence.frozen_fraction
        shape_check(delta_T, df)
        value = 1.0 + (eps - 1.0) * torch.exp(
            -(1.0 / beta) * delta_T
        )
        return torch.sum(torch.log(value) * df)

    def water_fraction_log_likelihood(self, water_fraction):
        beta_w = self.config.beta_w
        eps = self.config.eps
        threshold = self.config.water_fraction_threshold

        scaled = (water_fraction - threshold) / beta_w

        # sigmoid(-scaled) avoids explicitly computing
        # 1 / (1 + exp(scaled)).
        probability = torch.sigmoid(-scaled)

        return torch.sum(torch.log(probability + eps))

    def log_likelihood(self, z_gp, z_gmm):
        state = self.forward_model.physical_state(
            z_gp,
            z_gmm,
        )

        return (
            self.thawed_log_likelihood(state["delta_T"])
            + self.frozen_log_likelihood(state["delta_T"])
            + self.water_fraction_log_likelihood(
                state["water_fraction"]
            )
        )

    @staticmethod
    def gp_log_prior(z_gp):
        return -0.5 * torch.sum(z_gp ** 2)

    def gmm_log_prior_numpy(self, z_gmm):
        z_gmm_np = (
            z_gmm.detach().cpu().numpy()
        )

        value = self.gmm.to_log_probability_density(
            z_gmm_np,
        )
        return float(np.asarray(value).reshape(-1)[0])

    def log_prob(self, z_gp, z_gmm, *, verbose=False):
        z_gp  = self.forward_model.to_tensor(z_gp)
        z_gmm = self.forward_model.to_tensor(z_gmm)

        ll = self.log_likelihood(z_gp, z_gmm)
        gp_prior  = self.gp_log_prior(z_gp)
        gmm_prior = self.gmm_log_prior_numpy(z_gmm)

        total = ll + gp_prior + gmm_prior
        if verbose:
            print(f"log_likelihood: {ll}")
            print(f"gp_log_prior: {gp_prior}")
            print(f"gmm_log_prior: {gmm_prior}")
            print(f"total: {total}")
        return total

    # ---------------- GRADIENT ----------------
    def gradient(self, z_gp, z_gmm):
        """
        Compute the gradient of the log probability with respect to z_gp and z_gmm.

        """
        z_gp_tensor = self.forward_model.to_tensor(
            z_gp,
            requires_grad=True,
        )
        z_gmm_tensor = self.forward_model.to_tensor(
            z_gmm,
            requires_grad=True,
        )

        likelihood = self.log_likelihood(
            z_gp_tensor,
            z_gmm_tensor,
        )

        likelihood_grad_gp, likelihood_grad_gmm = (
            torch.autograd.grad(
                likelihood,
                (z_gp_tensor, z_gmm_tensor),
            )
        )

        likelihood_grad_gp = (
            likelihood_grad_gp.detach().cpu().numpy()
        )
        likelihood_grad_gmm = (
            likelihood_grad_gmm.detach().cpu().numpy()
        )

        gp_prior_grad = self.log_prior_gp_gradient(
            z_gp_tensor
        )
        gmm_prior_grad = self.gmm.log_prior_gmm_gradient(
            z_gmm_tensor.detach().cpu().numpy(), 
        )

        grad_gp  = likelihood_grad_gp  + gp_prior_grad
        grad_gmm = likelihood_grad_gmm + gmm_prior_grad

        return grad_gp, grad_gmm

    def packed_gradient(self, z_gp, z_gmm):
        grad_gp, grad_gmm = self.gradient(z_gp, z_gmm)
        return np.concatenate([grad_gp, grad_gmm])

    @staticmethod
    def log_prior_gp_gradient(z_gp):
        return -z_gp