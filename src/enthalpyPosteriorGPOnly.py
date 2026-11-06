import torch
import numpy as np
import scipy as sp

class enthalpyPosteriorGPOnly:

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
            gp,
            config,
    ):
        self.forward_model = forward_model
        self.evidence = evidence
        self.gp  = gp  # gaussian process class, "lowRankGP"
        self.config = config

        self.gp_dim    = self.gp.n_mode
        self.total_dim = self.gp_dim
        # absolute index to avoid confusion when slicing
        self.gp_dim_index  = slice(0,           self.gp_dim)

    def thawed_log_likelihood(self, delta_T):
        beta = self.config.beta
        dw = self.evidence.thawed_fraction

        delta_T = delta_T.squeeze(-1)
        dw = dw.squeeze(-1)

        if delta_T.shape != dw.shape:
            raise ValueError(
                f"delta_T and thawed_fraction must have identical shapes; "
                f"got {tuple(delta_T.shape)} and {tuple(dw.shape)}."
            )

        return -(1.0 / beta) * torch.sum(delta_T * dw)

    def frozen_log_likelihood(self, delta_T):
        beta = self.config.beta
        eps = self.config.eps
        df = self.evidence.frozen_fraction

        delta_T = delta_T.squeeze(-1)
        df = df.squeeze(-1)

        if delta_T.shape != df.shape:
            raise ValueError(
                f"delta_T and frozen_fraction must have identical shapes; "
                f"got {tuple(delta_T.shape)} and {tuple(df.shape)}."
            )

        value = (
            1.0
            + (eps - 1.0)
            * torch.exp(-delta_T / beta)
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

    def log_likelihood(self, z_gp):
        state = self.forward_model.physical_state(
            z_gp,
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

    def log_prob(self, z_gp, *, verbose=False):
        z_gp  = self.forward_model.to_tensor(z_gp)

        ll = self.log_likelihood(z_gp)
        gp_prior  = self.gp_log_prior(z_gp)

        total = ll + gp_prior
        if verbose:
            print(f"log_likelihood: {ll}")
            print(f"gp_log_prior: {gp_prior}")
            print(f"total: {total}")
        return total

    # ---------------- GRADIENT ----------------
    def gradient(self, z_gp):

        z_gp_tensor = self.forward_model.to_tensor(
            z_gp,
            requires_grad=True,
        ).reshape(-1, 1)

        likelihood = self.log_likelihood(
            z_gp_tensor,
        )

        likelihood_grad_gp = torch.autograd.grad(
            likelihood,
            (z_gp_tensor),
        )[0]

        likelihood_grad_gp = (
            likelihood_grad_gp.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        gp_prior_grad = -(
            z_gp_tensor.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        

        grad_gp = likelihood_grad_gp + gp_prior_grad
        return grad_gp

    def packed_gradient(self, z_gp):
        grad_gp = self.gradient(z_gp)

        return np.asarray(grad_gp).reshape(-1)

    def hessian(self, z_gp):
        """
        Compute the Hessian of the log posterior with respect to z_gp.


        """
        z_gp_tensor = self.forward_model.to_tensor(
            z_gp,
            requires_grad=True,
        ).reshape(-1, 1)

        def likelihood_fn(z_gp_tensor):
            return self.log_likelihood(z_gp_tensor)

        z_combined = z_gp_tensor.flatten()

        likelihood_hessian = torch.autograd.functional.hessian(
            likelihood_fn,
            z_combined
        )

        likelihood_hessian     = likelihood_hessian.detach().cpu().numpy()
        likelihood_hessian_gp  = likelihood_hessian[self.gp_dim_index, self.gp_dim_index]
        gp_prior_hessian = -np.eye(z_gp_tensor.shape[0])

        hessian_gp =  likelihood_hessian_gp  + gp_prior_hessian

        # assemble a total hessian
        hessian_total = likelihood_hessian.copy()
        hessian_total[self.gp_dim_index, self.gp_dim_index] = hessian_gp

        return hessian_gp, hessian_total

    def laplace_approx(self, z_gp, n_sample = 1000):
        """
        Laplace approximation of the posterior
        Take samples from the Laplace approximation of the posterior

        """
        hessian_gp, hessian_total = self.hessian(z_gp)
        neg_inv_hessian_total = -np.linalg.inv(hessian_total)

        # draw samples from the Laplace approximation
        laplace_samples = np.random.multivariate_normal(
            mean=z_gp.reshape(-1),
            cov=neg_inv_hessian_total,
            size=n_sample
        )

        return laplace_samples


    @staticmethod
    def log_prior_gp_gradient(z_gp):
        return -z_gp