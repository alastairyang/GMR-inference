import math

import torch
import torch.optim as optim

from src.observationOperator import (
    latent_operator_enthalpy,
    binary_soft_operator,
)


class PriorCorrection:
    """
    Estimate mode-dependent multiplicative corrections to PCA coefficients.

    The corrected coefficients are

        Eb_star_corrected = Eb_star * exp(alpha),

    where alpha has one entry per retained PCA mode. Negative alpha values
    contract a mode, while positive alpha values inflate it.

    The covariance of the simulated evidence is computed once at a fixed
    reference value of alpha and held constant during optimization.
    """

    def __init__(self):
        self.alpha = None

        # Fixed-covariance SVD factors
        self.cov_singular_values = None
        self.cov_Vh = None
        self.cov_rank = None

        # Quantities depending on fixed sigma_sq
        self.sigma_sq = None
        self.woodbury_weights = None
        self.logdet_S = None

    def load_simulation_var(
        self,
        V,
        Eb_star,
        Eb_mean,
        Eb_std,
        Tpmp,
        method,
        epsilon,
    ):
        """
        Load the simulated PCA coefficients and reconstruction variables.

        Parameters
        ----------
        V : torch.Tensor
            PCA basis used to reconstruct the basal enthalpy field.
        Eb_star : torch.Tensor
            PCA coefficients with shape (K, r_E), where K is the number
            of simulations and r_E is the number of retained PCA modes.
        Eb_mean, Eb_std : torch.Tensor
            Quantities used to invert the standardization.
        Tpmp : torch.Tensor
            Pressure-melting-point temperature field.
        method
            Standardization method passed to latent_operator_enthalpy.
        epsilon
            Relaxation constant passed to latent_operator_enthalpy.
        """
        if Eb_star.ndim != 2:
            raise ValueError(
                "Eb_star must have shape (K, r_E), with simulations "
                "along the first dimension and PCA modes along the last."
            )

        self.V = V
        self.Eb_star = Eb_star
        self.Eb_mean = Eb_mean
        self.Eb_std = Eb_std
        self.Tpmp = Tpmp
        self.method = method
        self.epsilon = epsilon

        self.n_simulations = Eb_star.shape[1]
        self.n_modes = Eb_star.shape[0]

    def load_observation(self, base, beta=1.0):
        """
        Load the observed basal thermal-state evidence.

        Parameters
        ----------
        base : torch.Tensor
            Observed basal thermal-state classifications:

                1   : thawed
                0   : frozen
                NaN : unobserved

            The spatial dimensions are flattened internally.
        beta : float, optional
            Parameter passed to binary_soft_operator. For gradient-based
            optimization, binary_soft_operator must implement a continuous,
            differentiable soft-classification map.
            differentiable soft-classification map.
        """
        self.base = base
        self.mask = ~torch.isnan(base)
        self.beta = beta
        self.n_obs = int(self.mask.sum().item())

        if self.n_obs == 0:
            raise ValueError("No valid observations were provided.")

    def _initialize_alpha(self):
        """
        Initialize one correction parameter per retained PCA mode.

        alpha_i = 0 corresponds to a multiplicative scale exp(alpha_i) = 1.
        """
        self.alpha = torch.nn.Parameter(
            torch.zeros(
                self.n_modes,
                dtype=self.Eb_star.dtype,
                device=self.Eb_star.device,
            )
        )

    def _simulated_obs(self, alpha):
        """
        Compute the simulated soft basal thermal-state evidence.

        Parameters
        ----------
        alpha : torch.Tensor
            Mode-dependent log-scale corrections with shape (r_E,).

        Returns
        -------
        simulated_evidence : torch.Tensor
            Simulated soft evidence. The first dimension must correspond
            to the K simulations.
        """
        if alpha.shape != (self.n_modes,):
            raise ValueError(
                f"alpha must have shape ({self.n_modes},), "
                f"but received {tuple(alpha.shape)}."
            )

        # Broadcasting applies the same correction to each ensemble member.
        # Negative alpha contracts a PCA mode; positive alpha inflates it.
        scale = torch.exp(alpha)
        Eb_star_corrected = self.Eb_star.T * scale.unsqueeze(0)
        Eb_star_corrected = Eb_star_corrected.T

        delta_T = latent_operator_enthalpy(
            self.V,
            Eb_star_corrected,
            self.Eb_mean,
            self.Eb_std,
            self.Tpmp,
            self.method,
            self.epsilon,
        )

        # IMPORTANT: binary_operator must be a smooth map during optimization.
        simulated_evidence = binary_soft_operator(
            delta_T,
            self.beta,
        )

        return simulated_evidence.T

    def _observed_simulated_evidence(self, alpha):
        """
        Evaluate the simulated evidence only at observed locations.

        Returns
        -------
        torch.Tensor
            Matrix with shape (K, d), where d is the number of observed
            evidence locations.
        """
        simulated_evidence = self._simulated_obs(alpha)
        print("simulated_evidence shape:", simulated_evidence.shape)

        if simulated_evidence.shape[0] != self.n_simulations:
            raise ValueError(
                "The first dimension returned by _simulated_obs must "
                "correspond to the simulation ensemble."
            )

        # Flatten all non-ensemble dimensions.
        simulated_evidence = simulated_evidence.reshape(
            self.n_simulations, -1
        )
        flat_mask = self.mask.reshape(-1)

        if simulated_evidence.shape[1] != flat_mask.numel():
            raise ValueError(
                "The simulated evidence and observed base do not have "
                "the same number of spatial locations."
            )

        # The mask acts on evidence locations, not simulations.
        return simulated_evidence[:, flat_mask]

    def _covariance_matrix(self, alpha_ref=None, rank_tolerance=None):
        """
        Precompute the compact SVD of the centered evidence matrix.

        The fixed centered matrix is

            C_ref = (Omega_ref - 1_K mean(Omega_ref)^T) / sqrt(K - 1).

        The fixed covariance is represented implicitly as

            S = C_ref.T @ C_ref + sigma_sq * I.

        Parameters
        ----------
        alpha_ref : torch.Tensor, optional
            Reference correction vector. By default, alpha_ref = 0, which
            corresponds to the uncorrected ensemble.
        rank_tolerance : float, optional
            Singular values smaller than this threshold are discarded.
            If omitted, a standard numerical-rank tolerance is used.
        """
        if self.n_simulations < 2:
            raise ValueError(
                "At least two simulations are required to estimate covariance."
            )

        if alpha_ref is None:
            alpha_ref = torch.zeros(
                self.n_modes,
                dtype=self.Eb_star.dtype,
                device=self.Eb_star.device,
            )
        else:
            alpha_ref = torch.as_tensor(
                alpha_ref,
                dtype=self.Eb_star.dtype,
                device=self.Eb_star.device,
            )

        if alpha_ref.shape != (self.n_modes,):
            raise ValueError(
                f"alpha_ref must have shape ({self.n_modes},), "
                f"but received {tuple(alpha_ref.shape)}."
            )

        # The covariance is deliberately detached from optimization.
        with torch.no_grad():
            Omega_ref = self._observed_simulated_evidence(alpha_ref)
            mean_ref = Omega_ref.mean(dim=0, keepdim=True)

            C_ref = (
                Omega_ref - mean_ref
            ) / math.sqrt(self.n_simulations - 1)

            # C_ref has shape (K, d).
            _, singular_values, Vh = torch.linalg.svd(
                C_ref,
                full_matrices=False,
            )

            # Remove numerically zero singular values.
            if singular_values.numel() == 0:
                keep = torch.zeros(
                    0,
                    dtype=torch.bool,
                    device=C_ref.device,
                )
            else:
                if rank_tolerance is None:
                    rank_tolerance = (
                        max(C_ref.shape)
                        * torch.finfo(C_ref.dtype).eps
                        * singular_values.max()
                    )

                keep = singular_values > rank_tolerance

            self.cov_singular_values = singular_values[keep].detach()
            self.cov_Vh = Vh[keep, :].detach()
            self.cov_rank = int(keep.sum().item())
            self.n_evidence = C_ref.shape[1]

        return (
            self.cov_singular_values,
            self.cov_Vh,
        )

    def _prepare_fixed_metric(
        self,
        sigma_sq,
        alpha_ref=None,
        rank_tolerance=None,
    ):
        """
        Precompute all fixed covariance quantities used by the likelihood.
        """
        sigma_sq = torch.as_tensor(
            sigma_sq,
            dtype=self.Eb_star.dtype,
            device=self.Eb_star.device,
        )

        if sigma_sq.numel() != 1:
            raise ValueError("sigma_sq must be a scalar.")

        sigma_sq = sigma_sq.reshape(())

        if sigma_sq.item() <= 0.0:
            raise ValueError("sigma_sq must be strictly positive.")

        self._covariance_matrix(
            alpha_ref=alpha_ref,
            rank_tolerance=rank_tolerance,
        )

        self.sigma_sq = sigma_sq.detach()

        s_sq = self.cov_singular_values.square()

        # Diagonal weights in the Woodbury correction:
        #
        # s_i^2 / [sigma^2 (sigma^2 + s_i^2)]
        self.woodbury_weights = (
            s_sq
            / (
                self.sigma_sq
                * (self.sigma_sq + s_sq)
            )
        ).detach()

        # log |S| = d log(sigma^2)
        #           + sum_i log(1 + s_i^2 / sigma^2)
        self.logdet_S = (
            self.n_evidence * torch.log(self.sigma_sq)
            + torch.log1p(s_sq / self.sigma_sq).sum()
        ).detach()

    def mahalanobis(self, alpha):
        """
        Evaluate P(alpha).T @ S^{-1} @ P(alpha) using the fixed SVD.

        The covariance matrix and its inverse are never formed explicitly.
        """
        if self.sigma_sq is None:
            raise RuntimeError(
                "The fixed covariance metric has not been prepared."
            )

        Omega = self._observed_simulated_evidence(alpha)

        # Ensemble-mean simulated evidence, shape (d,).
        simulated_mean = Omega.mean(dim=0)

        observed = self.base.reshape(-1)[self.mask.reshape(-1)]
        observed = observed.to(
            dtype=simulated_mean.dtype,
            device=simulated_mean.device,
        )

        residual = observed - simulated_mean

        # First Woodbury term:
        # ||P||^2 / sigma^2
        quadratic = residual.square().sum() / self.sigma_sq

        # Second Woodbury term:
        # P.T V diag(w_i) V.T P
        if self.cov_rank > 0:
            projected_residual = self.cov_Vh @ residual
            correction = (
                self.woodbury_weights
                * projected_residual.square()
            ).sum()

            quadratic = quadratic - correction

        return quadratic

    def log_likelihood(self, alpha, include_constants=True):
        """
        Compute the fixed-covariance Gaussian working log-likelihood.

        Parameters
        ----------
        alpha : torch.Tensor
            Current mode-dependent log-scale corrections.
        include_constants : bool, optional
            Include the normalizing constant and fixed log determinant.
            These terms can be omitted during optimization because they
            do not depend on alpha.
        """
        quadratic = self.mahalanobis(alpha)

        if not include_constants:
            return -0.5 * quadratic

        normalizing_constant = (
            self.n_evidence
            * math.log(2.0 * math.pi)
        )

        return -0.5 * (
            normalizing_constant
            + self.logdet_S
            + quadratic
        )

    @staticmethod
    def smooth_elastic_net(
        alpha,
        regularization_strength,
        l1_ratio=0.9,
        smoothing_epsilon=1.0e-6,
    ):
        """
        Differentiable approximation to the elastic-net penalty.

        The penalty is

            gamma * [
                eta * sum(sqrt(alpha_i^2 + eps^2) - eps)
                + (1 - eta) / 2 * ||alpha||_2^2
            ],

        where gamma is regularization_strength and eta is l1_ratio.

        Parameters
        ----------
        alpha : torch.Tensor
            Correction parameters.
        regularization_strength : float or torch.Tensor
            Overall penalty weight, gamma.
        l1_ratio : float, optional
            Relative weight eta assigned to the smoothed L1 component.
            Must lie in [0, 1].
        smoothing_epsilon : float, optional
            Positive smoothing parameter for the L1 approximation.
        """
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("l1_ratio must lie between 0 and 1.")

        if smoothing_epsilon <= 0.0:
            raise ValueError(
                "smoothing_epsilon must be strictly positive."
            )

        gamma = torch.as_tensor(
            regularization_strength,
            dtype=alpha.dtype,
            device=alpha.device,
        )
        eta = alpha.new_tensor(l1_ratio)
        eps = alpha.new_tensor(smoothing_epsilon)

        smooth_l1 = (
            torch.sqrt(alpha.square() + eps.square()) - eps
        ).sum()

        l2 = 0.5 * alpha.square().sum()

        return gamma * (
            eta * smooth_l1
            + (1.0 - eta) * l2
        )

    def objective(
        self,
        alpha,
        regularization_strength,
        l1_ratio,
        smoothing_epsilon,
    ):
        """
        Reduced MAP objective.

        Terms in the Gaussian likelihood that are constant with respect
        to alpha are omitted.
        """
        data_misfit = 0.5 * self.mahalanobis(alpha)

        penalty = self.smooth_elastic_net(
            alpha=alpha,
            regularization_strength=regularization_strength,
            l1_ratio=l1_ratio,
            smoothing_epsilon=smoothing_epsilon,
        )

        return data_misfit + penalty

    def solve_MAP(
        self,
        regularization_strength,
        sigma_sq,
        l1_ratio=0.9,
        smoothing_epsilon=1.0e-6,
        alpha_ref=None,
        rank_tolerance=None,
        lr=0.5,
        max_iter=100,
        tolerance_grad=1.0e-7,
        tolerance_change=1.0e-9,
    ):
        """
        Solve the smooth elastic-net MAP problem using L-BFGS.

        Parameters
        ----------
        regularization_strength : float
            Overall elastic-net weight, gamma.
        sigma_sq : float or torch.Tensor
            Isotropic residual variance, sigma^2.
        l1_ratio : float, optional
            Elastic-net mixing parameter:

                1.0 : smoothed Lasso
                0.0 : ridge
                0 < l1_ratio < 1 : smooth elastic net

        smoothing_epsilon : float, optional
            Smoothing parameter used in the differentiable approximation
            to the absolute value.
        alpha_ref : torch.Tensor, optional
            Reference correction used to construct the fixed covariance.
            The default is alpha_ref = 0.
        rank_tolerance : float, optional
            Numerical rank tolerance for the covariance SVD.
        lr : float, optional
            L-BFGS learning-rate parameter.
        max_iter : int, optional
            Maximum number of L-BFGS iterations.
        """
        self._initialize_alpha()

        # Compute C_ref, its compact SVD, and all fixed covariance terms.
        self._prepare_fixed_metric(
            sigma_sq=sigma_sq,
            alpha_ref=alpha_ref,
            rank_tolerance=rank_tolerance,
        )

        optimizer = optim.LBFGS(
            [self.alpha],
            lr=lr,
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()

            loss = self.objective(
                alpha=self.alpha,
                regularization_strength=regularization_strength,
                l1_ratio=l1_ratio,
                smoothing_epsilon=smoothing_epsilon,
            )

            loss.backward()
            return loss

        optimizer.step(closure)

        # Evaluate the final objective outside the closure for reporting.
        with torch.no_grad():
            final_loss = self.objective(
                alpha=self.alpha,
                regularization_strength=regularization_strength,
                l1_ratio=l1_ratio,
                smoothing_epsilon=smoothing_epsilon,
            )

        result = {
            "alpha": self.alpha.detach().clone(),
            "scale": torch.exp(self.alpha.detach()).clone(),
            "loss": final_loss.detach().clone(),
            "covariance_rank": self.cov_rank,
            "singular_values": self.cov_singular_values.clone(),
            "logdet_S": self.logdet_S.clone(),
        }

        return result
