import math

import torch
import torch.optim as optim
from tqdm.auto import tqdm

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

    def _prepare_observed_domain(self):
        """
        Cache all simulation quantities at observed spatial locations.
        """
        flat_mask = self.mask.reshape(-1)

        if self.V.shape[1] != flat_mask.numel():
            raise ValueError(
                "V must have one column per flattened physical location. "
                f"V has {self.V.shape[1]} columns, while the observation "
                f"domain has {flat_mask.numel()} locations."
            )

        self.V_observed = self.V[:, flat_mask]

        self.Eb_mean_observed = (
            self.Eb_mean.reshape(-1)[flat_mask]
        )
        self.Eb_std_observed = (
            self.Eb_std.reshape(-1)[flat_mask]
        )
        self.Tpmp_observed = (
            self.Tpmp.reshape(-1)[flat_mask]
        )
        self.base_observed = (
            self.base.reshape(-1)[flat_mask]
            .to(
                dtype=self.Eb_star.dtype,
                device=self.Eb_star.device,
            )
        )


    def _simulated_obs(self, alpha):
        """
        Compute soft evidence only at observed locations.

        Returns
        -------
        torch.Tensor
            Simulated evidence with shape:

                (n_simulations, n_observations)
        """
        if alpha.shape != (self.n_modes,):
            raise ValueError(
                f"alpha must have shape ({self.n_modes},), "
                f"but received {tuple(alpha.shape)}."
            )

        scale = torch.exp(alpha)

        # Shape: (n_modes, n_simulations)
        Eb_star_corrected = (
            self.Eb_star * scale[:, None]
        )

        # Output shape:
        #     (n_observations, n_simulations)
        delta_T_observed = latent_operator_enthalpy(
            self.V_observed,
            Eb_star_corrected,
            self.Eb_mean_observed,
            self.Eb_std_observed,
            self.Tpmp_observed,
            self.method,
            self.epsilon,
        )

        simulated_evidence = binary_soft_operator(
            delta_T_observed,
            self.beta,
        )

        # Return shape:
        #     (n_simulations, n_observations)
        return simulated_evidence.T

    def _observed_simulated_evidence(self, alpha):
        Omega = self._simulated_obs(alpha)

        expected_shape = (
            self.n_simulations,
            self.n_obs,
        )

        if Omega.shape != expected_shape:
            raise ValueError(
                f"Expected Omega to have shape {expected_shape}, "
                f"but received {tuple(Omega.shape)}."
            )

        return Omega


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

        observed = self.base_observed
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
        max_eval_per_iter=10,
        tolerance_grad=1.0e-7,
        tolerance_change=1.0e-9,
        show_progress=True,
        use_line_search=True,
    ):
        """
        Solve the smooth elastic-net MAP problem using L-BFGS.

        Parameters
        ----------
        regularization_strength : float
            Overall elastic-net weight.
        sigma_sq : float or torch.Tensor
            Isotropic residual variance.
        l1_ratio : float, optional
            Elastic-net mixing parameter.
        smoothing_epsilon : float, optional
            Smoothing parameter for the approximate L1 penalty.
        alpha_ref : torch.Tensor, optional
            Reference correction used to construct the fixed covariance.
        rank_tolerance : float, optional
            Numerical rank tolerance for the covariance SVD.
        lr : float, optional
            L-BFGS learning-rate parameter.
        max_iter : int, optional
            Maximum number of accepted outer L-BFGS iterations.
        max_eval_per_iter : int, optional
            Approximate maximum number of closure evaluations per outer
            L-BFGS iteration.
        tolerance_grad : float, optional
            Gradient convergence tolerance.
        tolerance_change : float, optional
            Objective-change convergence tolerance.
        show_progress : bool, optional
            Display progress bars.
        use_line_search : bool, optional
            Use strong-Wolfe line search. Disabling it greatly reduces
            closure evaluations but can make optimization less robust.
        """
        self._initialize_alpha()
        self._prepare_observed_domain()

        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")

        if max_eval_per_iter < 1:
            raise ValueError("max_eval_per_iter must be at least 1.")

        # Compute the fixed covariance representation.
        self._prepare_fixed_metric(
            sigma_sq=sigma_sq,
            alpha_ref=alpha_ref,
            rank_tolerance=rank_tolerance,
        )

        optimizer = optim.LBFGS(
            [self.alpha],
            lr=lr,
            max_iter=1,
            max_eval=1,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn="strong_wolfe",
        )

        closure_evaluations = 0
        loss_history = []
        gradient_history = []
        previous_loss = None
        convergence_reason = "maximum iterations reached"

        iteration_bar = tqdm(
            total=max_iter,
            desc="L-BFGS iterations",
            unit="iter",
            position=0,
            disable=not show_progress,
        )

        evaluation_bar = tqdm(
            total=max_iter * max_eval_per_iter,
            desc="Objective evaluations",
            unit="eval",
            position=1,
            leave=False,
            disable=not show_progress,
        )

        def closure():
            nonlocal closure_evaluations

            optimizer.zero_grad(set_to_none=True)

            loss = self.objective(
                alpha=self.alpha,
                regularization_strength=regularization_strength,
                l1_ratio=l1_ratio,
                smoothing_epsilon=smoothing_epsilon,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "The optimization objective became non-finite. "
                    "Check sigma_sq, beta, alpha, and the outputs of the "
                    "observation operator."
                )

            loss.backward()

            closure_evaluations += 1
            evaluation_bar.update(1)
            evaluation_bar.set_postfix(
                loss=f"{loss.detach().item():.6e}",
                refresh=True,
            )

            return loss

        try:
            for iteration in range(max_iter):
                optimizer.step(closure)

                # Evaluate diagnostics at the accepted parameter value.
                # This forward evaluation is not counted as a line-search
                # closure evaluation.
                with torch.no_grad():
                    current_loss = self.objective(
                        alpha=self.alpha,
                        regularization_strength=regularization_strength,
                        l1_ratio=l1_ratio,
                        smoothing_epsilon=smoothing_epsilon,
                    )

                current_loss_value = current_loss.item()

                # Recompute the gradient at the accepted parameter value.
                # The last line-search closure is not guaranteed to have been
                # evaluated exactly at the final accepted alpha.
                optimizer.zero_grad(set_to_none=True)

                diagnostic_loss = self.objective(
                    alpha=self.alpha,
                    regularization_strength=regularization_strength,
                    l1_ratio=l1_ratio,
                    smoothing_epsilon=smoothing_epsilon,
                )
                diagnostic_loss.backward()

                gradient_norm = (
                    self.alpha.grad.detach()
                    .abs()
                    .max()
                    .item()
                )

                alpha_norm = self.alpha.detach().norm().item()

                if previous_loss is None:
                    loss_change = float("inf")
                else:
                    loss_change = abs(
                        previous_loss - current_loss_value
                    )

                loss_history.append(current_loss_value)
                gradient_history.append(gradient_norm)

                iteration_bar.update(1)
                iteration_bar.set_postfix(
                    loss=f"{current_loss_value:.6e}",
                    grad=f"{gradient_norm:.3e}",
                    dloss=f"{loss_change:.3e}",
                    alpha=f"{alpha_norm:.3e}",
                    evals=closure_evaluations,
                    refresh=True,
                )

                if (
                    math.isfinite(gradient_norm)
                    and gradient_norm <= tolerance_grad
                ):
                    convergence_reason = "gradient tolerance reached"
                    break

                if (
                    previous_loss is not None
                    and loss_change <= tolerance_change
                ):
                    convergence_reason = (
                        "objective-change tolerance reached"
                    )
                    break

                previous_loss = current_loss_value

        finally:
            iteration_bar.close()
            evaluation_bar.close()

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            final_loss = self.objective(
                alpha=self.alpha,
                regularization_strength=regularization_strength,
                l1_ratio=l1_ratio,
                smoothing_epsilon=smoothing_epsilon,
            )

            final_mahalanobis = self.mahalanobis(self.alpha)

            final_penalty = self.smooth_elastic_net(
                alpha=self.alpha,
                regularization_strength=regularization_strength,
                l1_ratio=l1_ratio,
                smoothing_epsilon=smoothing_epsilon,
            )

        result = {
            "alpha": self.alpha.detach().clone(),
            "scale": torch.exp(self.alpha.detach()).clone(),
            "loss": final_loss.detach().clone(),
            "mahalanobis": final_mahalanobis.detach().clone(),
            "regularization_penalty": final_penalty.detach().clone(),
            "covariance_rank": self.cov_rank,
            "singular_values": self.cov_singular_values.clone(),
            "logdet_S": self.logdet_S.clone(),
            "iterations": len(loss_history),
            "closure_evaluations": closure_evaluations,
            "loss_history": loss_history,
            "gradient_history": gradient_history,
            "convergence_reason": convergence_reason,
        }

        return result
