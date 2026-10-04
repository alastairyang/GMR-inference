import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from src.observationOperator import (
    latent_temperature_operator_enthalpy,
    latent_waterfraction_operator_enthalpy,
    temperature_binary_soft_operator,
    waterfraction_binary_soft_operator,
)


class PriorCorrection:
    """
    Estimate additive corrections in standardized PCA coordinates.

    Let lambda_i^j be the raw PCA coefficient for mode i and simulation j,
    and let nu_i denote the explained variance of mode i.

    The standardized PCA score is

        z_i^j = lambda_i^j / sqrt(nu_i).

    The corrected standardized score is

        z_i^{j,*} = z_i^j + alpha_i,

    which corresponds to the corrected raw coefficient

        lambda_i^{j,*}
            = lambda_i^j + sqrt(nu_i) * alpha_i.

    Therefore, alpha_i represents a modal mean shift measured in standard
    deviations of PCA mode i.

    The covariance of the simulated thermal-state evidence is computed once
    at a fixed reference alpha and held constant during optimization.
    """

    def __init__(
        self,
        gamma_wf=0.005,
        wf_threshold=0.02,
    ):
        if gamma_wf <= 0.0:
            raise ValueError("gamma_wf must be strictly positive.")

        self.gamma_wf = float(gamma_wf)
        self.wf_threshold = float(wf_threshold)

        # Full standardized correction vector after optimization.
        self.alpha = None

        # Reduced optimization vector and active-mode indices.
        self.alpha_free = None
        self.active_modes = None

        # PCA quantities.
        self.V = None
        self.Eb_star = None
        self.Eb_star_standardized = None
        self.explained_variance = None
        self.score_std = None

        # Fixed covariance SVD factors.
        self.cov_singular_values = None
        self.cov_Vh = None
        self.cov_rank = None

        # Quantities depending on fixed sigma_sq.
        self.sigma_sq = None
        self.woodbury_weights = None
        self.logdet_S = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_simulation_var(
        self,
        V,
        Eb_star,
        explained_variance,
        Eb_mean,
        Eb_std,
        Tpmp,
        method,
        epsilon,
    ):
        """
        Load PCA scores and physical reconstruction variables.

        Parameters
        ----------
        V : torch.Tensor
            PCA basis with shape

                (n_modes, n_locations).

            Each row is one retained PCA mode.

        Eb_star : torch.Tensor
            Raw PCA coefficients with shape

                (n_modes, n_simulations).

        explained_variance : torch.Tensor
            Explained variance for each retained PCA mode, with shape

                (n_modes,).

            The standardized scores are computed internally as

                Eb_star / sqrt(explained_variance).

        Eb_mean, Eb_std : torch.Tensor
            Quantities used to invert the spatial standardization of the
            basal enthalpy field.

        Tpmp : torch.Tensor
            Pressure-melting-point temperature field.

        method
            Standardization method passed to the latent observation
            operators.

        epsilon
            Relaxation constant passed to the latent observation operators.
        """
        if not torch.is_tensor(Eb_star):
            raise TypeError("Eb_star must be a torch.Tensor.")

        if not Eb_star.is_floating_point():
            raise TypeError("Eb_star must have a floating-point dtype.")

        if Eb_star.ndim != 2:
            raise ValueError(
                "Eb_star must have shape "
                "(n_modes, n_simulations)."
            )

        if V.ndim != 2:
            raise ValueError(
                "V must have shape (n_modes, n_locations)."
            )

        n_modes, n_simulations = Eb_star.shape

        if V.shape[0] != n_modes:
            raise ValueError(
                "The first dimension of V must equal the first "
                "dimension of Eb_star. "
                f"Received V.shape={tuple(V.shape)} and "
                f"Eb_star.shape={tuple(Eb_star.shape)}."
            )

        explained_variance = torch.as_tensor(
            explained_variance,
            dtype=Eb_star.dtype,
            device=Eb_star.device,
        )

        if explained_variance.ndim != 1:
            raise ValueError(
                "explained_variance must be one-dimensional."
            )

        if explained_variance.shape[0] != n_modes:
            raise ValueError(
                "explained_variance must contain one value per PCA mode. "
                f"Expected {n_modes} values but received "
                f"{explained_variance.shape[0]}."
            )

        if not torch.isfinite(explained_variance).all():
            raise ValueError(
                "explained_variance contains NaN or infinite values."
            )

        if torch.any(explained_variance <= 0.0):
            bad_modes = torch.nonzero(
                explained_variance <= 0.0,
                as_tuple=False,
            ).flatten()

            raise ValueError(
                "All retained explained variances must be strictly "
                f"positive. Invalid mode indices: {bad_modes.tolist()}."
            )

        if not torch.isfinite(Eb_star).all():
            raise ValueError(
                "Eb_star contains NaN or infinite values."
            )

        if not torch.isfinite(V).all():
            raise ValueError(
                "V contains NaN or infinite values."
            )

        self.V = V
        self.Eb_star = Eb_star

        self.explained_variance = (
            explained_variance.detach().clone()
        )

        # Standard deviation of each PCA score.
        self.score_std = torch.sqrt(
            self.explained_variance
        )

        # Standardized PCA scores. This is mostly useful for diagnostics.
        self.Eb_star_standardized = (
            self.Eb_star / self.score_std[:, None]
        )

        self.Eb_mean = Eb_mean
        self.Eb_std = Eb_std
        self.Tpmp = Tpmp
        self.method = method
        self.epsilon = epsilon

        self.n_modes = n_modes
        self.n_simulations = n_simulations

    def load_observation(self, base, beta=1.0):
        """
        Load observed basal thermal-state evidence.

        Parameters
        ----------
        base : torch.Tensor
            Observed basal thermal-state classifications:

                1   : thawed
                0   : frozen
                NaN : unobserved

            Spatial dimensions are flattened internally.

        beta : float, optional
            Temperature scale passed to the differentiable binary
            observation operator.
        gamma : float, optional
            Scale parameter passed to the differentiable binary
            water fraction observation operator.
        """
        if beta <= 0.0:
            raise ValueError("beta must be strictly positive.")

        self.base = base
        self.mask = ~torch.isnan(base)
        self.beta = float(beta)
        self.n_obs = int(self.mask.sum().item())

        if self.n_obs == 0:
            raise ValueError("No valid observations were provided.")

    # ------------------------------------------------------------------
    # Parameterization
    # ------------------------------------------------------------------

    def _initialize_alpha(self, active_modes=None):
        """
        Initialize standardized additive corrections.

        Parameters
        ----------
        active_modes : sequence of int, optional
            PCA modes allowed to change. If omitted, all retained PCA
            modes are optimized.

        Notes
        -----
        Inactive modes remain at alpha_i = 0.
        """
        if active_modes is None:
            active_modes = torch.arange(
                self.n_modes,
                dtype=torch.long,
                device=self.Eb_star.device,
            )
        else:
            active_modes = torch.as_tensor(
                active_modes,
                dtype=torch.long,
                device=self.Eb_star.device,
            )

        if active_modes.ndim != 1:
            raise ValueError(
                "active_modes must be one-dimensional."
            )

        if active_modes.numel() == 0:
            raise ValueError(
                "At least one active mode must be selected."
            )

        if (
            active_modes.min().item() < 0
            or active_modes.max().item() >= self.n_modes
        ):
            raise ValueError(
                "Mode indices must lie between 0 and "
                f"{self.n_modes - 1}."
            )

        if (
            torch.unique(active_modes).numel()
            != active_modes.numel()
        ):
            raise ValueError(
                "active_modes contains duplicate indices."
            )

        self.active_modes = active_modes

        self.alpha_free = torch.nn.Parameter(
            torch.zeros(
                active_modes.numel(),
                dtype=self.Eb_star.dtype,
                device=self.Eb_star.device,
            )
        )

    def _full_alpha(self):
        """
        Construct the full standardized correction vector.

        Inactive modes remain exactly zero.
        """
        if self.alpha_free is None:
            raise RuntimeError(
                "The optimization parameters have not been initialized."
            )

        alpha_full = torch.zeros(
            self.n_modes,
            dtype=self.alpha_free.dtype,
            device=self.alpha_free.device,
        )

        return alpha_full.index_copy(
            0,
            self.active_modes,
            self.alpha_free,
        )

    def _corrected_coefficients(self, alpha):
        """
        Convert standardized additive corrections into raw PCA scores.

        Parameters
        ----------
        alpha : torch.Tensor
            Standardized additive correction with shape (n_modes,).

        Returns
        -------
        torch.Tensor
            Corrected raw PCA coefficients with shape

                (n_modes, n_simulations).
        """
        if alpha.shape != (self.n_modes,):
            raise ValueError(
                f"alpha must have shape ({self.n_modes},), "
                f"but received {tuple(alpha.shape)}."
            )

        # alpha_i is measured in standard deviations of mode i.
        raw_coefficient_shift = (
            self.score_std * alpha
        )

        # The same modal mean shift is applied to every ensemble member.
        corrected = (
            self.Eb_star
            + raw_coefficient_shift[:, None]
        )

        return corrected

    # ------------------------------------------------------------------
    # Observation-domain preparation
    # ------------------------------------------------------------------

    def _prepare_observed_domain(self):
        """
        Cache reconstruction quantities at observed spatial locations.
        """
        if self.V is None:
            raise RuntimeError(
                "Simulation variables must be loaded before solve_MAP."
            )

        if not hasattr(self, "mask"):
            raise RuntimeError(
                "Observations must be loaded before solve_MAP."
            )

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
            .to(
                dtype=self.Eb_star.dtype,
                device=self.Eb_star.device,
            )
        )

        self.Eb_std_observed = (
            self.Eb_std.reshape(-1)[flat_mask]
            .to(
                dtype=self.Eb_star.dtype,
                device=self.Eb_star.device,
            )
        )

        self.Tpmp_observed = (
            self.Tpmp.reshape(-1)[flat_mask]
            .to(
                dtype=self.Eb_star.dtype,
                device=self.Eb_star.device,
            )
        )

        self.base_observed = (
            self.base.reshape(-1)[flat_mask]
            .to(
                dtype=self.Eb_star.dtype,
                device=self.Eb_star.device,
            )
        )

    # ------------------------------------------------------------------
    # Forward operators
    # ------------------------------------------------------------------

    def _simulated_obs(self, alpha):
        """
        Compute soft thermal-state evidence at observed locations.

        Returns
        -------
        torch.Tensor
            Simulated evidence with shape

                (n_simulations, n_observations).
        """
        Eb_star_corrected = self._corrected_coefficients(alpha)

        # Expected output shape:
        #     (n_observations, n_simulations)
        delta_T_observed = latent_temperature_operator_enthalpy(
            self.V_observed,
            Eb_star_corrected,
            self.Eb_mean_observed,
            self.Eb_std_observed,
            self.Tpmp_observed,
            self.method,
            self.epsilon,
        )

        simulated_evidence = temperature_binary_soft_operator(
            delta_T_observed,
            self.beta,
        )

        # Return:
        #     (n_simulations, n_observations)
        return simulated_evidence.T

    def _simulated_waterfraction(self, alpha):
        """
        Compute physical water fraction over the full domain.

        Returns
        -------
        torch.Tensor
            Simulated water-fraction fields. The precise dimensions
            depend on the latent observation operator, but should include
            simulations and spatial locations.
        """
        Eb_star_corrected = self._corrected_coefficients(alpha)

        water_fraction = latent_waterfraction_operator_enthalpy(
            self.V,
            Eb_star_corrected,
            self.Eb_mean,
            self.Eb_std,
            self.Tpmp,
            self.method,
            self.epsilon,
        )

        return water_fraction

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

    # ------------------------------------------------------------------
    # Fixed covariance metric
    # ------------------------------------------------------------------

    def _covariance_matrix(
        self,
        alpha_ref=None,
        rank_tolerance=None,
    ):
        """
        Compute the compact SVD of the fixed centered evidence matrix.

        The centered reference evidence is

            C_ref = (Omega_ref - mean(Omega_ref)) / sqrt(K - 1).

        The covariance is represented implicitly as

            S = C_ref.T @ C_ref + sigma_sq * I.

        Parameters
        ----------
        alpha_ref : torch.Tensor, optional
            Reference standardized additive correction. The default is
            alpha_ref = 0.

        rank_tolerance : float, optional
            Singular values below this threshold are discarded.
        """
        if self.n_simulations < 2:
            raise ValueError(
                "At least two simulations are required to estimate "
                "the covariance."
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

        with torch.no_grad():
            Omega_ref = self._observed_simulated_evidence(
                alpha_ref
            )

            mean_ref = Omega_ref.mean(
                dim=0,
                keepdim=True,
            )

            C_ref = (
                Omega_ref - mean_ref
            ) / math.sqrt(self.n_simulations - 1)

            _, singular_values, Vh = torch.linalg.svd(
                C_ref,
                full_matrices=False,
            )

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
                else:
                    rank_tolerance = torch.as_tensor(
                        rank_tolerance,
                        dtype=C_ref.dtype,
                        device=C_ref.device,
                    )

                keep = singular_values > rank_tolerance

            self.cov_singular_values = (
                singular_values[keep].detach()
            )

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
        Precompute fixed covariance quantities for the likelihood.
        """
        sigma_sq = torch.as_tensor(
            sigma_sq,
            dtype=self.Eb_star.dtype,
            device=self.Eb_star.device,
        )

        if sigma_sq.numel() != 1:
            raise ValueError("sigma_sq must be a scalar.")

        sigma_sq = sigma_sq.reshape(())

        if not torch.isfinite(sigma_sq):
            raise ValueError("sigma_sq must be finite.")

        if sigma_sq.item() <= 0.0:
            raise ValueError(
                "sigma_sq must be strictly positive."
            )

        self._covariance_matrix(
            alpha_ref=alpha_ref,
            rank_tolerance=rank_tolerance,
        )

        self.sigma_sq = sigma_sq.detach()

        s_sq = self.cov_singular_values.square()

        self.woodbury_weights = (
            s_sq
            / (
                self.sigma_sq
                * (self.sigma_sq + s_sq)
            )
        ).detach()

        self.logdet_S = (
            self.n_evidence * torch.log(self.sigma_sq)
            + torch.log1p(
                s_sq / self.sigma_sq
            ).sum()
        ).detach()

    def _apply_precision(self, vector):
        """
        Compute S^{-1} vector using the fixed SVD representation.
        """
        precision_vector = vector / self.sigma_sq

        if self.cov_rank > 0:
            projected = self.cov_Vh @ vector

            precision_vector = (
                precision_vector
                - self.cov_Vh.T
                @ (
                    self.woodbury_weights
                    * projected
                )
            )

        return precision_vector

    def mahalanobis(self, alpha):
        """
        Compute the fixed-covariance Mahalanobis distance.
        """
        if self.sigma_sq is None:
            raise RuntimeError(
                "The fixed covariance metric has not been prepared."
            )

        Omega = self._observed_simulated_evidence(alpha)

        simulated_mean = Omega.mean(dim=0)
        residual = self.base_observed - simulated_mean

        precision_residual = self._apply_precision(
            residual
        )

        quadratic = torch.dot(
            residual,
            precision_residual,
        )

        # Tiny negative values can occur through floating-point
        # cancellation in the Woodbury expression.
        if (
            not quadratic.requires_grad
            and quadratic.item() < 0.0
            and quadratic.item() > -1.0e-10
        ):
            quadratic = quadratic.clamp_min(0.0)

        return quadratic

    def log_likelihood(
        self,
        alpha,
        include_constants=True,
    ):
        """
        Compute the fixed-covariance Gaussian working log-likelihood.
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

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------

    @staticmethod
    def smooth_elastic_net(
        alpha,
        regularization_strength,
        l1_ratio=0.9,
        smoothing_epsilon=1.0e-6,
    ):
        """
        Elastic-net penalty on standardized additive corrections.

        Because alpha is expressed in PCA standard deviations, the same
        regularization weight has a comparable interpretation across modes.
        """
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError(
                "l1_ratio must lie between 0 and 1."
            )

        if smoothing_epsilon <= 0.0:
            raise ValueError(
                "smoothing_epsilon must be strictly positive."
            )

        gamma = torch.as_tensor(
            regularization_strength,
            dtype=alpha.dtype,
            device=alpha.device,
        )

        if gamma.numel() != 1:
            raise ValueError(
                "regularization_strength must be scalar."
            )

        if gamma.item() < 0.0:
            raise ValueError(
                "regularization_strength cannot be negative."
            )

        eta = alpha.new_tensor(l1_ratio)
        eps = alpha.new_tensor(smoothing_epsilon)

        smooth_l1 = (
            torch.sqrt(
                alpha.square() + eps.square()
            )
            - eps
        ).sum()

        l2 = 0.5 * alpha.square().sum()

        return gamma * (
            eta * smooth_l1
            + (1.0 - eta) * l2
        )

    def water_fraction_penalty(
        self,
        alpha,
        regularization_strength,
    ):
        """
        Compute a stable, averaged water-fraction barrier.

        The pointwise penalty is

            softplus((phi - phi_max) / gamma_wf).

        It is small below the water-fraction threshold and grows
        approximately linearly above the threshold.

        The penalty is averaged rather than summed so that its magnitude
        is less sensitive to ensemble size and grid resolution.
        """
        rho_w = torch.as_tensor(
            regularization_strength,
            dtype=alpha.dtype,
            device=alpha.device,
        )

        if rho_w.numel() != 1:
            raise ValueError(
                "Water-fraction regularization strength must be scalar."
            )

        if rho_w.item() < 0.0:
            raise ValueError(
                "Water-fraction regularization strength cannot be negative."
            )

        water_fraction = self._simulated_waterfraction(
            alpha
        )

        normalized_exceedance = (
            water_fraction - self.wf_threshold
        ) / self.gamma_wf

        pointwise_penalty = F.softplus(
            normalized_exceedance
        )

        return rho_w * pointwise_penalty.mean()

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def objective_components(
        self,
        alpha,
        rho_e,
        rho_w,
        l1_ratio,
        smoothing_epsilon,
    ):
        """
        Return each component of the reduced MAP objective.
        """
        data_misfit = 0.5 * self.mahalanobis(alpha)

        elastic_penalty = self.smooth_elastic_net(
            alpha=alpha,
            regularization_strength=rho_e,
            l1_ratio=l1_ratio,
            smoothing_epsilon=smoothing_epsilon,
        )

        water_penalty = self.water_fraction_penalty(
            alpha=alpha,
            regularization_strength=rho_w,
        )

        return {
            "data_misfit": data_misfit,
            "elastic_penalty": elastic_penalty,
            "water_penalty": water_penalty,
        }

    def objective(
        self,
        alpha,
        rho_e,
        rho_w,
        l1_ratio,
        smoothing_epsilon,
    ):
        """
        Reduced MAP objective.
        """
        components = self.objective_components(
            alpha=alpha,
            rho_e=rho_e,
            rho_w=rho_w,
            l1_ratio=l1_ratio,
            smoothing_epsilon=smoothing_epsilon,
        )

        return (
            components["data_misfit"]
            + components["elastic_penalty"]
            + components["water_penalty"]
        )

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def solve_MAP(
        self,
        rho_e,
        rho_w,
        sigma_sq,
        l1_ratio=0.9,
        smoothing_epsilon=1.0e-6,
        alpha_ref=None,
        rank_tolerance=None,
        lr=0.1,
        max_iter=100,
        max_eval_per_iter=20,
        tolerance_grad=1.0e-7,
        tolerance_change=1.0e-9,
        show_progress=True,
        active_modes=None,
        use_line_search=True,
    ):
        """
        Solve for additive standardized PCA corrections using L-BFGS.

        Parameters
        ----------
        rho_e : float
            Elastic-net regularization strength.

        rho_w : float
            Water-fraction regularization strength.

        sigma_sq : float or torch.Tensor
            Isotropic residual variance in the fixed covariance model.

        l1_ratio : float, optional
            Elastic-net mixing parameter.

        smoothing_epsilon : float, optional
            Smoothing scale for the approximate L1 term.

        alpha_ref : torch.Tensor, optional
            Reference standardized additive correction used to construct
            the fixed evidence covariance. The default is zero.

        rank_tolerance : float, optional
            Numerical rank tolerance for the evidence SVD.

        lr : float, optional
            Initial L-BFGS step size.

        max_iter : int, optional
            Maximum number of accepted outer L-BFGS iterations.

        max_eval_per_iter : int, optional
            Maximum number of closure evaluations allowed for each outer
            L-BFGS iteration.

        tolerance_grad : float, optional
            Maximum-gradient convergence tolerance.

        tolerance_change : float, optional
            Objective-change convergence tolerance.

        show_progress : bool, optional
            Display progress bars.

        active_modes : sequence of int, optional
            Modes allowed to change. If omitted, all retained PCA modes
            are optimized.

        use_line_search : bool, optional
            Use strong-Wolfe line search.
        """
        if self.Eb_star is None:
            raise RuntimeError(
                "Call load_simulation_var before solve_MAP."
            )

        if not hasattr(self, "base"):
            raise RuntimeError(
                "Call load_observation before solve_MAP."
            )

        if max_iter < 1:
            raise ValueError(
                "max_iter must be at least 1."
            )

        if max_eval_per_iter < 1:
            raise ValueError(
                "max_eval_per_iter must be at least 1."
            )

        if lr <= 0.0:
            raise ValueError(
                "lr must be strictly positive."
            )

        self._initialize_alpha(
            active_modes=active_modes
        )

        self._prepare_observed_domain()

        self._prepare_fixed_metric(
            sigma_sq=sigma_sq,
            alpha_ref=alpha_ref,
            rank_tolerance=rank_tolerance,
        )

        line_search_fn = (
            "strong_wolfe"
            if use_line_search
            else None
        )

        optimizer = optim.LBFGS(
            [self.alpha_free],
            lr=lr,
            max_iter=1,
            max_eval=max_eval_per_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn=line_search_fn,
        )

        closure_evaluations = 0
        loss_history = []
        data_misfit_history = []
        elastic_penalty_history = []
        water_penalty_history = []
        gradient_history = []
        alpha_norm_history = []

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

            alpha_full = self._full_alpha()

            loss = self.objective(
                alpha=alpha_full,
                rho_e=rho_e,
                rho_w=rho_w,
                l1_ratio=l1_ratio,
                smoothing_epsilon=smoothing_epsilon,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "The objective became non-finite. Check the "
                    "observation operators, sigma_sq, beta, gamma_wf, "
                    "and PCA inputs."
                )

            loss.backward()

            if self.alpha_free.grad is None:
                raise RuntimeError(
                    "No gradient was produced for alpha_free."
                )

            if not torch.isfinite(
                self.alpha_free.grad
            ).all():
                raise RuntimeError(
                    "The alpha gradient became non-finite."
                )

            closure_evaluations += 1

            evaluation_bar.update(1)
            evaluation_bar.set_postfix(
                loss=f"{loss.detach().item():.6e}",
                refresh=True,
            )

            return loss

        try:
            for _ in range(max_iter):
                optimizer.step(closure)

                # Evaluate all components at the accepted point.
                with torch.no_grad():
                    alpha_full = self._full_alpha()

                    components = self.objective_components(
                        alpha=alpha_full,
                        rho_e=rho_e,
                        rho_w=rho_w,
                        l1_ratio=l1_ratio,
                        smoothing_epsilon=smoothing_epsilon,
                    )

                    current_loss = (
                        components["data_misfit"]
                        + components["elastic_penalty"]
                        + components["water_penalty"]
                    )

                current_loss_value = current_loss.item()
                data_value = components["data_misfit"].item()
                elastic_value = components[
                    "elastic_penalty"
                ].item()
                water_value = components[
                    "water_penalty"
                ].item()

                # Recompute the gradient at the accepted point.
                optimizer.zero_grad(set_to_none=True)

                alpha_full = self._full_alpha()

                diagnostic_loss = self.objective(
                    alpha=alpha_full,
                    rho_e=rho_e,
                    rho_w=rho_w,
                    l1_ratio=l1_ratio,
                    smoothing_epsilon=smoothing_epsilon,
                )

                diagnostic_loss.backward()

                if self.alpha_free.grad is None:
                    raise RuntimeError(
                        "No diagnostic gradient was produced."
                    )

                gradient_norm = (
                    self.alpha_free.grad
                    .detach()
                    .abs()
                    .max()
                    .item()
                )

                alpha_norm = (
                    alpha_full.detach().norm().item()
                )

                if previous_loss is None:
                    loss_change = float("inf")
                else:
                    loss_change = abs(
                        previous_loss
                        - current_loss_value
                    )

                loss_history.append(current_loss_value)
                data_misfit_history.append(data_value)
                elastic_penalty_history.append(
                    elastic_value
                )
                water_penalty_history.append(
                    water_value
                )
                gradient_history.append(gradient_norm)
                alpha_norm_history.append(alpha_norm)

                iteration_bar.update(1)
                iteration_bar.set_postfix(
                    loss=f"{current_loss_value:.6e}",
                    data=f"{data_value:.3e}",
                    water=f"{water_value:.3e}",
                    elastic=f"{elastic_value:.3e}",
                    grad=f"{gradient_norm:.3e}",
                    alpha=f"{alpha_norm:.3e}",
                    refresh=True,
                )

                if (
                    math.isfinite(gradient_norm)
                    and gradient_norm <= tolerance_grad
                ):
                    convergence_reason = (
                        "gradient tolerance reached"
                    )
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

        # --------------------------------------------------------------
        # Final results and physical diagnostics
        # --------------------------------------------------------------

        with torch.no_grad():
            final_alpha = self._full_alpha()
            self.alpha = final_alpha.detach().clone()

            final_components = self.objective_components(
                alpha=final_alpha,
                rho_e=rho_e,
                rho_w=rho_w,
                l1_ratio=l1_ratio,
                smoothing_epsilon=smoothing_epsilon,
            )

            final_loss = (
                final_components["data_misfit"]
                + final_components["elastic_penalty"]
                + final_components["water_penalty"]
            )

            final_mahalanobis = self.mahalanobis(
                final_alpha
            )

            final_corrected_coefficients = (
                self._corrected_coefficients(
                    final_alpha
                )
            )

            raw_coefficient_shift = (
                self.score_std * final_alpha
            )

            final_water_fraction = (
                self._simulated_waterfraction(
                    final_alpha
                )
            )

            exceedance = torch.relu(
                final_water_fraction
                - self.wf_threshold
            )

            fraction_above_threshold = (
                final_water_fraction
                > self.wf_threshold
            ).to(
                dtype=self.Eb_star.dtype
            ).mean()

            result = {
                # Standardized additive correction.
                "alpha": self.alpha.clone(),
                "alpha_free": (
                    self.alpha_free.detach().clone()
                ),
                "active_modes": (
                    self.active_modes.detach().clone()
                ),

                # Equivalent raw PCA coefficient shift.
                "raw_coefficient_shift": (
                    raw_coefficient_shift.clone()
                ),
                "corrected_coefficients": (
                    final_corrected_coefficients.clone()
                ),

                # PCA scaling information.
                "explained_variance": (
                    self.explained_variance.clone()
                ),
                "score_std": self.score_std.clone(),

                # Final objective values.
                "loss": final_loss.detach().clone(),
                "data_misfit": (
                    final_components[
                        "data_misfit"
                    ].detach().clone()
                ),
                "mahalanobis": (
                    final_mahalanobis.detach().clone()
                ),
                "regularization_penalty_e": (
                    final_components[
                        "elastic_penalty"
                    ].detach().clone()
                ),
                "regularization_penalty_w": (
                    final_components[
                        "water_penalty"
                    ].detach().clone()
                ),

                # Water-fraction diagnostics.
                "water_fraction_max": (
                    final_water_fraction.max()
                    .detach()
                    .clone()
                ),
                "water_fraction_mean": (
                    final_water_fraction.mean()
                    .detach()
                    .clone()
                ),
                "fraction_above_wf_threshold": (
                    fraction_above_threshold
                    .detach()
                    .clone()
                ),
                "mean_wf_exceedance": (
                    exceedance.mean()
                    .detach()
                    .clone()
                ),
                "max_wf_exceedance": (
                    exceedance.max()
                    .detach()
                    .clone()
                ),

                # Covariance information.
                "covariance_rank": self.cov_rank,
                "singular_values": (
                    self.cov_singular_values.clone()
                ),
                "logdet_S": self.logdet_S.clone(),

                # Optimization diagnostics.
                "iterations": len(loss_history),
                "closure_evaluations": (
                    closure_evaluations
                ),
                "loss_history": loss_history,
                "data_misfit_history": (
                    data_misfit_history
                ),
                "elastic_penalty_history": (
                    elastic_penalty_history
                ),
                "water_penalty_history": (
                    water_penalty_history
                ),
                "gradient_history": gradient_history,
                "alpha_norm_history": alpha_norm_history,
                "convergence_reason": convergence_reason,
            }

        return result
