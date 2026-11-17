import numpy as np

class RegressPosterior:
    """
    Class for performing regression on the posterior samples using Gaussian mixture models.
    
    Author: Donglai Yang
    Contact: dyang379@gatech.edu
    """
    
    def __init__(
            self, 
            simulation, 
            forward_model,
            posterior,
            predictor_index,
            response_index,
            ):
        self.simulation = simulation # the Simulation class object
        self.predictor_index = predictor_index # the index of the predictor variable (posterior sample index)
        self.posterior = posterior # the posterior class (enthalpyPosterior)
        self.forward_model = forward_model # the forward model class (latentEnthalpyModel)
        self.response_index = response_index # the index of the response variable
        self.joint_md = None
    def form_joint_distribution(self, n_gaussian):
        field_index = [[self.predictor_index, self.response_index]]
        self.joint_md = self.simulation.fit_joint_gmm(
            n_gaussians = [n_gaussian],
            index = field_index
        )

        self._regression_indices()
        return
    def test_regression(
            self, 
            n_sample_plot=5, 
            n_sample_uq = 200, 
            savepath=None,
            y_name='Predictor', 
            x_name='Response'
        ):
        """    
        Perform Gaussian mixture regression on TEST set
        using the formed joint distribution and visualize the results.

        X and Y naming convention adopts the conditional probability perspective,
        meaning P(X|Y) where X is the response and Y is the predictor.
        """

        import matplotlib.pyplot as plt

        if self.joint_md is None:
            raise ValueError("Joint distribution has not been formed.")
        
        sims = self.simulation
        n_test_samples_plot = n_sample_plot
        rand_idx = np.random.choice(range(self.simulation.n_samples_test), size=n_test_samples_plot, replace=False)
        plt.figure(figsize=(26, 5 * n_test_samples_plot))

        title_fontsize = 18
        error_cmap = 'viridis'
        uq_cmap    = 'plasma'
        extent_km = [s / 1000 for s in sims.extent]

        for count, i in enumerate(rand_idx):
            response_test = self.simulation.X_test[self.response_index][i, :]  
            predictor_test = self.simulation.X_test[self.predictor_index][i, :] 

            # conditioning
            x_pred_gmm = self.joint_md[0].condition(self.condition_index, predictor_test)

            # sample from this conditional distribution to get uncertainty
            x_uq_samples = x_pred_gmm.sample(n_sample_uq)
            x_uq_samples_ori = np.zeros((self.simulation.nx * self.simulation.ny, n_sample_uq))
            # Inverse transform to original space
            # then the uq samples
            for j, sample in enumerate(x_uq_samples):
                x_uq_samples_ori[:,j] = self.simulation.pca_x[self.response_index].inverse_transform(sample.reshape(1, -1))

            # compute the mean from the ensemble
            x_pred_mean = np.mean(x_uq_samples_ori, axis=1)
            x_pred_img = x_pred_mean.reshape(self.simulation.nx, self.simulation.ny)
            
            # compute std along each dimension of the uq samples
            x_uq_std = np.std(x_uq_samples_ori, axis=1)
            x_uq_std = x_uq_std.reshape(self.simulation.nx, self.simulation.ny)

            # compute absolute error
            abs_error_img = np.abs(x_pred_img - 
                                   sims.pca_x[self.response_index].inverse_transform(response_test.reshape(1, -1)).reshape(sims.nx, sims.ny))

            # X ground truth
            x_test_gt = sims.pca_x[self.response_index].inverse_transform(
                response_test.reshape(1, -1))
            x_test_img = x_test_gt.reshape(sims.nx, sims.ny)

            # observed Y (in this case the predictor, basal enthalpy)
            obs_Y = sims.pca_x[self.predictor_index].inverse_transform(predictor_test.reshape(1, -1))
            obs_Y = obs_Y.reshape(sims.nx, sims.ny)

            # mask out the non-domain area
            obs_Y      = np.where(sims.domain_mask, obs_Y, np.nan)
            x_test_img = np.where(sims.domain_mask, x_test_img, np.nan)
            x_pred_img = np.where(sims.domain_mask, x_pred_img, np.nan)
            x_uq_std = np.where(sims.domain_mask, x_uq_std, np.nan)
            abs_error_img = np.where(sims.domain_mask, abs_error_img, np.nan)

            # Plotting: five columns: observed Y, True X, predicted X, error (RMSE), uncertainty (stddev)
            plt.subplot(n_test_samples_plot, 5, count * 5 + 1)
            plt.imshow(obs_Y, cmap='RdBu', vmin=-2, vmax=2, 
                       extent=extent_km,
                       origin="lower")
            plt.title(rf'Observed {y_name}' + "\n" + f'(Sample {i+1})', fontsize=title_fontsize)
            plt.colorbar()
            
            plt.subplot(n_test_samples_plot, 5, count * 5 + 2)
            plt.imshow(x_pred_img, cmap='RdBu', vmin=-2, vmax=2, 
                       origin="lower", extent=extent_km)
            plt.title(rf'Predicted {x_name}' + "\n" + f'(Sample {i+1})', fontsize=title_fontsize)
            plt.colorbar()

            plt.subplot(n_test_samples_plot, 5, count * 5 + 3)
            plt.imshow(x_test_img, cmap='RdBu', vmin=-2, vmax=2, 
                       origin="lower", extent=extent_km)
            plt.title(rf'True {x_name}' + "\n" + f'(Sample {i+1})', fontsize=title_fontsize)
            plt.colorbar()

            plt.subplot(n_test_samples_plot, 5, count * 5 + 4)
            plt.imshow(abs_error_img, cmap=error_cmap, vmin = 0, vmax = 2, 
                       origin="lower", extent=extent_km)
            plt.title('Abs Error', fontsize=title_fontsize)
            plt.colorbar()

            plt.subplot(n_test_samples_plot, 5, count * 5 + 5)
            plt.imshow(x_uq_std, cmap=uq_cmap, vmin = 0, vmax = 1, 
                       origin="lower", extent=extent_km)
            plt.title('Uncertainty ($\sigma$)', fontsize=title_fontsize)
            plt.colorbar()

        # Save the individual-sample figure using the method argument.
        if savepath:
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.show()

        # Aggregate validation metrics across ALL test samples.
        response_pca = sims.pca_x[self.response_index]

        n_test = sims.n_samples_test
        if n_test < 1:
            raise ValueError("No test samples available.")

        n_pixels = sims.nx * sims.ny
        squared_error_sum = np.zeros(n_pixels, dtype=float)
        conditional_std_sum = np.zeros(n_pixels, dtype=float)

        for i in range(n_test):
            predictor_test = sims.X_test[self.predictor_index][i, :]
            response_test = sims.X_test[self.response_index][i, :]

            conditional = self.joint_md[0].condition(
                self.condition_index,
                predictor_test,
            )

            response_draws = np.asarray(
                conditional.sample(n_sample_uq)
            ).reshape(n_sample_uq, -1)

            # Shape: (n_sample_uq, n_pixels).
            # These remain in standardized field space, matching
            # the individual-sample plots above.
            reconstructed = response_pca.inverse_transform(
                response_draws
            )

            predicted_mean = reconstructed.mean(axis=0)
            predicted_std = reconstructed.std(axis=0)

            true_response = response_pca.inverse_transform(
                response_test.reshape(1, -1)
            ).ravel()

            squared_error_sum += (true_response - predicted_mean) ** 2
            conditional_std_sum += predicted_std

            # Release the large reconstruction before the next iteration.
            del reconstructed

        # Pixelwise RMSE across test cases, not mean absolute error.
        rmse_map = np.sqrt(
            squared_error_sum / n_test
        ).reshape(sims.nx, sims.ny)

        # Average conditional std across test cases.
        mean_std_map = (
            conditional_std_sum / n_test
        ).reshape(sims.nx, sims.ny)

        fig, axes = plt.subplots(
            1, 2,
            figsize=(12, 5),
            constrained_layout=True,
        )

        # mask out the non-domain part
        rmse_map     = np.where(sims.domain_mask, rmse_map, np.nan)
        mean_std_map = np.where(sims.domain_mask, mean_std_map, np.nan)

        panels = [
            (rmse_map, "Test-set RMSE", 1, error_cmap),
            (mean_std_map, "Mean conditional standard deviation", 1, uq_cmap),
        ]


        for ax, (values, title, vmax, cmap) in zip(axes, panels):
            image = ax.imshow(
                values,
                origin="lower",
                cmap=cmap,
                vmin=0,
                vmax=vmax,
                extent=extent_km,
            )
            ax.set_title(title)
            ax.set_xlabel("X (km)")
            ax.set_ylabel("Y (km)")
            fig.colorbar(
                image,
                ax=ax,
                label="Standardized response units",
            )

        fig.suptitle(f"Accuracy and uncertainty across {n_test} test samples", fontsize=20)

        if savepath:
            from pathlib import Path

            path = Path(savepath)
            aggregate_path = path.with_name(
                f"{path.stem}_aggregate{path.suffix or '.png'}"
            )
            fig.savefig(
                aggregate_path,
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

        # return {
        #     "rmse": rmse_map,
        #     "mean_conditional_std": mean_std_map,
        #     "n_test_samples": n_test,
        # }
        return

    def regress_posterior(
            self, 
            posterior_samples,
            n_conditional_samples=100,
            random_state=0,
            num_sample_save=0,
            max_working_mb=2048
        ):
        """
        Perform Gaussian Mixture regression on all the latent posterior samples. 
        This will enough projecting on to the new latent space defined
        by the simulations, which may be lower dimensional than the original space used
        in the Bayesian state estimation. 
        """
        sims = self.simulation

        sim_ndim = sims.pca_x[self.predictor_index].n_components
        if posterior_samples.shape[1] < sim_ndim:
            raise ValueError(
                "Posterior samples have fewer dimensions than the simulation latent space."
            )

        posterior_sample_newpc = self._project_to_sim_latent(posterior_samples)

        z = np.asarray(posterior_sample_newpc)

        if z.ndim != 2 or z.shape[0] == 0:
            raise ValueError("Expected nonempty enthalpy coefficients of shape (N, M).")
        if z.shape[1] != len(self.condition_index):
            raise ValueError("Enthalpy PC count does not match conditioning indices.")

        if (
            not isinstance(n_conditional_samples, (int, np.integer))
            or n_conditional_samples < 1
        ):
            raise ValueError("n_conditional_samples must be a positive integer.")

        if (
            not isinstance(num_sample_save, (int, np.integer))
            or num_sample_save < 0
        ):
            raise ValueError("num_sample_save must be a nonnegative integer.")

        # for the response variable
        pca = sims.pca_x[self.response_index]
        scaler = sims.scaling_models[self.response_index]
        n_pcs, n_pixels = pca.components_.shape

        if n_pixels != sims.nx * sims.ny or scaler.n_features_in_ != n_pixels:
            raise ValueError("PCA, scaler, and grid dimensions do not match.")

        # Calculate the REAL ensemble size before checking memory.
        n_draws = z.shape[0] * n_conditional_samples
        n_saved = min(num_sample_save, n_draws)

        # ------  Budget includes coefficients, output maps, saved fields -------
        budget = int(max_working_mb * 1024**2)

        base_bytes = 8 * (
            n_draws * n_pcs
            + (5 + n_saved) * n_pixels
        )

        # Conservative allowance for reconstructed blocks and temporaries.
        bytes_per_pixel = 8 * (5 * n_draws + n_pcs)
        available_bytes = int(0.8 * budget) - base_bytes

        chunk_size = min(
            n_pixels,
            2048,
            max(0, available_bytes // bytes_per_pixel),
        )

        if chunk_size < 1:
            raise MemoryError(
                "Estimated working memory exceeds the budget. "
                "Reduce sample counts or increase max_working_mb."
            )

        print(
            f"{n_draws:,} draws; "
            f"{chunk_size} pixels per reconstruction block"
        )
        # ---------------------------------------------------------------------


        # Allocate only after passing the memory check.
        rng = np.random.default_rng(random_state)
        save_indices = rng.choice(n_draws, size=n_saved, replace=False)

        coefficients = np.empty((n_draws, n_pcs))
        saved_samples = np.empty((n_saved, n_pixels))

        # Multiple conditional GHF draws for EVERY enthalpy sample.
        for i, enthalpy_sample in enumerate(z):
            conditional = sims.joint_gmms[0].condition(
                self.condition_index,
                enthalpy_sample,
            )
            draws = np.asarray(conditional.sample(n_conditional_samples))

            if draws.shape != (n_conditional_samples, n_pcs):
                raise ValueError("Conditional GMM output does not match GHF PCA.")

            start = i * n_conditional_samples
            coefficients[start:start + n_conditional_samples] = draws

        # Match PCA.inverse_transform if whitening was used.
        if getattr(pca, "whiten", False):
            coefficients *= np.sqrt(pca.explained_variance_)[None, :]

        results = {
            key: np.empty(n_pixels)
            for key in ("mean", "std", "p05", "p95", "interval_width")
        }

        for start in range(0, n_pixels, chunk_size):
            stop = min(start + chunk_size, n_pixels)
            sl = slice(start, stop)

            # Blockwise PCA inverse transform.
            block = coefficients @ pca.components_[:, sl]
            block += pca.mean_[sl]

            # Equivalent to scaler.inverse_transform, applied blockwise.
            if scaler.with_std:
                block *= scaler.scale_[sl]
            if scaler.with_mean:
                block += scaler.mean_[sl]

            # Save physical-unit realizations before quantile modifies block.
            if n_saved:
                saved_samples[:, sl] = block[save_indices, :]

            results["mean"][sl] = block.mean(axis=0)
            results["std"][sl] = block.std(axis=0)

            q05, q95 = np.quantile(
                block,
                [0.05, 0.95],
                axis=0,
                overwrite_input=True,
            )
            results["p05"][sl] = q05
            results["p95"][sl] = q95
            results["interval_width"][sl] = q95 - q05

            del block, q05, q95

        # Return only AFTER all blocks have been processed.
        output = {
            key: values.reshape(sims.nx, sims.ny)
            for key, values in results.items()
        }
        output["samples"] = saved_samples.reshape(n_saved, sims.nx, sims.ny)
        output["n_samples_saved"] = n_saved
        output["n_samples_drawn"] = n_draws

        return output
    def plot_predictions(
        self,
        prediction,
        field_name="GHF",
        units=None,
        save_path=None,
        value_limits=None,
    ):
        """Plot response variable mean, pointwise interval bounds, and interval width."""
        import matplotlib.pyplot as plt

        extent_km = [x / 1e3 for x in self.simulation.extent]

        mean = prediction["mean"]
        width = prediction["interval_width"]
        p05 = prediction["p05"]
        p95 = prediction["p95"]

        if value_limits is None:
            vmin = min(p05.min(), mean.min(), p95.min())
            vmax = max(p05.max(), mean.max(), p95.max())
        else:
            vmin, vmax = value_limits

        colorbar_label = (
            f"{field_name} ({units})" if units else field_name
        )

        panels = [
            (mean, f"{field_name}: posterior mean"),
        ]

        figsize = 5 * (len(panels) + 1)
        fig, axes = plt.subplots(
            1, len(panels)+1,
            figsize=(figsize, 5),
            constrained_layout=True,
        )

        for ax, (field, title) in zip(axes[:2], panels):
            # apply simulation mask
            field = np.where(self.simulation.domain_mask, field, np.nan)
            image = ax.imshow(
                field * 1e3,
                origin="lower",
                cmap="Reds",
                vmin=vmin,
                vmax=vmax,
                extent=extent_km,
            )
            ax.set_title(title)
            ax.set_xlabel("X (km)")
            ax.set_ylabel("Y (km)")
            # colorbar
            fig.colorbar(
                image,
                ax=ax,
                shrink=0.85,
                label=colorbar_label,
            )
        # fig.colorbar(
        #     image,
        #     ax=list(axes[:2]),
        #     shrink=0.85,
        #     label=colorbar_label,
        # )
        width = np.where(self.simulation.domain_mask, width, np.nan)
        width_image = axes[1].imshow(
            width * 1e3,
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=20,
            extent=extent_km,
        )
        axes[1].set_title("90% interval width: P95 − P05")
        axes[1].set_xlabel("X (km)")
        axes[1].set_ylabel("Y (km)")

        fig.colorbar(
            width_image,
            ax=axes[1],
            shrink=0.85,
            label=colorbar_label,
        )
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        return 

    def _project_to_sim_latent(self, posterior_samples):
        """
        Project posterior samples of PMM-PCA data onto the simulation latent space. This is 
        necessary since the simulation may have reconstructed a new PC subspace
        for GMR purposes (usually a lower subspace since it's a joint distribution 
        of two variables).

        *** Fow now we ignore the GP prior
        """
        def decode_gmm_samples(z_gmm):
            n_sample = z_gmm.shape[0]
            samples = []
            for i in range(n_sample):
                sample = z_gmm[i,:].reshape(-1, 1)
                samples.append(self.forward_model.decode_gmm(sample))
            return np.array(samples).squeeze()
        
        z_gmm = posterior_samples[:, self.posterior.gmm_dim_index]
        z_gmm = z_gmm.reshape(z_gmm.shape[0], -1)
        enthalpy_gmm = decode_gmm_samples(z_gmm)

        enthalpy_gmm = np.asarray(enthalpy_gmm).reshape(z_gmm.shape[0], -1)

        # Shape: (n_posterior_samples, n_enthalpy_PCs)
        enthalpy_gmm_projected = self.simulation.pca_x[
            self.predictor_index].transform(enthalpy_gmm)

        return enthalpy_gmm_projected


    def _regression_indices(self):
        ndim_predictor = self.simulation.ndim_reduced[self.predictor_index]
        ndim_response  = self.simulation.ndim_reduced[self.response_index]

        # get the linear index for the dimension.
        # needs to be consistent with field_index order (which puts predictor first, then response)
        self.condition_index = list(range(ndim_predictor))
        self.target_index = list(range(ndim_predictor, ndim_predictor + ndim_response))
        print("Condition index:", self.condition_index)
        print("Target index:", self.target_index)
        return
