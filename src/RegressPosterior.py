import numpy as np

class RegressPosterior:
    def __init__(
            self, 
            simulation, 
            posterior_samples,
            predictor_index,
            response_index,
            ):
        self.simulation = simulation # the Simulation class object
        self.posterior_samples = posterior_samples
        self.predictor_index = predictor_index # the index of the predictor variable (posterior sample index)
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
            plt.title(rf'Observed {x_name}' + "\n" + f'(Sample {i+1})', fontsize=title_fontsize)
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


