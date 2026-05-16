from src.amortization import propagate_uncertainty
from src.utilities import standardize
from src.optimization import log_posterior_gradient, log_posterior, log_posterior_hessian
from src.ice import enthalpy_to_temperature
from src.hamiltonianMC import whitened_potential

from gmr.utils import check_random_state
from gmr import GMM
from scipy.optimize import minimize, Bounds

import pyro.infer.mcmc as mcmc
import pyro.ops.stats as stats

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import torch
import torch.optim as optim

class model:
    """ Latent space Bayesian inference model for ice sheet basal temperature estimation
    
    Author: Donglai Yang
    Affiliation: Georgia Institute of Technology
    Date: 2026-04-26
    """
    def __init__(self, extent=None, coord=None):
        self.extent = extent
        self.coord = coord

        # in our naming convention, '_ori' means standardized but not reduced (original dimension)
        # 'reduced' means the dimension has been reduced by PCA.
        self.X_ori = None
        self.Y_ori = None
        # standardization data (from simulation ensemble)
        self.X_mean = None 
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self.X_epsilon = None # relaxed standardization
        self.Y_epsilon = None # relaxed standardization
        # observation data
        self.Y_obs_ori = None
        # mask
        self.flight_mask = None
        self.domain_mask = None
        # basal thermal evidence
        self.thawed_mask = None
        self.thawed_fractional_area = None
        self.frozen_mask = None
        self.frozen_fractional_area = None
        self.pmp = None

        self.X_reduced = None # PCA reduction
        self.Y_reduced = None
        self.random_state = check_random_state(42)

        # join distribution data
        self.XY_train       = None
        self.XY_validation  = None
        self.XY_test        = None

        # models
        self.pca_y    = None
        self.pca_x    = None
        self.gmm      = None
        self.gmm_prop = None # after pushforward (~prior under obs)
        self.mcmc_md  = None 

        # results
        self.X_MAP       = None
        self.Tb_MAP      = None
        self.hessian_MAP = None
        # --- posterior
        self.post_samples= None
        self.Tb_p5       = None
        self.Tb_p95      = None
        self.Tb_std      = None
        self.Tb_mean     = None

        # dimension and indices
        self.nx = None
        self.ny = None
        self.ndim_ori     = None
        self.ndim_reduced_total = None
        self.ndim_reduced_x     = None
        self.ndim_reduced_y     = None
        self.n_channel = 1
        self.n_samples_total      = None
        self.n_samples_validation = None
        self.n_samples_test       = None
        self.n_samples_train      = None
        pass

    def load_sim_data(self, X, Y, domain_mask = None, flight_mask=None, show_plot=True):
        """ 
        Load the simulation data without any training splitting. We assume that these data have been standardized.
        
        Parameters
        ----------
        X: ndarray of shape (nx, ny, n_channel, n_features)
            The input features. Assuming input data are 2D data ensemble. 
        Y: ndarray of shape (nx, ny, n_channel, n_features)
            The output features.
        domain_mask: ndarray of shape (nx, ny, n_channel, n_features), optional
            Boolean mask indicating the valid data point in the simulation domain (continuous).
        flight_mask: ndarray of shape (nx, ny, n_channel, n_features), optional
            Boolean mask indicating the valid data point in a flight domain (discrete flight tracks).
        show_plot: bool, optional
            Whether to show the plot of the data.
        """
        self.nx, self.ny = X.shape[0], X.shape[1]
        self.ndim_ori = self.nx * self.ny * self.n_channel
        self.n_channel = X.shape[2]
        self.n_samples_total = X.shape[3]

        if show_plot:
            random_indices = np.random.choice(self.n_samples_total, size=5, replace=False)
            # figure size
            plt.figure(figsize=(9, 5))
            for i, idx in enumerate(random_indices):
                plt.subplot(2, 5, i + 1)
                X_plot = X[:, :, :, idx].copy()
                if domain_mask is not None:
                    X_plot[domain_mask == False] = np.nan # set the values outside the model boundary to NaN for better visualization
                plt.imshow(X_plot, cmap='viridis', vmin = -2, vmax = 2)
                plt.gca().invert_yaxis()
                plt.gca().axis('off')
                plt.title(f'Eb Sample {idx}')
                # plt.colorbar()
            for i, idx in enumerate(random_indices):
                plt.subplot(2, 5, i + 6)
                Y_plot = Y[:, :, 0, idx].copy()  
                if domain_mask is not None:
                    Y_plot[domain_mask == False] = np.nan
                
                plt.imshow(Y_plot, cmap='viridis', vmin=-2, vmax=2)
                plt.gca().invert_yaxis()
                plt.gca().axis('off')
                plt.title(f'Na Sample {idx}')

            plt.tight_layout()

        self.X_ori = X.reshape((self.nx * self.ny * self.n_channel, self.n_samples_total)).T
        self.Y_ori = Y.reshape((self.nx * self.ny * self.n_channel, self.n_samples_total)).T

        self.domain_mask = domain_mask
        self.flight_mask = flight_mask
        return 
    
    def load_split_data(self, X_train, Y_train,
                              X_validation, Y_validation,
                              X_test, Y_test):
        """ 
        Load the pre-split data into the model. This is an alternative to the split_data method.
        
        Parameters
        ----------
        X_train: ndarray of shape (n_samples_train, n_features)
            The training data in the original space.
        Y_train: ndarray of shape (n_samples_train, n_targets)
            The training target data in the original space.
        X_validation: ndarray of shape (n_samples_validation, n_features)
            The validation data in the original space.
        Y_validation: ndarray of shape (n_samples_validation, n_targets)
            The validation target data in the original space.
        X_test: ndarray of shape (n_samples_test, n_features)
            The test data in the original space.
        Y_test: ndarray of shape (n_samples_test, n_targets)
            The test target data in the original space.
        """
        # Apply fitted PCA on the whole data set to 
        # the training, validation, and test data separately        
        def flatten(X):
            return X.reshape((self.nx * self.ny * self.n_channel, -1)).T

        X_reduced_train = self.pca_x.fit_transform(flatten(X_train))
        Y_reduced_train = self.pca_y.fit_transform(flatten(Y_train))
        X_reduced_validation = self.pca_x.transform(flatten(X_validation))
        Y_reduced_validation = self.pca_y.transform(flatten(Y_validation))
        X_reduced_test = self.pca_x.transform(flatten(X_test))
        Y_reduced_test = self.pca_y.transform(flatten(Y_test))

        XY_train      = np.hstack((X_reduced_train, Y_reduced_train))
        XY_validation = np.hstack((X_reduced_validation, Y_reduced_validation))
        XY_test       = np.hstack((X_reduced_test, Y_reduced_test))
        self.XY_train = XY_train
        self.XY_validation = XY_validation
        self.XY_test = XY_test

        self.n_samples_train = XY_train.shape[0]
        self.n_samples_validation = XY_validation.shape[0]
        self.n_samples_test = XY_test.shape[0]

        print("Shape of XY_train:", self.XY_train.shape)
        print("Shape of XY_validation:", self.XY_validation.shape)
        print("Shape of XY_test:", self.XY_test.shape)
        return

    def load_obs_data(self, Y_obs, show_plot=True):
        """  
        Load the observation data.
        """
        print("shape of Y_obs:", Y_obs.shape)
        Y_obs_standardized = standardize(Y_obs.flatten(), self.Y_mean, self.Y_std,
                                         method='relaxation',
                                         epsilon=self.Y_epsilon)
        
        self.Y_obs_ori = Y_obs_standardized
        if show_plot:
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(Y_obs_standardized.reshape(self.nx, self.ny), cmap='bwr', vmin=-5, vmax=5)
            plt.title('Standardized Observed Ns')
            plt.colorbar()
            plt.gca().invert_yaxis()

            # second plot: histogram of the standardized observed Ns
            plt.subplot(1, 3, 2)
            plt.hist(Y_obs_standardized.reshape(self.nx, self.ny)[self.flight_mask].flatten(), bins=50, color='blue', alpha=0.7)
            plt.xlim(-15, 15)
            # plot y line at x = 0
            plt.axvline(x=0, color='red', linestyle='--')
            plt.title('Histogram of Standardized Observed Ns (Flightline Masked)')
            plt.xlabel('Standardized Ns Value')
            plt.ylabel('Frequency')

            plt.subplot(1, 3, 3)
            plt.imshow(Y_obs.reshape(self.nx, self.ny), cmap='viridis', vmin=0, vmax=30)
            plt.title('Observed Attenuation Rate')
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.show()

        return
    
    def load_standardization_data(self, X_mean, X_std, Y_mean, Y_std, X_epsilon=None, Y_epsilon=None):
        """
        Load the standardization data (mean, std) from the simulation ensemble
          -> Going between standardized space and the physical space
        """
        # shape check: the mean and std should all be flatten
        if X_mean.shape != (self.nx * self.ny * self.n_channel,):
            raise ValueError(f"X_mean should have shape {(self.nx * self.ny * self.n_channel,)}, but got {X_mean.shape}")
        if X_std.shape != (self.nx * self.ny * self.n_channel,):
            raise ValueError(f"X_std should have shape {(self.nx * self.ny * self.n_channel,)}, but got {X_std.shape}")
        if Y_mean.shape != (self.nx * self.ny * self.n_channel,):
            raise ValueError(f"Y_mean should have shape {(self.nx * self.ny * self.n_channel,)}, but got {Y_mean.shape}")
        if Y_std.shape != (self.nx * self.ny * self.n_channel,):
            raise ValueError(f"Y_std should have shape {(self.nx * self.ny * self.n_channel,)}, but got {Y_std.shape}")
        self.X_mean = X_mean
        self.X_std = X_std
        self.Y_mean = Y_mean
        self.Y_std = Y_std
        self.X_epsilon = X_epsilon
        self.Y_epsilon = Y_epsilon
        return 
    
    def load_evidence(self, \
                      thawed_mask,
                      thawed_frac_area, 
                      frozen_mask,
                      frozen_frac_area,
                      pmp,
                      show_plot=True):
        """
        Load the basal thermal evidence
        """
        # below are within the domain bound
        self.thawed_mask            = thawed_mask
        self.thawed_fractional_area = thawed_frac_area
        self.frozen_mask            = frozen_mask
        self.frozen_fractional_area = frozen_frac_area
        # pmp may be defined beyond domain bound
        self.pmp                    = pmp
        pmp_plot = pmp.copy().reshape(self.nx, self.ny)
        pmp_plot[self.domain_mask == False] = np.nan
        if show_plot:
            # visualize both and pmp
            plt.figure(figsize=(8, 12))
            plt.subplot(3, 2, 1)
            plt.imshow(self.frozen_mask.reshape(self.nx, self.ny), cmap='gray')
            plt.title('Frozen Base Mask')
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.subplot(3, 2, 2)
            plt.imshow(self.thawed_mask.reshape(self.nx, self.ny), cmap='gray')
            plt.title('Thawed Base Mask')
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.subplot(3, 2, 3)
            plt.imshow(self.frozen_fractional_area.reshape(self.nx, self.ny), cmap='Blues', alpha=0.5)
            plt.title('Frozen Fractional Area')
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.subplot(3, 2, 4)
            plt.imshow(self.thawed_fractional_area.reshape(self.nx, self.ny), cmap='Reds', alpha=0.5)
            plt.title('Thawed Fractional Area')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.subplot(3, 2, 5)
            plt.imshow(pmp_plot.reshape(self.nx, self.ny), cmap='hot')
            plt.title('Pressure Melting Point')
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.show()
        return

    def find_reduction_model(self, n_component_x, n_component_y):
        """ 
        Reduce the dimensionality of the input and output data using PCA.
        
        Parameters
        ----------
        n_component_x: int
            The number of principal components to compute for X.
        n_component_y: int
            The number of principal components to compute for Y.
        """
        pca_x = PCA(n_components=n_component_x)
        pca_y = PCA(n_components=n_component_y)
        # self.X_reduced = pca_x.fit_transform(self.X_ori)
        # self.Y_reduced = pca_y.fit_transform(self.Y_ori)
        self.pca_x = pca_x
        self.pca_y = pca_y
        print("PCA model saved to self.pca_x and self.pca_y.")

        self.ndim_reduced_total = n_component_x + n_component_y
        self.ndim_reduced_x     = n_component_x
        self.ndim_reduced_y     = n_component_y

        # add x and y indices
        self.y_indices = np.arange(self.ndim_reduced_y)
        self.x_indices = np.arange(self.ndim_reduced_y, self.ndim_reduced_total)
        return
    
    def split_data(self, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
        """ 
        Split the data into training, validation, and test sets.
        
        Parameters
        ----------
        train_ratio: float
            The ratio of the training set.
        validation_ratio: float
            The ratio of the validation set.
        test_ratio: float
            The ratio of the test set.
        """
        assert train_ratio + validation_ratio + test_ratio == 1.0, "The sum of the ratios must be 1."
        
        n_train = int(self.n_samples_total * train_ratio)
        n_validation = int(self.n_samples_total * validation_ratio)
        n_test = self.n_samples_total - n_train - n_validation

        self.n_samples_train = n_train
        self.n_samples_validation = n_validation
        self.n_samples_test = n_test

        XY_reduced = np.hstack((self.X_reduced, self.Y_reduced))
        indices = np.arange(self.n_samples_total)
        self.random_state.shuffle(indices)

        self.XY_train = XY_reduced[indices[:n_train]]
        self.XY_validation = XY_reduced[indices[n_train:n_train + n_validation]]
        self.XY_test = XY_reduced[indices[n_train + n_validation:]]

        print("Shape of XY_train:", self.XY_train.shape)
        print("Shape of XY_validation:", self.XY_validation.shape)
        print("Shape of XY_test:", self.XY_test.shape)
        return
        
    def train_gmm_XY(self, n_components):
        """ 
        Train a Gaussian Mixture Model on the joint distribution of X, Y in their latent space
        The joint Probability is combined in (Y, X) order

        Parameters
        ----------
        n_components: int
            The number of components for the Gaussian Mixture Model.
        """
        gmm = GMM(n_components=n_components, random_state=self.random_state)

        start_time = time.time()
        gmm.from_samples(self.XY_train)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"GMM training completed in {training_time:.2f} seconds.")
        self.gmm = gmm
        return 
    
    def derive_prior(self, beta=0.1, lambda1=1, lambda2=1, show_plot=True):
        """
        Derive the prior P(X) -- under Y_obs -- from the joint P(X,Y)
        This involves two steps:
            1. Finding the optimal Y in the latent space -> the mean (mu_obs_latent)
            2. Use the residue projected the PCA space along with variance from sim. ensemble
               to construct a covariance matrix as sigma_obs_latent   
            3. Then pushforward, \int P(X|Y) P(Y_obs) dY, which is analytical due to 
               (i) Gaussianity and (ii) Linear operation of PCA
        
        Parameters
        ----------
        beta: float
            Prefactor in the objective function 
        lambda1: float
            weight for data residual term in the covariance estimation
        lambda2: float
            weight for Y variance from simulation in the covariance estimation
        """
        # find mean and covariance
        z_optimal, residual_latent, residual_recon, residual = self.compute_optimal_Y_in_latent(beta=beta)

        residual_latent = residual_latent.reshape(-1, 1)  # shape (ndim_reduced_y, 1)
        residual_latent_outer = residual_latent @ residual_latent.T  # shape (ndim_reduced_y, ndim_reduced_y)
        variance_latent = np.diag(self.pca_y.explained_variance_)  # shape (ndim_reduced_y, ndim_reduced_y)

        cov_obs_latent = lambda1 * residual_latent_outer + lambda2 * variance_latent
        print("first 5x5 block of cov_obs_latent:")
        print(cov_obs_latent[:4, :4])
        # Should be positive definite — all eigenvalues > 0
        eigvals = np.linalg.eigvalsh(cov_obs_latent)
        print("min eigenvalue:", eigvals.min())  # must be > 0

        # assign mean value
        mu_obs_latent = z_optimal

        # propagate the observational uncertainty through the GMM to get the uncertainty in X
        gmm_propagated = propagate_uncertainty(mu_obs_latent.T,
                                               cov_obs_latent,
                                               self.gmm, 
                                               np.arange(self.ndim_reduced_y), 
                                               self.XY_train[:,:self.ndim_reduced_y])

        self.gmm_prop = gmm_propagated

        if show_plot:
            # sample from gmm_propagated to get the distribution of X
            n_samples = 300
            X_samples = gmm_propagated.sample(n_samples)
            X_samples_prop_ori = np.zeros((self.ndim_ori, n_samples))
            for i, sample in enumerate(X_samples):
                X_samples_prop_ori[:,i] = self.pca_x.inverse_transform(sample)

            # sample from un-propagated model for comparison
            XY_samples_ori = self.gmm.sample(n_samples=1000)
            Y_mean_unprop = np.mean(XY_samples_ori[:, :self.ndim_reduced_y], axis=0)
            # check to see if Y_mean_unprop is identical to mu_obs_latent
            if np.allclose(Y_mean_unprop, mu_obs_latent, atol=1e-2):
                print("Warning:The mean of Y from unpropagated GMM matches the optimized Y_obs mean.")
            gmm_unpropagated = self.gmm.condition(np.arange(self.ndim_reduced_y), Y_mean_unprop)
            X_samples_unprop = gmm_unpropagated.sample(n_samples)
            X_samples_unprop_ori = np.zeros((self.ndim_ori, n_samples))
            for i, sample in enumerate(X_samples_unprop):
                X_samples_unprop_ori[:,i] = self.pca_x.inverse_transform(sample)


            plt.figure(figsize=(10, 10))

            X_samples_mean_unprop = np.mean(X_samples_unprop_ori, axis=1).reshape(self.nx, self.ny)
            X_samples_std_unprop = np.std(X_samples_unprop_ori, axis=1).reshape(self.nx, self.ny)
            X_samples_mean_unprop[self.domain_mask == False] = np.nan
            X_samples_std_unprop[self.domain_mask == False] = np.nan
            plt.subplot(3, 2, 1)
            plt.imshow(X_samples_mean_unprop, cmap='bwr', vmin=-5, vmax=5)
            plt.title('Mean of $E_b$ (Unpropagated)')
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.subplot(3, 2, 2)
            plt.imshow(X_samples_std_unprop, cmap='hot', vmin=0, vmax=3)
            plt.title('Std of $E_b$ (Unpropagated)')
            plt.colorbar()
            plt.gca().invert_yaxis()

            X_samples_mean_prop = np.mean(X_samples_prop_ori, axis=1).reshape(self.nx, self.ny)
            X_samples_std_prop = np.std(X_samples_prop_ori, axis=1).reshape(self.nx, self.ny)
            X_samples_mean_prop[self.domain_mask == False] = np.nan
            X_samples_std_prop[self.domain_mask == False] = np.nan
            plt.subplot(3, 2, 3)
            plt.imshow(X_samples_mean_prop, cmap='bwr', vmin=-5, vmax=5)
            plt.title('Mean of $E_b$ (Obs. propagated)')
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.subplot(3, 2, 4)
            plt.imshow(X_samples_std_prop, cmap='hot', vmin=0, vmax=3)
            plt.title('Std of $E_b$ (Obs. propagated)')
            plt.colorbar()
            plt.gca().invert_yaxis()

            # plt those two on a histogram
            plt.subplot(3, 2, 5)
            plt.hist(X_samples_mean_unprop[self.domain_mask].flatten(), bins=50, color='blue', alpha=0.7, label='Unpropagated')
            plt.hist(X_samples_mean_prop[self.domain_mask].flatten(), bins=50, color='red', alpha=0.7, label='Propagated')
            plt.legend()
            
            # plot two std on a histogram
            plt.subplot(3, 2, 6)
            plt.hist(X_samples_std_unprop[self.domain_mask].flatten(), bins=50, color='blue', alpha=0.7, label='Unpropagated')  
            plt.hist(X_samples_std_prop[self.domain_mask].flatten(), bins=50, color='red', alpha=0.7, label='Propagated')   
            plt.legend()
            plt.savefig('../figs/propagation_effect.png', dpi=300)

            plt.show()
        return 

    def compute_optimal_Y_in_latent(self, beta=0.1, show_plot=True):
        """  
        Solve the optimization problem with regularization to find the optimal Y in the latent space

        Parameters 
        ----------
            beta: float

        """
        def objective_scaled(z_scaled, Y_obs, V, beta):
            """Objective in scaled z space — avoids scaler transform inside loop."""
            z_orig = scaler.inverse_transform(z_scaled.reshape(1, -1)).flatten()
            
            # Likelihood term (in original space)
            residual = Y_obs - z_orig @ V
            likelihood_term = np.mean(residual**2)
            
            # Prior term (directly in scaled space — no transform needed)
            density = gmm_latent.to_probability_density(z_scaled.reshape(1, -1))
            if density <= 0:
                prior_term = 1e6
            else:
                prior_term = -np.log(float(density))
            print(f" Likelihood: {likelihood_term/beta:.3f}, Prior: {prior_term:.3f}")
            return likelihood_term / beta + prior_term
        def gradient_scaled(z_scaled, Y_obs, V, beta):
            # --- Likelihood gradient (analytical) ---
            z_orig = scaler.inverse_transform(z_scaled.reshape(1, -1)).flatten()
            residual = Y_obs - z_orig @ V
            grad_lik_orig = -2 * (residual @ V.T) / len(Y_obs)
            grad_lik_scaled = grad_lik_orig * scaler.scale_

            # --- Prior gradient (central differences) ---
            eps = 1e-4
            grad_prior = np.zeros_like(z_scaled)
            for i in range(len(z_scaled)):
                z_plus = z_scaled.copy();  z_plus[i] += eps
                z_minus = z_scaled.copy(); z_minus[i] -= eps
                f_plus  = -np.log(float(gmm_latent.to_probability_density(z_plus.reshape(1,-1)))  + 1e-300)
                f_minus = -np.log(float(gmm_latent.to_probability_density(z_minus.reshape(1,-1))) + 1e-300)
                grad_prior[i] = (f_plus - f_minus) / (2 * eps)

            return grad_lik_scaled / beta + grad_prior

        gmm_latent = GMM(n_components=2, random_state=np.random.RandomState(42))
        y_latent_all = self.pca_y.transform(self.Y_ori)

        n_latent_samples = y_latent_all.shape[0]
        indices = np.arange(n_latent_samples)
        local_rng = np.random.RandomState(42)  # fixed, isolated seed
        local_rng.shuffle(indices)

        split_point = int(0.8 * n_latent_samples)
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]
        y_latent_train = y_latent_all[train_indices]
        y_latent_test = y_latent_all[test_indices]
        scaler = StandardScaler()
        y_latent_train_scaled = scaler.fit_transform(y_latent_train)
        y_latent_test_scaled  = scaler.transform(y_latent_test)
        # train
        gmm_latent.from_samples(y_latent_train_scaled)

        # initial state for the optimization 
        best_k = np.argmax(gmm_latent.priors)
        z_init_scaled = gmm_latent.means[best_k].copy()
        print(f"Density at init: {gmm_latent.to_probability_density(z_init_scaled.reshape(1,-1))}")

        # Bounds in scaled space: ±5 std (which is just ±5 since data is standardized)
        lb_scaled = np.full(self.ndim_reduced_y, -5.0)
        ub_scaled = np.full(self.ndim_reduced_y,  5.0)

        result = minimize(objective_scaled, z_init_scaled,
                  args=(self.Y_obs_ori.flatten(), self.pca_y.components_, beta),
                  jac=gradient_scaled,
                  method='L-BFGS-B',
                  bounds=Bounds(lb_scaled, ub_scaled))

        # Convert optimal back to original space
        z_optimal_scaled = result.x
        z_optimal = scaler.inverse_transform(z_optimal_scaled.reshape(1, -1)).flatten()
        print("Optimization success:", result.success)

        print(result.message)
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")
        print(f"Final objective: {result.fun:.4f}")

        Y_obs_reconstructed_optimal = z_optimal @ self.pca_y.components_
        Y_obs_reconstructed_optimal_img = Y_obs_reconstructed_optimal.reshape(self.nx, self.ny)
        residual = self.Y_obs_ori.flatten() - Y_obs_reconstructed_optimal
        residual_latent = self.pca_y.transform(residual.reshape(1, -1)).flatten()
        residual_recon = self.pca_y.inverse_transform(residual_latent.reshape(1, -1)).reshape(self.nx, self.ny)

        if show_plot:
            plt.figure(figsize=(20, 6))
            plt.subplot(1,3,1)
            plt.imshow(Y_obs_reconstructed_optimal_img, cmap='bwr', vmin=-2, vmax=2)
            plt.title('Reconstructed Observed Y from Optimized Latent z')
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.subplot(1,3,2)
            plt.imshow(self.Y_obs_ori.reshape(self.nx, self.ny), cmap='bwr', vmin=-2, vmax=2)
            plt.title('Original Mean of Observed Y')
            plt.colorbar()
            plt.gca().invert_yaxis()
            # projecting the residue to PCA space and show the reconstruction
            plt.subplot(1,3,3)
            plt.imshow(residual_recon, cmap='bwr', vmin=-2, vmax=2)
            plt.title('PCA Reconstruction of Residual')
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.show()
        return z_optimal, residual_latent, residual_recon, residual
    
    def compute_MAP(self, beta=1, n_iter=20, lr=0.5, show_trajectory=False):
        """ 
        Compute the Maximum A Posteriori
        """
        # find good initial value for optimization
        # --> here we use mean 

        n_samples = 300
        X_samples = self.gmm_prop.sample(n_samples)
        X_samples_mean = np.mean(X_samples, axis=0)
        init_Eb_ori = X_samples_mean.flatten()

        init_Eb_reduced = torch.from_numpy(init_Eb_ori)
        Tpmp = torch.from_numpy(self.pmp).flatten()
        dw = torch.from_numpy(self.thawed_fractional_area)
        df = torch.from_numpy(self.frozen_fractional_area)

        Eb_mean_data = torch.from_numpy(self.X_mean)
        Eb_std_data = torch.from_numpy(self.X_std)

        # start the optimization using L-BFGS
        n_iter = 20
        X = init_Eb_reduced.clone().requires_grad_(True)
        optimizer = optim.LBFGS([X], lr=lr, max_iter=n_iter)

        saved_snapshots = []
        def closure():
            """ closure function for L-BFGS optimizer
            """
            optimizer.zero_grad()
            X_detached = X.detach()

            # function value: negative log posterior
            value = -log_posterior(X_detached, 
                                   self.pca_x.components_, 
                                   self.gmm_prop,
                                   beta,
                                   Tpmp, 
                                   Eb_mean_data, 
                                   Eb_std_data,
                                   dw, 
                                   df,
                                   verbose=False, 
                                   Eb_epsilon=self.X_epsilon)
            # gradient
            grad = -log_posterior_gradient(X_detached, 
                                           self.pca_x.components_, 
                                           self.gmm_prop, 
                                           beta, 
                                           Tpmp, 
                                           Eb_mean_data, 
                                           Eb_std_data, 
                                           dw, 
                                           df, 
                                           Eb_epsilon=self.X_epsilon)
            # Convert gradient to torch tensor if needed
            if not isinstance(grad, torch.Tensor):
                grad = torch.tensor(grad, dtype=X.dtype, device=X.device)
            
            X.grad = grad
            return value

        for i in range(n_iter):
            print(f"Starting iteration {i+1}/{n_iter}...")
            optimizer.step(closure)
            # monitor the loss (or the neg log posterior)
            with torch.no_grad():
                current_loss = -log_posterior(X, 
                                              self.pca_x.components_, 
                                              self.gmm_prop,
                                              beta, 
                                              Tpmp, 
                                              Eb_mean_data, 
                                              Eb_std_data,
                                              dw, 
                                              df,
                                              verbose=False, 
                                              Eb_epsilon=self.X_epsilon)
                print(f"Iteration {i+1}/{n_iter}, Negative Log Posterior: {current_loss.item():.4f}")
            # save the first n iterations and the last iteration
            if i < 9 or i == n_iter - 1:
                saved_snapshots.append(X.detach().cpu().numpy().copy())

            # save the hessian matrix at the last iteration for analysis
            if i == n_iter - 1:
                optimized_X = X.detach().requires_grad_(True)   # fresh leaf, grad-enabled
                hessian = log_posterior_hessian(
                    optimized_X, self.pca_x.components_, self.gmm_prop,
                    beta, Tpmp, Eb_mean_data, Eb_std_data,
                    dw, df,
                    Eb_epsilon=self.X_epsilon

                    )
        
        X_optimized  = X.detach().numpy()
        Eb_std_data  = Eb_std_data.numpy()
        Eb_mean_data = Eb_mean_data.numpy()
        Eb_MAP = self.pca_x.inverse_transform(X_optimized.reshape(1, -1)).reshape(self.nx, self.ny)
        Eb_MAP_ori = torch.from_numpy(Eb_MAP * Eb_std_data.reshape(self.nx, self.ny) + Eb_mean_data.reshape(self.nx, self.ny))
        Tb_MAP = enthalpy_to_temperature(Eb_MAP_ori.flatten(), Tpmp.flatten()).reshape(self.nx, self.ny).numpy()
        Tb_MAP[self.domain_mask == False] = np.nan

        # add
        self.Tb_MAP = Tb_MAP
        self.hessian_MAP = hessian
        self.X_MAP = X_optimized

        plt.figure(figsize=(6, 10))
        plt.subplot(2, 1, 1)
        plt.imshow(Tb_MAP, cmap='RdBu_r', vmin=255, vmax=273.15)
        plt.title('MAP Estimate of E_b')
        plt.colorbar()
        plt.gca().invert_yaxis()
        dT_to_pmp = Tpmp.reshape(self.nx, self.ny) - Tb_MAP
        dT_to_pmp[self.domain_mask == False] = np.nan
        # plot as contour lines
        plt.subplot(2, 1, 2)
        plt.imshow(dT_to_pmp, cmap='hot', vmin=0, vmax=5)
        plt.title('MAP Estimate of T_b - T_pmp')
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.show()

        if show_trajectory:
            # plot all the snapshots to see the optimization trajectory
            n_snapshots = len(saved_snapshots)
            plt.figure(figsize=(20, 6))
            for i, snapshot in enumerate(saved_snapshots):
                Eb_snapshot = self.pca_x.inverse_transform(snapshot.reshape(1, -1)).reshape(256, 256)
                Eb_snapshot_ori = torch.from_numpy(Eb_snapshot * Eb_std_data.reshape(256, 256) + Eb_mean_data.reshape(256, 256))
                Tb_snapshot = enthalpy_to_temperature(Eb_snapshot_ori.flatten(), Tpmp.flatten()).reshape(256, 256)
                plt.subplot(2, n_snapshots//2, i+1)
                plt.imshow(Tb_snapshot * self.domain_mask, cmap='RdBu_r', vmin=250, vmax=273.15)
                
                plt.title(f'Iteration {i+1}')
                plt.colorbar()
                plt.gca().invert_yaxis()
            plt.suptitle('Optimization Trajectory of T_b')
            plt.show()
        return
    
    def derive_posterior(self, beta=1, warmup_steps=1000, num_samples=4000):
        """
        Derive the posterior distribution with Hamiltonian Monte Carlo
        """

        H_pot = torch.tensor(-self.hessian_MAP, dtype=torch.float64)
        # stable matrix inversion
        jitter = 1e-5 * torch.eye(H_pot.shape[0], dtype=torch.float64)
        cov_matrix = torch.inverse(H_pot + jitter)

        # L is the lower triangular Cholesky factor
        L = torch.linalg.cholesky(cov_matrix)
        X_opt_tensor = torch.tensor(self.X_MAP, dtype=torch.float64)
        potential = whitened_potential(
            X_opt_tensor=X_opt_tensor,
            L=L,
            V=self.pca_x.components_,
            gmm=self.gmm_prop,
            beta=beta,
            Tpmp=self.pmp,
            Eb_mean=self.X_mean,
            Eb_std=self.X_std,
            dw=self.thawed_fractional_area,
            df=self.frozen_fractional_area,
            Eb_epsilon=self.X_epsilon
        )
        nuts_kernel = mcmc.NUTS(
            potential_fn=potential,
            adapt_step_size=True,       # NUTS easily finds a large step size in spherical space
            adapt_mass_matrix=True,    # Keep mass matrix as Identity!
            max_tree_depth=12           # Let it run deep if it wants, it will be fast now
        )

        # 4. Initialize and Run MCMC
        # Since z_initial = MAP, the corresponding u_initial is exactly a vector of 0s
        initial_params = {"u": torch.zeros_like(X_opt_tensor)}

        num_samples = num_samples
        warmup_steps = warmup_steps  # Space is perfectly isotropic now, warmup is incredibly fast

        # run HMC
        mcmc_run = mcmc.MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            initial_params=initial_params
        )

        print(f"Starting HMC sampling: {num_samples} samples, {warmup_steps} warmup...")
        mcmc_run.run(extra_fields=("potential_energy",))
        torch.save(mcmc_run, "../data/posterior-hmc-models/mcmc_run.pt")
        self.mcmc_md = mcmc_run
        return 
    
    def analyze_posterior_samples(self, beta=1, loading=True):
        """
        Extract the HMC samples and analyze their properties, including:
        1. Log probability distribution of the samples to understand the posterior landscape.
        2. Percentile-based credible intervals (e.g., 5th and 95th percentiles) to quantify uncertainty in the inferred parameters.
        """
        # ------------------------ Extracting and Analyzing HMC Samples ------------------------
        # u_samples: shape (N, M) numpy array
        if loading:
            self.mcmc_md = torch.load("../data/posterior-hmc-models/mcmc_run.pt")

        H_pot = torch.tensor(-self.hessian_MAP, dtype=torch.float64)
        # stable matrix inversion
        jitter = 1e-5 * torch.eye(H_pot.shape[0], dtype=torch.float64)
        cov_matrix = torch.inverse(H_pot + jitter)

        # L is the lower triangular Cholesky factor
        L = torch.linalg.cholesky(cov_matrix)
        X_opt_tensor = torch.tensor(self.X_MAP, dtype=torch.float64)
        potential = whitened_potential(
            X_opt_tensor=X_opt_tensor,
            L=L,
            V=self.pca_x.components_,
            gmm=self.gmm_prop,
            beta=beta,
            Tpmp=self.pmp,
            Eb_mean=self.X_mean,
            Eb_std=self.X_std,
            dw=self.thawed_fractional_area,
            df=self.frozen_fractional_area,
            Eb_epsilon=self.X_epsilon
        )

        u_samples = self.mcmc_md.get_samples()["u"].cpu().numpy()  # shape (N, M)
        u_samples = torch.tensor(u_samples, dtype=torch.float64)
        num_samples = u_samples.shape[0]
        log_probs = []

        with torch.no_grad():
            for i in range(num_samples):
                u_i = u_samples[i]  # shape (M,)
                
                # potential_fn_whitened returns NEGATIVE log prob → negate it
                neg_lp = potential({"u": u_i})
                log_probs.append(neg_lp.item())

        log_probs = np.array(log_probs)  # shape: (N,)
        # Sort samples by log probability (ascending)
        sorted_indices = np.argsort(log_probs) 
        samples_sorted = u_samples[sorted_indices]
        log_probs_sorted = log_probs[sorted_indices]

        # Find the 5th and 95th percentile THRESHOLDS on log prob
        lp_p5  = np.percentile(log_probs_sorted, 5)
        lp_p95 = np.percentile(log_probs_sorted, 95)

        # The actual samples at those boundaries
        z_samples = (X_opt_tensor + (L @ samples_sorted.T).T).numpy()

        idx_p5  = np.argmin(np.abs(log_probs_sorted - lp_p5))
        idx_p95 = np.argmin(np.abs(log_probs_sorted - lp_p95))

        z_p5  = z_samples[idx_p5]  
        z_p95 = z_samples[idx_p95]  

        n_keep = int(0.90 * num_samples)
        hpd_samples = z_samples[sorted_indices[-n_keep:]]  # highest density samples

        # Apply the reverse transformation to the samples: z_samples = MAP + u_samples @ L^T
        hmc_samples = self.X_MAP + (u_samples.numpy() @ L.numpy().T)
        print(f"HMC Sampling Complete. Extracted shape: {hmc_samples.shape}")
        self.post_samples = hmc_samples

        Eb_samples_norm = self.pca_x.inverse_transform(hmc_samples) 
        Eb_std_flat     = self.X_std.flatten()
        Eb_mean_flat    = self.X_mean.flatten()
        Eb_samples_ori  = Eb_samples_norm * Eb_std_flat + Eb_mean_flat

        Tb_samples = np.zeros_like(Eb_samples_ori)
        Tpmp_flat  = self.pmp.flatten()

        for i in range(num_samples):
            # Depending on whether enthalpy_to_temperature expects torch tensors or numpy arrays
            Eb_i = torch.from_numpy(Eb_samples_ori[i])
            Tpmp_i = torch.from_numpy(Tpmp_flat) if isinstance(Tpmp_flat, np.ndarray) else Tpmp_flat
            
            Tb_samples[i] = enthalpy_to_temperature(Eb_i, Tpmp_i).numpy()

        # convert the 5% and 95% 
        Eb_p5 = self.pca_x.inverse_transform(z_p5) * Eb_std_flat + Eb_mean_flat
        Eb_p95 = self.pca_x.inverse_transform(z_p95) * Eb_std_flat + Eb_mean_flat

        # Calculate Standard Deviation directly across the sample axis
        self.Tb_std = np.std(Tb_samples, axis=0).reshape(256, 256)

        # Summarize the HPD region as a band
        self.Tb_p5 = Tb_samples[sorted_indices[-n_keep:]].min(axis=0)
        self.Tb_p95 = Tb_samples[sorted_indices[-n_keep:]].max(axis=0)
        # self.Tb_p5 = enthalpy_to_temperature(Eb_p5, Tpmp_flat, istorch=False)
        # self.Tb_p95 = enthalpy_to_temperature(Eb_p95, Tpmp_flat, istorch=False)

        self.Tb_p5 = self.Tb_p5.reshape(self.nx, self.ny)
        self.Tb_p95 = self.Tb_p95.reshape(self.nx, self.ny)
        self.Tb_p5[self.domain_mask == False] = np.nan
        self.Tb_p95[self.domain_mask == False] = np.nan

        return 
    
    def posterior_quality_check(self, slicing=1000):
        """ 
        Checking the quality of the HMC samples with standard diagnostics:
        1. Effective Sample Size (ESS) to assess the number of independent samples.
        2. Trace plots to visually inspect the sampling trajectory for each latent dimension.
        3. Autocorrelation plots to evaluate the correlation between samples at different lags.

        Parameters
        ----------
        slicing: int
            The number of initial samples to discard as "wandering phase" before the chain stabilizes
        """
        samples_tensor = torch.tensor(self.post_samples).unsqueeze(0) 
        ess = stats.effective_sample_size(samples_tensor).numpy()

        print(f"--- HMC Sampling Metrics ---")
        print(f"Total Samples Drawn: {self.post_samples.shape[0]}")
        print(f"Mean ESS across all dimensions: {ess.mean():.2f}")
        print(f"Minimum ESS (Worst mixing dimension): {ess.min():.2f}")
        print(f"Maximum ESS (Best mixing dimension): {ess.max():.2f}")

        # Slice off the wandering phase
        stable_samples = self.post_samples[slicing:]

        # Re-run the ESS calculation on stable_samples
        samples_tensor = torch.tensor(stable_samples).unsqueeze(0) 
        ess = stats.effective_sample_size(samples_tensor).numpy()
        print(f"Post-Burn-in Mean ESS: {ess.mean():.2f}")
        print(f"Post-Burn-in Min ESS: {ess.min():.2f}")

        # 2. Plot Trace and Autocorrelation for the first 3 latent dimensions
        n_dims_to_plot = min(3, self.post_samples.shape[1])
        fig, axes = plt.subplots(n_dims_to_plot, 2, figsize=(12, 3 * n_dims_to_plot))

        for i in range(n_dims_to_plot):
            # Left Column: Trace Plot
            axes[i, 0].plot(self.post_samples[:, i], alpha=0.7, color='b')
            axes[i, 0].set_title(f"Trace Plot: Latent Component {i}")
            axes[i, 0].set_ylabel("Value")
            axes[i, 0].set_xlabel("Sample Step")
            
            # Right Column: Autocorrelation (ACF) Plot
            # We must mean-center the data for plt.acorr
            centered_data = self.post_samples[:, i] - np.mean(self.post_samples[:, i])
            axes[i, 1].acorr(centered_data, maxlags=50, usevlines=True, normed=True, lw=2)
            axes[i, 1].set_title(f"Autocorrelation (ACF): Latent Component {i}")
            axes[i, 1].set_xlim(0, 50)
            axes[i, 1].set_xlabel("Lag")
        plt.tight_layout()
        plt.show()
        return 
    def plot_gmm_samples(self, n_samples=3):
        """   
        Visualize GMM predictions from random test samples. Default to 3 samples. 
        
        """
        n_test_samples_plot = 3 
        rand_idx = np.random.choice(range(self.n_samples_test), size=n_test_samples_plot, replace=False)
        for i in rand_idx:
            y_test = self.XY_test[i, :self.ndim_reduced_y]  # Y part
            x_test = self.XY_test[i, self.ndim_reduced_y:]  # X part

            # Predict X given Y
            condition_index = np.arange(self.ndim_reduced_y)
            x_pred_gmm = self.gmm.condition(condition_index, y_test)

            # sample from this conditional distribution to get uncertainty
            n_uq_sample = 400
            x_uq_samples = x_pred_gmm.sample(n_uq_sample)
            x_uq_samples_ori = np.zeros((self.nx * self.ny, n_uq_sample))
            # Inverse transform to original space
            # then the uq samples
            for j, sample in enumerate(x_uq_samples):
                x_uq_samples_ori[:,j] = self.pca_x.inverse_transform(sample)

            # compute the mean from the ensemble
            x_pred_mean = np.mean(x_uq_samples_ori, axis=1)
            x_pred_img = x_pred_mean.reshape(self.nx, self.ny)
            
            # compute std along each dimension of the uq samples
            x_uq_std = np.std(x_uq_samples_ori, axis=1)
            x_uq_std = x_uq_std.reshape(self.nx, self.ny)

            # Reshape X for visualization
            x_test_original = self.pca_x.inverse_transform(x_test.reshape(1, -1))
            x_test_img = x_test_original.reshape(self.nx, self.ny)

            # observed Y
            obs_Y = self.pca_y.inverse_transform(y_test)
            obs_Y = obs_Y.reshape(self.nx, self.ny)
            # Plotting: five columns: observed Y, True X, predicted X, error (RMSE), uncertainty (stddev)
            plt.figure(figsize=(24, 4))
            plt.subplot(1, 5, 1)
            plt.imshow(obs_Y, cmap='bwr', vmin=-2, vmax=2)
            plt.title('Observed Y')
            plt.colorbar()
            # invert y axis
            plt.gca().invert_yaxis()

            plt.subplot(1, 5, 2)
            plt.imshow(x_test_img, cmap='bwr', vmin=-2, vmax=2)
            plt.title('True X')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.subplot(1, 5, 3)
            plt.imshow(x_pred_img, cmap='bwr', vmin=-2, vmax=2)
            plt.title('Predicted X')
            plt.colorbar()
            plt.gca().invert_yaxis()
            
            plt.subplot(1, 5, 4)
            rmse_img = np.sqrt((x_test_img - x_pred_img) ** 2)
            plt.imshow(rmse_img, cmap='hot', vmin = 0, vmax = 1)
            plt.title('RMSE')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.subplot(1, 5, 5)
            plt.imshow(x_uq_std, cmap='hot', vmin = 0, vmax = 1)
            plt.title('Uncertainty (stddev)')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.suptitle(f'Test Sample {i+1}')
            # save the figure, dpi = 300
            plt.show()

    def pca_scree(self, n_component_x, n_component_y):
        """ 
        Perform PCA on the input data and plot the scree plot.

        Parameters
        ----------
        n_component_x: int
            The number of principal components to compute for X.
        n_component_y: int
            The number of principal components to compute for Y.
        """
        pca_x = PCA(n_components=n_component_x)
        pca_y = PCA(n_components=n_component_y)
        pca_x.fit(self.X_ori)
        pca_y.fit(self.Y_ori)
        cum_variance_x = np.cumsum(pca_x.explained_variance_ratio_)
        cum_variance_y = np.cumsum(pca_y.explained_variance_ratio_)
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, n_component_x + 1), cum_variance_x, marker='o', label='X')
        plt.plot(np.arange(1, n_component_y + 1), cum_variance_y, marker='o', label='Y')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        plt.legend()
        plt.grid()
        plt.show()
        return 
    
    def pca_recon_inspection(self, n_component_x, n_component_y):
        """ 
        Visually inspect the PCA reconstruction quality
        
        """
        pca_x = PCA(n_components=n_component_x)
        pca_y = PCA(n_components=n_component_y)
        pca_x.fit(self.X_ori)
        pca_y.fit(self.Y_ori)

        X_recon = pca_x.inverse_transform(pca_x.transform(self.X_ori))
        Y_recon = pca_y.inverse_transform(pca_y.transform(self.Y_ori))

        # visualize three random samples
        random_indices = np.random.choice(self.n_samples_total, size=3, replace=False)
        plt.figure(figsize=(10, 8))
        for i, idx in enumerate(random_indices):
            X_sample = self.X_ori[idx].reshape(self.nx, self.ny, self.n_channel)
            plt.subplot(3, 3, i + 1)
            plt.imshow(X_sample, cmap='viridis', vmin = -2, vmax = 2)
            plt.title(f'X Sample {idx}')
            plt.gca().invert_yaxis()
            plt.colorbar()
        for i, idx in enumerate(random_indices):
            X_recon_sample = X_recon[idx].reshape(self.nx, self.ny, self.n_channel)
            plt.subplot(3, 3, i + 4)
            plt.imshow(X_recon_sample, cmap='viridis', vmin = -2, vmax = 2)
            plt.title(f'Reconstructed X {idx}')
            plt.gca().invert_yaxis()
            plt.colorbar()
        # difference
        for i, idx in enumerate(random_indices):
            plt.subplot(3, 3, i + 7)
            X_sample = self.X_ori[idx].reshape(self.nx, self.ny, self.n_channel)
            X_recon_sample = X_recon[idx].reshape(self.nx, self.ny, self.n_channel)
            residue = X_sample - X_recon_sample
            plt.imshow(residue[:, :, 0], cmap='bwr', vmin = -2, vmax = 2)
            plt.title(f'X difference {idx}')
            plt.gca().invert_yaxis()
            plt.colorbar()
        plt.tight_layout()

        plt.figure(figsize=(10, 8))
        for i, idx in enumerate(random_indices):
            Y_sample = self.Y_ori[idx].reshape(self.nx, self.ny, self.n_channel)
            plt.subplot(3, 3, i + 1)
            plt.imshow(Y_sample, cmap='viridis', vmin = -2, vmax = 2)
            plt.title(f'Y Sample {idx}')
            plt.gca().invert_yaxis()
            plt.colorbar()
        for i, idx in enumerate(random_indices):
            Y_recon_sample = Y_recon[idx].reshape(self.nx, self.ny, self.n_channel)
            plt.subplot(3, 3, i + 4)
            plt.imshow(Y_recon_sample, cmap='viridis', vmin = -2, vmax = 2)
            plt.title(f'Reconstructed Y {idx}')
            plt.gca().invert_yaxis()
            plt.colorbar()
        # difference
        for i, idx in enumerate(random_indices):
            plt.subplot(3, 3, i + 7)
            Y_sample = self.Y_ori[idx].reshape(self.nx, self.ny, self.n_channel)
            Y_recon_sample = Y_recon[idx].reshape(self.nx, self.ny, self.n_channel)
            residue = Y_sample - Y_recon_sample
            plt.imshow(residue[:, :, 0], cmap='bwr', vmin = -2, vmax = 2)
            plt.title(f'Y difference {idx}')
            plt.gca().invert_yaxis()
            plt.colorbar()
        plt.tight_layout()
        return
    
    def plot_evidence_consistency(self, T, beta=1):
        """
        Check the consistency between a basal temperature field against the known basal thermal evidence

        Parameters
        ----------
        T: 2D array (nx*ny, )
            Basal temperature field in the original space
        """
        epsilon = 0.5 * beta
        pmp = self.pmp.flatten().copy()

        T_thawed = np.nan * np.ones_like(T)
        T_frozen = np.nan * np.ones_like(T)
        pmp_thawed = np.nan * np.ones_like(pmp)
        pmp_frozen = np.nan * np.ones_like(pmp)
        T_thawed[self.thawed_mask==True]   = T[self.thawed_mask==True]
        T_frozen[self.frozen_mask==True]   = T[self.frozen_mask==True]
        pmp_thawed[self.thawed_mask==True] = pmp[self.thawed_mask==True]
        pmp_frozen[self.frozen_mask==True] = pmp[self.frozen_mask==True]
        dT_thawed = np.abs(pmp_thawed - T_thawed)
        dT_frozen = np.abs(pmp_frozen - T_frozen)
        consist_thawed = np.where(dT_thawed < epsilon, 1, 0)
        consist_frozen = np.where(dT_frozen > epsilon, 1, 0)
        total_consist   = consist_thawed + consist_frozen

        inconsist_thawed = np.where(dT_thawed >= epsilon, 1, 0)
        inconsist_frozen = np.where(dT_frozen <= epsilon, 1, 0)
        total_inconsist = inconsist_thawed + inconsist_frozen

        total_consist_count = np.nansum(total_consist)
        total_evidence_count = np.sum(self.thawed_mask) + np.sum(self.frozen_mask)
        total_consist_fraction = total_consist_count / total_evidence_count
        print(f"Total consistent points: {total_consist_count} out of {total_evidence_count} ({total_consist_fraction:.2%})")
        
        # make the total_consist and total_inconsist into scattered point dataset
        X, Y = np.meshgrid(self.coord[0], self.coord[1])
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        consistent_points_x = X_flat[total_consist == 1]
        consistent_points_y = Y_flat[total_consist == 1]
        inconsistent_points_x = X_flat[total_inconsist == 1]
        inconsistent_points_y = Y_flat[total_inconsist == 1]

        plt.figure(figsize=(8, 8))
        plt.imshow(T.reshape(self.nx, self.ny), 
                   extent=self.extent,
                   cmap='RdBu_r', 
                   vmin=250, vmax=273.15,
                   alpha=0.5)
        plt.colorbar()
        plt.scatter(consistent_points_x/1e3, consistent_points_y/1e3, 
                    color='green', 
                    s=3,
                    alpha=0.7, 
                    label='Consistent with Evidence',
                    edgecolors='black',
                    linewidths=0.2)
        plt.scatter(inconsistent_points_x/1e3, inconsistent_points_y/1e3, 
                    color='red', 
                    s=3,
                    alpha=0.7, 
                    label='Inconsistent with Evidence',
                    edgecolors='black',
                    linewidths=0.2)
        plt.title('Basal Temperature Field with Consistency Markers')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        plt.show()
        return 
        