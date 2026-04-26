from src.amortization import form_obs_cov_col, build_spatial_covariance_operator, sample_from_spatial_cov, pushforward, propagate_uncertainty
from src.amortization import compute_conditional_expected_val, compute_conditional_std_val_latent
from src.utilities import standardize
from src.optimization import log_prior_gradient
from src.optimization import log_posterior_gradient, log_posterior, log_posterior_hessian
from src.optimization import finite_difference_check
from src.ice import compute_pmp, enthalpy_to_temperature

from gmr.utils import check_random_state
from gmr import GMM

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import time
import torch

class model:
    def __init__(self):

        self.X_ori = None
        self.Y_ori = None
        self.X_reduced = None # PCA reduction
        self.Y_reduced = None
        self.random_state = check_random_state(42)

        # pca model: for (inverse) transformation
        self.pca_y = None
        self.pca_x = None

        # join distribution data
        self.XY_train       = None
        self.XY_validation = None
        self.XY_test        = None

        self.gmm = None

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

    def load_data(self, X, Y, mask=None, show_plot=True):
        """ 
        Load the simulation data. We assume that these data have been standardized.
        
        Parameters
        ----------
        X: ndarray of shape (nx, ny, n_channel, n_features)
            The input features. Assuming input data are 2D data ensemble. 
        Y: ndarray of shape (nx, ny, n_channel, n_features)
            The output features.
        mask: ndarray of shape (nx, ny, n_channel, n_features), optional
            Boolean mask indicating the valid data point in a spatial domain. 
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
                X_plot[mask == 0] = np.nan # set the values outside the model boundary to NaN for better visualization
                plt.imshow(X_plot, cmap='viridis', vmin = -2, vmax = 2)
                plt.gca().invert_yaxis()
                plt.gca().axis('off')
                plt.title(f'Eb Sample {idx}')
                # plt.colorbar()
            for i, idx in enumerate(random_indices):
                plt.subplot(2, 5, i + 6)
                
                Y_plot = Y[:, :, 0, idx].copy()  
                Y_plot[mask == 0] = np.nan
                
                plt.imshow(Y_plot, cmap='viridis', vmin=-2, vmax=2)
                plt.gca().invert_yaxis()
                plt.gca().axis('off')
                plt.title(f'Na Sample {idx}')

            plt.tight_layout()

        self.X_ori = X.reshape((self.nx * self.ny * self.n_channel, self.n_samples_total)).T
        self.Y_ori = Y.reshape((self.nx * self.ny * self.n_channel, self.n_samples_total)).T
        return 
    
    def reduce(self, n_component_x, n_component_y):
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
        self.X_reduced = pca_x.fit_transform(self.X_ori)
        self.Y_reduced = pca_y.fit_transform(self.Y_ori)
        self.pca_x = pca_x
        self.pca_y = pca_y

        self.ndim_reduced_total = n_component_x + n_component_y
        self.ndim_reduced_x     = n_component_x
        self.ndim_reduced_y     = n_component_y
        return
    
    def data_split(self, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
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
    def train_gmm(self, n_components):
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

    def plot_gmm_samples(self, n_samples=3):
        # test first three samples from the test set and visualize the prediction
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
            x_uq_samples_ori = np.zeros((256*256, n_uq_sample))
            # Inverse transform to original space
            # then the uq samples
            for j, sample in enumerate(x_uq_samples):
                x_uq_samples_ori[:,j] = self.pca_x.inverse_transform(sample)

            # compute the mean from the ensemble
            x_pred_mean = np.mean(x_uq_samples_ori, axis=1)
            x_pred_img = x_pred_mean.reshape(256, 256)
            
            # compute std along each dimension of the uq samples
            x_uq_std = np.std(x_uq_samples_ori, axis=1)
            x_uq_std = x_uq_std.reshape(256, 256)

            # Reshape X for visualization
            x_test_original = self.pca_x.inverse_transform(x_test.reshape(1, -1))
            x_test_img = x_test_original.reshape(256, 256)

            # observed Y
            obs_Y = self.pca_y.inverse_transform(y_test)
            obs_Y = obs_Y.reshape(256, 256)
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