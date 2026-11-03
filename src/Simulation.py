import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from gmr import GMM


class Simulation:
    """
    A class carrying the physics-based simulation information
    This includes its nondimensional data and PCA-reduced representation.
    """
    def __init__(self, extent=None, coord=None):
        self.extent = extent
        self.coord = coord
        self.X_grid, self.Y_grid = np.meshgrid(coord[0], coord[1])
        xx = coord[0].ravel()
        yy = coord[1].ravel()   
        self.delta_x = xx[1] - xx[0]
        self.delta_y = yy[1] - yy[0]

        # in our naming convention, '_ori' means standardized but not reduced (original dimension)
        # 'reduced' means the dimension has been reduced by PCA.
        self.X_ori = None

        # mask
        self.flight_mask = None
        self.domain_mask = None

        # random state for reproducibility
        self.random_state = check_random_state(42)

    def load_sim_data(self, X, labels, domain_mask = None, flight_mask=None):
        """ 
        Load the simulation data without any training splitting. We assume that these data have been standardized.
        
        Parameters
        ----------
        X: list of arrays
            The input features. Each list should have the shape (n_samples, nx, ny)
        labels: list of strings
            The labels corresponding to each array in X.
        domain_mask: ndarray of shape (n_samples, nx, ny), optional
            Boolean mask indicating the valid data point in the simulation domain (continuous).
        flight_mask: ndarray of shape (n_samples, nx, ny), optional
            Boolean mask indicating the valid data point in a flight domain (discrete flight tracks).
        """

    
        self.n_samples_total, self.nx, self.ny = X[0].shape[0], X[0].shape[1], X[0].shape[2]
        assert all(
            X[i].shape[0] == self.n_samples_total and 
            X[i].shape[1] == self.nx and 
            X[i].shape[2] == self.ny for i in range(len(X))), \
                "All arrays in X must have the same shape"

        self.ndim_ori = self.nx * self.ny
        self.n_fields    = len(X)
        self.labels      = labels
        self.domain_mask = domain_mask
        self.flight_mask = flight_mask

        self.X_ori = [X[i].reshape((
                        self.n_samples_total, self.nx * self.ny)) 
                        for i in range(self.n_fields)]

        # standardize the data with relaxation
        eps_relax = 0.05 # 5 percentile 
        self.X_mean = [np.mean(self.X_ori[i], axis=0) for i in range(self.n_fields)]
        self.X_std  = [np.std(self.X_ori[i], axis=0) for i in range(self.n_fields)]
        self.X_eps  = [eps_relax * self.X_mean[i] for i in range(self.n_fields)]

        # sklearn standard scaler
        scaling_models = [StandardScaler() for _ in range(self.n_fields)]
        self.X_norm = [scaling_models[i].fit_transform(self.X_ori[i]) for i in range(self.n_fields)]
        self.scaling_models = scaling_models


        # get domain boundary line
        if domain_mask is not None:
            domain_bound_line = self._get_domain_outline(domain_mask)
            # # flip y wrt to the midpoint of the domain extent to allow origin='upper' in imshow
            # y_mid = (self.extent[2] + self.extent[3]) / 2 
            # corrected_y = 2 * y_mid - domain_bound_line[1]
            self.domain_bound_line = domain_bound_line

        return 

    def show_data(self, n_samples_to_plot=5):
        colormap = 'RdBu'

        random_indices = np.random.choice(
            self.n_samples_total, 
            size=n_samples_to_plot, 
            replace=False
        )
        # figure size
        plt.figure(figsize=(15, 8))
        for i, idx in enumerate(random_indices):

            for j, idx_field in enumerate(range(self.n_fields)):

                plt.subplot(len(self.X_norm), 
                            n_samples_to_plot, 
                            j * n_samples_to_plot + i + 1
                            )
                X_plot = self.X_norm[idx_field][idx, :].reshape((
                            self.nx, 
                            self.ny, 
                            )).copy()
                if self.domain_mask is not None:
                    X_plot[self.domain_mask == False] = np.nan # set the values outside the model boundary to NaN for better visualization
                plt.imshow(X_plot, cmap=colormap, vmin = -2, vmax = 2)
                plt.gca().invert_yaxis()
                plt.gca().axis('off')
                plt.title(f'{self.labels[idx_field]} Sample {idx}')

        # add a single color bar
        cbar_ax = plt.gcf().add_axes([1.05, 0.15, 0.02, 0.7])
        norm = plt.Normalize(vmin=-2, vmax=2)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        # add label
        cbar_ax.set_ylabel('Normalized Value', rotation=270, labelpad=15)
        sm.set_array([])
        plt.colorbar(sm, cax=cbar_ax)

        plt.tight_layout()


    def find_reduction_model_pca(self, n_latent, show_scree=False):
        """ 
        Reduce the dimensionality of the input and output data using PCA.
        
        Parameters
        ----------
        n_latent: list of integers
            The number of principal components to compute for each field in X.
        """
        self._input_list_check(n_latent)

        self.X_reduced = []
        self.pca_x = []
        for i, n_latent_i in enumerate(n_latent):
            pca_x_i = PCA(n_components=n_latent_i)
            self.X_reduced.append(pca_x_i.fit_transform(self.X_norm[i]))
            self.pca_x.append(pca_x_i)

        self.ndim_reduced = [n_latent_i for n_latent_i in n_latent]

        if show_scree:
            plt.figure(figsize=(8,6))
            for i, pca_x_i in enumerate(self.pca_x):
                cum_var = np.cumsum(pca_x_i.explained_variance_ratio_)
                plt.plot(
                    np.arange(1, self.ndim_reduced[i]+1), 
                    cum_var, 
                    'o-',
                    label=self.labels[i]
                    )
                plt.xlabel('Principal Component')
                plt.ylabel('Cumulative Explained Variance Ratio')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()

        return

    def split_data(self, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
        """ 
        Split the data into training, validation, and test sets. This function
        operates on the reduced X data (self.X_reduced).
        
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

        indices = np.arange(self.n_samples_total)
        self.random_state.shuffle(indices)

        self.X_train = []
        self.X_validation = []
        self.X_test = []
        print(f"Splitting data: Total samples = {self.n_samples_total}")
        for i in range(self.n_fields):
            self.X_train.append(self.X_reduced[i][indices[:n_train], :])
            self.X_validation.append(self.X_reduced[i][indices[n_train:n_train + n_validation], :])
            self.X_test.append(self.X_reduced[i][indices[n_train + n_validation:], :])
            print(f"-------- Field {i}: ")
            print(f"    Shape of X_train[{i}]:", self.X_train[i].shape)
            print(f"    Shape of X_validation[{i}]:", self.X_validation[i].shape)
            print(f"    Shape of X_test[{i}]:", self.X_test[i].shape)

        print(f"-----------------------------")
        print(f"-----------------------------")

        return

    def gmm_prior(self, n_gaussians):
        """
        Fit a Gaussian Mixture Model (GMM) to the reduced X data as a prior.

        Parameters
        ----------
        n_gaussians: list of int
            The number of Gaussian components in the mixture for each field.
        """

        self._input_list_check(n_gaussians)
        
        gmms = []
        print("Fitting GMMs for each field")
        for i, n_gaussian in zip(range(self.n_fields), n_gaussians):
            print(f"--------- Field {i}, number of Gaussians: {n_gaussian}")
            gmm = GMM(
                n_components=n_gaussian, 
                random_state=self.random_state,
                )

            gmm.from_samples(
                self.X_train[i], 
                R_diff=1e-6,
                )
            gmms.append(gmm)
        self.gmms = gmms
        return

    def _input_list_check(self, X):
        """
        Check if the input is a list with the correct length.
        """
        if not isinstance(X, list):
            raise ValueError("Input must be a list.")
        if len(X) != self.n_fields:
            raise ValueError("Input list must have the same length as the number of fields.")
    def _get_domain_outline(self, mask):
        """
        Extract the outline of a boolean domain mask as x, y coordinates.

        Parameters
        ----------
        mask : 2D boolean array, shape (ny, nx)

        Returns
        -------
        segments : list of (x, y) coordinate arrays, one per contour segment
        """
        ny, nx = mask.shape

        xmin, xmax, ymin, ymax = self.extent
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)

        X, Y = np.meshgrid(x, y)

        fig_tmp, ax_tmp = plt.subplots()
        try:
            cs = ax_tmp.contour(
                X, Y, mask.astype(float), levels=[0.5]
            )

            segments = [
                (verts[:, 0].copy(), verts[:, 1].copy())
                for verts in cs.allsegs[0]
                if len(verts) > 0
            ]
        finally:
            plt.close(fig_tmp)

        return segments

    # def load_split_data(
    #         self, 
    #         X_train, 
    #         Y_train,
    #         X_validation, 
    #         Y_validation,
    #         X_test, 
    #         Y_test
    #     ):
    #     """ 
    #     Load the pre-split data into the model. This is an alternative to the split_data method.
        
    #     Parameters
    #     ----------
    #     X_train: ndarray of shape (n_samples_train, n_features)
    #         The training data in the original space.
    #     Y_train: ndarray of shape (n_samples_train, n_targets)
    #         The training target data in the original space.
    #     X_validation: ndarray of shape (n_samples_validation, n_features)
    #         The validation data in the original space.
    #     Y_validation: ndarray of shape (n_samples_validation, n_targets)
    #         The validation target data in the original space.
    #     X_test: ndarray of shape (n_samples_test, n_features)
    #         The test data in the original space.
    #     Y_test: ndarray of shape (n_samples_test, n_targets)
    #         The test target data in the original space.
    #     """
    #     # Apply fitted PCA on the whole data set to 
    #     # the training, validation, and test data separately        
    #     def flatten(X):
    #         return X.reshape((self.nx * self.ny * self.n_channel, -1)).T

    #     # ✅ DO NOT refit — pca_x and pca_y are already fitted in find_reduction_model_pca
    #     # Just transform each split
    #     X_reduced_train      = self.pca_x.transform(flatten(X_train))
    #     Y_reduced_train      = self.pca_y.transform(flatten(Y_train))

    #     if Y_validation.shape[3] > 0:
    #         X_reduced_validation = self.pca_x.transform(flatten(X_validation))
    #         Y_reduced_validation = self.pca_y.transform(flatten(Y_validation))
    #     else:
    #         X_reduced_validation = np.empty((0, self.pca_x.n_components_))
    #         Y_reduced_validation = np.empty((0, self.pca_y.n_components_))

    #     X_reduced_test = self.pca_x.transform(flatten(X_test))
    #     Y_reduced_test = self.pca_y.transform(flatten(Y_test))

    #     # Stack (Y first, then X — matching your index convention)
    #     self.XY_train      = np.hstack((Y_reduced_train,      X_reduced_train))
    #     self.XY_validation = np.hstack((Y_reduced_validation, X_reduced_validation))
    #     self.XY_test       = np.hstack((Y_reduced_test,       X_reduced_test))

    #     self.n_samples_train      = self.XY_train.shape[0]
    #     self.n_samples_validation = self.XY_validation.shape[0]
    #     self.n_samples_test       = self.XY_test.shape[0]

    #     print("Shape of XY_train:", self.XY_train.shape)
    #     print("Shape of XY_validation:", self.XY_validation.shape)
    #     print("Shape of XY_test:", self.XY_test.shape)
    #     return

    # def load_standardization_data(self, X_mean, X_std, Y_mean, Y_std, X_epsilon=None, Y_epsilon=None):
    #     """
    #     Load the standardization data (mean, std) from the simulation ensemble
    #       -> Going between standardized space and the physical space
    #     """
    #     # shape check: the mean and std should all be flatten
    #     if X_mean.shape != (self.nx * self.ny * self.n_channel,):
    #         raise ValueError(f"X_mean should have shape {(self.nx * self.ny * self.n_channel,)}, but got {X_mean.shape}")
    #     if X_std.shape != (self.nx * self.ny * self.n_channel,):
    #         raise ValueError(f"X_std should have shape {(self.nx * self.ny * self.n_channel,)}, but got {X_std.shape}")
    #     if Y_mean.shape != (self.nx * self.ny * self.n_channel,):
    #         raise ValueError(f"Y_mean should have shape {(self.nx * self.ny * self.n_channel,)}, but got {Y_mean.shape}")
    #     if Y_std.shape != (self.nx * self.ny * self.n_channel,):
    #         raise ValueError(f"Y_std should have shape {(self.nx * self.ny * self.n_channel,)}, but got {Y_std.shape}")
    #     self.X_mean = X_mean
    #     self.X_std = X_std
    #     self.Y_mean = Y_mean
    #     self.Y_std = Y_std
    #     self.X_epsilon = X_epsilon
    #     self.Y_epsilon = Y_epsilon
    #     return 
