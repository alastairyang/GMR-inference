import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
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

    def show_data(self, n_samples_to_plot=5, savepath=None):
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
        cbar_ax.tick_params(labelsize=16)
        sm.set_array([])
        plt.colorbar(sm, cax=cbar_ax)

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)


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

    def show_mode(self, modes = None, savepath = None):
        """
        Show principal components.

        Parameters
        ----------
        modes: list of int, optional
            The indices of the principal components to show. If None, show every 10th component.
        savepath: str, optional
            Path to save the figure. If None, the figure is not saved.
        """
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap='RdBu_r')
        
        if modes is None:
            modes = range(0, self.ndim_reduced[0], 10)

        for i_field in range(0, self.n_fields):
            plt.figure(figsize=(18, 8))
            for idx, i in enumerate(modes):
                vec = self.pca_x[i_field].components_[i].flatten()
                vec = np.where(self.domain_mask.flatten(), vec, np.nan)
                plt.subplot(2, len(modes) // 2 + len(modes) % 2, idx + 1)
                plt.imshow(vec.reshape(self.nx, self.ny), 
                           cmap='RdBu_r',
                           vmin=-0.02, vmax=0.02,
                           origin='lower'
                           )
                plt.title(f"Mode {i}")
                plt.axis('off')

            # add a color bar to the right of the figure
            cbar = plt.colorbar(sm, 
                                ax=plt.gcf().get_axes(), 
                                orientation='vertical', 
                                fraction=0.02, pad=0.06)
            cbar.set_label('Mode Amplitude', fontsize=15)
            # only show -1, 0, 1 ticks
            cbar.set_ticks([-1, 0, 1])
            
            if savepath is not None:
                savepath = savepath.replace(".png", f"_field{i_field}.png")
                plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
            plt.show()

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

    def fit_gmm_prior(self, n_gaussians):
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
    def fit_joint_gmm(self, n_gaussians, index):
        """
        Fit a Gaussian Mixture Model (GMM) to the reduced X data (joint of multiple fields)
 

        Parameters
        ----------
        n_gaussians: list of int
            The number of Gaussian components in the mixture for each field.
        index: nested list of int, optional
            The indices of the joint fields to fit the GMM for. If None, fit for all fields.
        """

        if index is not None:
            if len(n_gaussians) != len(index):
                raise ValueError("Length of n_gaussians must match the length of index.")
        
        gmms = []
        print("Fitting GMMs for each field")
        if index is None:
            index = list(range(self.n_fields))
        for idx, n_gaussian in zip(index, n_gaussians):
            print(f"--------- Field {idx}, number of Gaussians: {n_gaussian}")
            gmm = GMM(
                n_components=n_gaussian, 
                random_state=self.random_state,
                )
            train_data = np.concatenate([self.X_train[j] for j in idx], axis=1)
            gmm.from_samples(
                train_data, 
                R_diff=1e-6,
                )
            gmms.append(gmm)
        
        return gmms

    def fit_pls(self, x_idx, y_idx, n_components=2, scale=True):
        """
        Fit PLS regression between the PCA coefficients of two fields.

        Parameters
        ----------
        x_idx : int
            Index of the known field (predictor).
        y_idx : int
            Index of the unknown field (target).
        n_components : int, default=2
            Number of PLS latent components. This is independent of the
            number of PCA components retained for either field.
        scale : bool, default=True
            Whether to scale predictor and target PCA coefficients to unit
            variance before fitting. PLS always centers the data.

        Returns
        -------
        pls : sklearn.cross_decomposition.PLSRegression
            Fitted model. Predictions are in the target field's PCA space.
        """
        if not hasattr(self, "X_train"):
            raise RuntimeError(
                "Call find_reduction_model_pca() and split_data() before fit_pls()."
            )

        for name, idx in (("x_idx", x_idx), ("y_idx", y_idx)):
            if not isinstance(idx, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
            if not 0 <= idx < self.n_fields:
                raise ValueError(
                    f"{name} must be between 0 and {self.n_fields - 1}."
                )

        X = self.X_train[x_idx]
        Y = self.X_train[y_idx]

        if X.shape[0] < 2:
            raise ValueError("PLS regression requires at least two training samples.")

        max_components = min(X.shape[0], X.shape[1])
        if not isinstance(n_components, (int, np.integer)):
            raise TypeError("n_components must be an integer.")
        if not 1 <= n_components <= max_components:
            raise ValueError(
                f"n_components must be between 1 and {max_components}."
            )

        pls = PLSRegression(
            n_components=n_components,
            scale=scale,
        )
        pls.fit(X, Y)

        self.pls_model = pls
        self.pls_x_idx = x_idx
        self.pls_y_idx = y_idx

        return pls

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
