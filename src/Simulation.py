import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
        self.Y_ori = None
        # standardization data (from simulation ensemble)
        self.X_mean = None 
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self.X_epsilon = None # relaxed standardization
        self.Y_epsilon = None # relaxed standardization

        # mask
        self.flight_mask = None
        self.domain_mask = None

        # join distribution data
        self.XY_train       = None
        self.XY_validation  = None
        self.XY_test        = None

        # models
        self.pca_y    = None
        self.pca_x    = None

        # dimension and indices
        self.nx = None
        self.ny = None
        self.x_indices = None # Eb_star indices
        self.y_indices = None # Na_star indices
        self.ndim_ori     = None
        self.ndim_reduced_total = None
        self.ndim_reduced_x     = None
        self.ndim_reduced_y     = None
        self.n_channel = 1
        self.n_samples_total      = None
        self.n_samples_validation = None
        self.n_samples_test       = None
        self.n_samples_train      = None

    def load_sim_data(self, X, Y, domain_mask = None, flight_mask=None, show_plot=True, x_name='X', y_name='Y'):
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
        colormap = 'RdBu'

        if show_plot:
            random_indices = np.random.choice(self.n_samples_total, size=5, replace=False)
            # figure size
            plt.figure(figsize=(9, 5))
            for i, idx in enumerate(random_indices):
                plt.subplot(2, 5, i + 1)
                X_plot = X[:, :, :, idx].copy()
                if domain_mask is not None:
                    X_plot[domain_mask == False] = np.nan # set the values outside the model boundary to NaN for better visualization
                plt.imshow(X_plot, cmap=colormap, vmin = -2, vmax = 2)
                plt.gca().invert_yaxis()
                plt.gca().axis('off')
                plt.title(f'{x_name} Sample {idx}')
                # plt.colorbar()
            for i, idx in enumerate(random_indices):
                plt.subplot(2, 5, i + 6)
                Y_plot = Y[:, :, 0, idx].copy()  
                if domain_mask is not None:
                    Y_plot[domain_mask == False] = np.nan
                
                plt.imshow(Y_plot, cmap=colormap, vmin=-2, vmax=2)
                plt.gca().invert_yaxis()
                plt.gca().axis('off')
                plt.title(f'{y_name} Sample {idx}')

            plt.tight_layout()

        self.X_ori = X.reshape((self.nx * self.ny * self.n_channel, self.n_samples_total)).T
        self.Y_ori = Y.reshape((self.nx * self.ny * self.n_channel, self.n_samples_total)).T

        self.domain_mask = domain_mask
        self.flight_mask = flight_mask
        # get domain boundary line
        domain_bound_line = self._get_domain_outline(domain_mask)[0]
        # flip y wrt to the midpoint of the domain extent to allow origin='upper' in imshow
        y_mid = (self.extent[2] + self.extent[3]) / 2 
        corrected_y = 2 * y_mid - domain_bound_line[1]
        self.domain_bound_line = (domain_bound_line[0], corrected_y)

        return 

    def _get_domain_outline(self, mask):
        """
        Extract the outline of a boolean domain mask as x, y coordinates.
        
        Parameters
        ----------
        mask   : 2D boolean array, shape (ny, nx)
        
        Returns
        -------
        segments : list of (x, y) coordinate arrays, one per contour path
        """
        ny, nx = mask.shape

        xmin, xmax, ymin, ymax = self.extent
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)


        X, Y = np.meshgrid(x, y)

        # Use contour at level 0.5 to find the boolean boundary
        fig_tmp, ax_tmp = plt.subplots()
        cs = ax_tmp.contour(X, Y, mask.astype(float), levels=[0.5])
        plt.close(fig_tmp)  # don't display the temp figure

        segments = []
        for path in cs.collections[0].get_paths():
            verts = path.vertices
            segments.append((verts[:, 0], verts[:, 1]))
            

        return segments



    def load_split_data(
            self, 
            X_train, 
            Y_train,
            X_validation, 
            Y_validation,
            X_test, 
            Y_test
        ):
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

        # ✅ DO NOT refit — pca_x and pca_y are already fitted in find_reduction_model_pca
        # Just transform each split
        X_reduced_train      = self.pca_x.transform(flatten(X_train))
        Y_reduced_train      = self.pca_y.transform(flatten(Y_train))

        if Y_validation.shape[3] > 0:
            X_reduced_validation = self.pca_x.transform(flatten(X_validation))
            Y_reduced_validation = self.pca_y.transform(flatten(Y_validation))
        else:
            X_reduced_validation = np.empty((0, self.pca_x.n_components_))
            Y_reduced_validation = np.empty((0, self.pca_y.n_components_))

        X_reduced_test = self.pca_x.transform(flatten(X_test))
        Y_reduced_test = self.pca_y.transform(flatten(Y_test))

        # Stack (Y first, then X — matching your index convention)
        self.XY_train      = np.hstack((Y_reduced_train,      X_reduced_train))
        self.XY_validation = np.hstack((Y_reduced_validation, X_reduced_validation))
        self.XY_test       = np.hstack((Y_reduced_test,       X_reduced_test))

        self.n_samples_train      = self.XY_train.shape[0]
        self.n_samples_validation = self.XY_validation.shape[0]
        self.n_samples_test       = self.XY_test.shape[0]

        print("Shape of XY_train:", self.XY_train.shape)
        print("Shape of XY_validation:", self.XY_validation.shape)
        print("Shape of XY_test:", self.XY_test.shape)
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

    def find_reduction_model_pca(self, n_component_x, n_component_y):
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
        print("PCA model saved to self.pca_x and self.pca_y.")

        self.ndim_reduced_total = n_component_x + n_component_y
        self.ndim_reduced_x     = n_component_x
        self.ndim_reduced_y     = n_component_y

        # add x and y indices: (first y indices, then x indices)
        self.y_indices = np.arange(self.ndim_reduced_y)
        self.x_indices = np.arange(self.ndim_reduced_y, self.ndim_reduced_total)
        return