import numpy as np

def shape_check(*arrays):
    """  
    Check that all input arrays have the same shape.

    Parameters:
    -------
    *arrays: list of arrays
        list of arrays to check

    Raises:
    -------
    ValueError: if any two arrays have different shapes
    """
    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) > 1:
        raise ValueError("All input arrays must have the same shape, but got shapes: {}".format(shapes))

def reverse_standardize(X, mean, std, method='standard', epsilon=None):
    """
    reverse z-score standardization (standard)
    or with relaxation 

    Parameters:
-------
    X: array, shape (n_samples, n_features)
        standardized data to be reverse standardized
    mean: array, shape (n_features,)
        mean used for standardization
    std: array, shape (n_features,)
        std used for standardization
    epsilon: float, optional
        small value added to std for relaxation method
    method: str
        method for reverse standardization, 'standard' or 'relaxation'

    """
    # first check X, mean, and std have the same shape
    # or it will do outer product and our computer will EXPLODE SIR
    shape_check(X, mean, std)
    if method == 'standard':
        return X * std + mean
    elif method == 'relaxation':
        if epsilon is None:
            raise ValueError("Epsilon must be provided for relaxation method")
        else:
            return X * (std + epsilon) + mean
    else:
        raise ValueError("Unknown method: {}".format(method))
    
def standardize(X, mean, std, method='standard', epsilon=None):
    """
    z-score standardization (standard)
    or with relaxation 

    Parameters:
    -------
    X: array, shape (n_samples, n_features)
        data to be standardized
    mean: array, shape (n_features,)
        mean used for standardization   

    std: array, shape (n_features,)
        std used for standardization
    epsilon: float, optional
        small value added to std for relaxation method
    method: str
        method for standardization, 'standard' or 'relaxation'


    """
    # first check X, mean, and std have the same shape
    # or it will do outer product and our computer will EXPLODE SIR
    shape_check(X, mean, std)
    if method == 'standard':
        print("Using standard z-score standardization")
        return (X - mean) / std
    elif method == 'relaxation':
        if epsilon is None:
            raise ValueError("Epsilon must be provided for relaxation method")
        else:
            print("Using relaxation z-score standardization with epsilon =", epsilon)
            return (X - mean) / (std + epsilon)
    else:
        raise ValueError("Unknown method: {}".format(method))
# ---------- NOT ACTIVELY USED ----------
def build_distance_matrix(x, y):
    """
    Build a distance matrix

    Parameters:
    -------
    x: array, shape (n,)
        1D array of x coordinates
    y: array, shape (n,)
        1D array of y coordinates

    Returns:
    -------
    dist_mtx: array, shape (n, n)
        distance matrix where dist_mtx[i,j] is the distance between (x[i], y[i]) and (x[j], y[j])
    
    """
    n = len(x)
    dist_mtx = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_mtx[i, j] = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
    return dist_mtx


def spatial_density_weighting(coord):
    """
    Compute spatial and density weights for the evidence points based on their coordinates and pairwise distances.
    Originally intended to find a weighting scheme for the evidence poitns
    to reduce overcounting the closely spaced points

    Parameters:
    -------
    coord: list of arrays
        list of coordinate arrays (e.g., [x, y]) for the evidence points
    """
    from scipy.stats import gaussian_kde
    # Normalize both to [0, 1]
    def minmax(x):
        return (x - x.min()) / (x.max() - x.min())

    coords = np.vstack(coord)
    kde = gaussian_kde(coords)
    density = kde(coords)  # estimated density at each point

    density_weight_kde = 1.0 / (density + 1e-10)
    density_weight_kde = minmax(density_weight_kde)

    return density_weight_kde


def build_decorr_weight_matrix(X_vec, combined_idx):
    """   
    Build a decorrelation weight matrix based covariance (similarity measure)

    Parameters:
    -------
    X_vec: array, shape (n_samples, n_features)
        Input data vectors
    combined_idx: array, shape (n_samples,)
        Indices of the combined data points

    Returns:
    -------
    weight_mtx: array, shape (n_samples, n_samples)
        Decorrelation weight matrix
    sim_mtx: array, shape (n_samples, n_samples)
        Similarity matrix
    """
    def redundancy_weights(sim_mtx):
        sim_abs = np.abs(sim_mtx)
        sim_min = sim_abs.min()
        sim_max = sim_abs.max()
        sim_norm = (sim_abs - sim_min) / (sim_max - sim_min)
        return 1.0 - sim_norm

    n = combined_idx.shape[0]
    combined_idx_mtx = np.zeros((n, n))
    sim_mtx = np.zeros((n, n))

    for loc1, idx1 in enumerate(combined_idx):   # idx1 = actual spatial index
        data_idx1 = X_vec[idx1]                  # ← correct: index by VALUE
        remaining = combined_idx[loc1:]           # ← renamed to avoid collision

        combined_idx_mtx[loc1, loc1:loc1+len(remaining)] = remaining.flatten()


        for loc2, idx2 in enumerate(remaining):
            data_idx2 = X_vec[idx2]
            covariance = np.cov(data_idx1, data_idx2)[0, 1]
            sim_mtx[loc1, loc1+loc2] = covariance
            sim_mtx[loc1+loc2, loc1] = covariance

    weight_mtx = redundancy_weights(sim_mtx)
    return weight_mtx, sim_mtx