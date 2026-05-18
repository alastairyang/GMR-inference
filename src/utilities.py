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
def shape_check(X, mean, std):
    """
    Checks that mean and std are compatible with X for broadcasting.
    Allows:
      - X shape (n_features,)     with mean/std shape (n_features,)
      - X shape (N, n_features)   with mean/std shape (n_features,)
      - X shape (N, n_features)   with mean/std shape (N, n_features)
    """
    if mean.shape != std.shape:
        raise ValueError(
            f"mean and std must have the same shape. "
            f"Got mean={mean.shape}, std={std.shape}"
        )
    # Allow exact match or broadcasting along first dimension
    if X.shape != mean.shape and X.shape[-1] != mean.shape[-1]:
        raise ValueError(
            f"Shape mismatch: X={X.shape} is not compatible with "
            f"mean/std={mean.shape}. Expected X.shape[-1] == mean.shape[-1]."
        )


def reverse_standardize(X, mean, std, method='standard', epsilon=None):
    """
    Reverse z-score standardization.
    Vectorized across the first dimension of X.

    Parameters
    ----------
    X : array, shape (n_features,) or (N, n_features)
    mean : array, shape (n_features,)
    std  : array, shape (n_features,)
    method : 'standard' or 'relaxation'
    epsilon : float, required if method='relaxation'
    """
    shape_check(X, mean, std)

    if method == 'standard':
        return X * std + mean

    elif method == 'relaxation':
        if epsilon is None:
            raise ValueError("Epsilon must be provided for relaxation method")
        return X * (std + epsilon) + mean

    else:
        raise ValueError(f"Unknown method: {method}")


def standardize(X, mean, std, method='standard', epsilon=None):
    """
    Z-score standardization.
    Vectorized across the first dimension of X.

    Parameters
    ----------
    X : array, shape (n_features,) or (N, n_features)
    mean : array, shape (n_features,)
    std  : array, shape (n_features,)
    method : 'standard' or 'relaxation'
    epsilon : float, required if method='relaxation'
    """
    shape_check(X, mean, std)

    if method == 'standard':
        return (X - mean) / std

    elif method == 'relaxation':
        if epsilon is None:
            raise ValueError("Epsilon must be provided for relaxation method")
        return (X - mean) / (std + epsilon)

    else:
        raise ValueError(f"Unknown method: {method}")

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

def low_high_percentile(samples, log_probs, low_percentile=5, high_percentile=95):
    """
    Compute the low and high percentiles of the samples based on their log probabilities.

    Parameters:
    -------
    samples: array, shape (n_samples, n_features)
        MCMC samples
    log_probs: array, shape (n_samples,)
        Log probabilities of the samples
    low_percentile: float
        Percentile for the lower bound (default: 5)
    high_percentile: float
        Percentile for the upper bound (default: 95)

    Returns:
    -------
    low_bound: array, shape (n_features,)
        Low percentile bound of the samples
    high_bound: array, shape (n_features,)
        High percentile bound of the samples
    """
    # Sort samples by log probability (ascending)
    sorted_indices = np.argsort(log_probs) 
    samples_sorted = samples[sorted_indices]
    log_probs_sorted = log_probs[sorted_indices]

    lp_low  = np.percentile(log_probs_sorted, low_percentile)
    lp_high = np.percentile(log_probs_sorted, high_percentile)

    # the single representative samples closest to each threshold
    idx_low  = np.argmin(np.abs(log_probs_sorted - lp_low))
    idx_high = np.argmin(np.abs(log_probs_sorted - lp_high))

    low_bound = samples_sorted[idx_low]  
    high_bound = samples_sorted[idx_high]  
    return low_bound, high_bound

def read_exp(filepath):
    """
    Parse an Elmer/Ice .exp file.
    Handles multi-segment files (segments separated by NaN NaN rows).
    Returns x, y arrays for the first (or only) segment, with NaNs stripped.
    """
    xs, ys = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # skip header/comment lines
            if not line or line.startswith('#') or line.startswith('!'):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                x_val, y_val = float(parts[0]), float(parts[1])
                xs.append(x_val)
                ys.append(y_val)
            except ValueError:
                continue

    xs = np.array(xs)
    ys = np.array(ys)

    # strip NaN rows (segment separators)
    valid = ~(np.isnan(xs) | np.isnan(ys))
    return xs[valid], ys[valid]